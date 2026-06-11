import os
import json
import torch
import h5py
import numpy as np
import argparse
import sys

# 将上级目录加入环境变量
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import DATA_PATH, FIELD_ORDER


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute field-wise mean/std statistics from the train split only."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="data/splits/iid_split.json",
        help="Path to split json file, relative to project root or absolute path."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/stats/rbc_field_stats.json",
        help="Path to output stats json file, relative to project root or absolute path."
    )

    return parser.parse_args()


def resolve_path(project_root, path):
    """
    如果传入的是绝对路径，直接返回；
    如果传入的是相对路径，则默认相对于项目根目录 DC_MNO。
    """
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


def compute_stats_low_ram(split_file, output_file):
    with open(split_file, 'r', encoding='utf-8') as f:
        full_split_config = json.load(f)
        train_config = full_split_config["train"]

    sums = torch.zeros(4, dtype=torch.float64)
    sum_sqs = torch.zeros(4, dtype=torch.float64)
    total_pixels = 0

    print("🚀 启动低内存流式扫描 (Low-RAM Streaming)...")
    print(f"📌 使用 split 文件: {split_file}")
    print("📌 只使用 train 部分计算统计量，避免 val/test 信息泄漏。")

    with h5py.File(DATA_PATH, 'r') as f:
        for item in train_config:
            group_name = item["group"]
            traj_indices = item["trajectories"]

            if group_name not in f:
                print(f"⚠️ 跳过不存在的工况组: {group_name}")
                continue

            group = f[group_name]
            print(f"正在处理工况组: {group_name} | trajectories={traj_indices}")

            # 每次只取 1 条轨迹进入内存
            for traj_idx in traj_indices:
                traj_data = []

                for field in FIELD_ORDER:
                    # 这里沿用你原来的读取方式：
                    # group[field][traj_idx, :, :, :]
                    # 如果你之前统计脚本已经能正常跑，说明这个索引方式与你的数据格式匹配。
                    field_array = group[field][traj_idx, :, :, :]
                    traj_data.append(field_array)

                # stack 后形状: [4, T, H, W]
                traj_tensor = torch.tensor(
                    np.stack(traj_data, axis=0),
                    dtype=torch.float64
                )

                # 累加统计量
                sums += traj_tensor.sum(dim=(1, 2, 3))
                sum_sqs += (traj_tensor ** 2).sum(dim=(1, 2, 3))

                T, H, W = traj_tensor.shape[1], traj_tensor.shape[2], traj_tensor.shape[3]
                total_pixels += T * H * W

    if total_pixels == 0:
        raise RuntimeError("total_pixels = 0，说明没有成功读取任何 train 数据，请检查 split 文件。")

    mean = sums / total_pixels
    variance = (sum_sqs / total_pixels) - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=1e-12))

    stats_dict = {}

    print("\n✅ 统计完成！train split 场统计量如下：")
    for i, field in enumerate(FIELD_ORDER):
        stats_dict[field] = {
            "mean": mean[i].item(),
            "std": std[i].item(),
        }
        print(f"[{field}] mean={mean[i].item():.6e}, std={std[i].item():.6e}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=4)

    print(f"\n📁 统计量已成功保存至: {output_file}")


if __name__ == "__main__":
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    split_file = resolve_path(project_root, args.split)
    output_file = resolve_path(project_root, args.output)

    compute_stats_low_ram(split_file, output_file)