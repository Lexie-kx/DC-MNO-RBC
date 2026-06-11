import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.rbc_dataset import RBCDataset
from models.operators.fno2d import PlainFNO2d
from constants import FIELD_ORDER


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate M3-State on cross-parameter test split."
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Path to split json file, relative to project root."
    )

    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="Path to field stats json file, relative to project root."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint, relative to project root."
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output csv file, relative to project root."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    return parser.parse_args()


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


def load_stats(stats_path, device):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    means = []
    stds = []

    for field in FIELD_ORDER:
        means.append(stats[field]["mean"])
        stds.append(stats[field]["std"])

    mean = torch.tensor(means, dtype=torch.float32, device=device).view(1, 4, 1, 1)
    std = torch.tensor(stds, dtype=torch.float32, device=device).view(1, 4, 1, 1)

    return mean, std


def denormalize(y_norm, mean, std):
    return y_norm * std + mean


def main():
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    split_path = resolve_path(project_root, args.split)
    stats_path = resolve_path(project_root, args.stats)
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    output_path = resolve_path(project_root, args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 [Cross-Param M3-State Evaluation]")
    print(f"📌 Device: {device}")
    print(f"📌 Split: {split_path}")
    print(f"📌 Stats: {stats_path}")
    print(f"📌 Checkpoint: {checkpoint_path}")
    print(f"📌 Output: {output_path}")

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"找不到 split 文件: {split_path}")

    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"找不到 stats 文件: {stats_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到 checkpoint 文件: {checkpoint_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_config = json.load(f)

    test_dataset = RBCDataset(
        split_config=split_config["test"],
        normalize=True,
        stats_path=stats_path
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"📊 Test samples: {len(test_dataset)}")

    model = PlainFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    mean, std = load_stats(stats_path, device)

    # 累加物理空间误差
    field_sse = torch.zeros(4, dtype=torch.float64, device=device)
    field_target_sq = torch.zeros(4, dtype=torch.float64, device=device)
    field_numel = torch.zeros(4, dtype=torch.float64, device=device)

    global_sse = torch.tensor(0.0, dtype=torch.float64, device=device)
    global_target_sq = torch.tensor(0.0, dtype=torch.float64, device=device)
    global_numel = torch.tensor(0.0, dtype=torch.float64, device=device)

    print("\n🔥 开始 test set 评估...")

    with torch.no_grad():
        for batch_x_norm, batch_y_norm in test_loader:
            batch_x_norm = batch_x_norm.to(device)
            batch_y_norm = batch_y_norm.to(device)

            # M3-State 直接预测下一帧 normalized state
            pred_y_norm = model(batch_x_norm)

            # 反归一化到物理空间
            pred_y_phys = denormalize(pred_y_norm, mean, std)
            true_y_phys = denormalize(batch_y_norm, mean, std)

            diff = pred_y_phys - true_y_phys

            # 每个物理场分别统计
            for c in range(4):
                diff_c = diff[:, c, :, :].double()
                true_c = true_y_phys[:, c, :, :].double()

                field_sse[c] += torch.sum(diff_c ** 2)
                field_target_sq[c] += torch.sum(true_c ** 2)
                field_numel[c] += diff_c.numel()

            global_sse += torch.sum(diff.double() ** 2)
            global_target_sq += torch.sum(true_y_phys.double() ** 2)
            global_numel += diff.numel()

    rows = []

    print("\n================ Cross-Param M3-State Evaluation ================")
    print(f"{'Field':<12} | {'Rel-L2 (%)':>12} | {'MSE':>12}")
    print("-" * 55)

    for i, field in enumerate(FIELD_ORDER):
        rel_l2 = torch.sqrt(field_sse[i] / (field_target_sq[i] + 1e-12)) * 100.0
        mse = field_sse[i] / field_numel[i]

        rows.append({
            "model": "M3-State",
            "field": field,
            "rel_l2_percent": rel_l2.item(),
            "mse": mse.item(),
            "checkpoint": checkpoint_path,
            "split": split_path,
            "stats": stats_path,
        })

        print(f"{field:<12} | {rel_l2.item():12.4f} | {mse.item():12.6e}")

    global_rel_l2 = torch.sqrt(global_sse / (global_target_sq + 1e-12)) * 100.0
    global_mse = global_sse / global_numel

    rows.append({
        "model": "M3-State",
        "field": "global",
        "rel_l2_percent": global_rel_l2.item(),
        "mse": global_mse.item(),
        "checkpoint": checkpoint_path,
        "split": split_path,
        "stats": stats_path,
    })

    print("-" * 55)
    print(f"{'global':<12} | {global_rel_l2.item():12.4f} | {global_mse.item():12.6e}")
    print("==================================================================")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"\n✅ 评估完成，结果已保存至: {output_path}")


if __name__ == "__main__":
    main()