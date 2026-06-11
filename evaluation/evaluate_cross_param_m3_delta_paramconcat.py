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
        description="Evaluate M3-Delta-ParamConcat on cross-parameter test split."
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


def append_param_channels(batch_x_norm, batch_param):
    """
    batch_x_norm: [B, 16, H, W]
    batch_param:  [B, 2] = [log10(Ra), log10(Pr)]

    拼接 4 个参数通道:
        log10(Ra)
        log10(Pr)
        log10(nu)
        log10(kappa)

    其中:
        log10(kappa) = -0.5 * (log10(Ra) + log10(Pr))
        log10(nu)   = -0.5 * (log10(Ra) - log10(Pr))

    返回:
        batch_x_param: [B, 20, H, W]
    """

    if batch_x_norm.ndim != 4:
        raise ValueError(f"❌ batch_x_norm 应为 [B, C, H, W]，但得到 {batch_x_norm.shape}")

    B, C, H, W = batch_x_norm.shape

    if C != 16:
        raise ValueError(f"❌ 原始输入通道应为 16，但得到 {C}")

    if batch_param.ndim != 2 or batch_param.shape[1] != 2:
        raise ValueError(
            f"❌ batch_param 应为 [B, 2] = [log10(Ra), log10(Pr)]，但得到 {batch_param.shape}"
        )

    if batch_param.shape[0] != B:
        raise ValueError(
            f"❌ batch size 不匹配: batch_x B={B}, batch_param B={batch_param.shape[0]}"
        )

    batch_param = batch_param.to(device=batch_x_norm.device, dtype=batch_x_norm.dtype)

    log_ra = batch_param[:, 0]
    log_pr = batch_param[:, 1]

    log_nu = -0.5 * (log_ra - log_pr)
    log_kappa = -0.5 * (log_ra + log_pr)

    param_vec = torch.stack(
        [log_ra, log_pr, log_nu, log_kappa],
        dim=1
    )

    param_channels = param_vec.view(B, 4, 1, 1).expand(B, 4, H, W)

    batch_x_param = torch.cat([batch_x_norm, param_channels], dim=1)

    if batch_x_param.shape[1] != 20:
        raise ValueError(f"❌ 拼接参数后输入通道应为 20，但得到 {batch_x_param.shape[1]}")

    return batch_x_param


def main():
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    split_path = resolve_path(project_root, args.split)
    stats_path = resolve_path(project_root, args.stats)
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    output_path = resolve_path(project_root, args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 [Cross-Param M3-Delta-ParamConcat Evaluation]")
    print(f"📌 Device: {device}")
    print(f"📌 Split: {split_path}")
    print(f"📌 Stats: {stats_path}")
    print(f"📌 Checkpoint: {checkpoint_path}")
    print(f"📌 Output: {output_path}")
    print("👉 Input channels: 20 = 16 history channels + 4 parameter channels")
    print("👉 Parameter channels: log10(Ra), log10(Pr), log10(nu), log10(kappa)")

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
        stats_path=stats_path,
        return_params=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"📊 Test samples: {len(test_dataset)}")

    # 检查一个 batch
    sample_x, sample_y, sample_param = next(iter(test_loader))
    print(f"✅ 原始 X shape: {sample_x.shape}，应为 [B, 16, 256, 64]")
    print(f"✅ Y shape:      {sample_y.shape}，应为 [B, 4, 256, 64]")
    print(f"✅ param shape:  {sample_param.shape}，应为 [B, 2]")
    print(f"👉 示例 param = [log10(Ra), log10(Pr)] = {sample_param[0].tolist()}")

    sample_x_param = append_param_channels(sample_x, sample_param)
    print(f"✅ ParamConcat X shape: {sample_x_param.shape}，应为 [B, 20, 256, 64]")

    model = PlainFNO2d(
        in_channels=20,
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
        for batch_x_norm, batch_y_norm, batch_param in test_loader:
            batch_x_norm = batch_x_norm.to(device)
            batch_y_norm = batch_y_norm.to(device)
            batch_param = batch_param.to(device)

            # x_norm: [B, 16, H, W]
            # 最后一帧状态是最后 4 个通道
            x_last_norm = batch_x_norm[:, -4:, :, :]

            # 拼接参数通道: [B, 16, H, W] -> [B, 20, H, W]
            batch_x_param = append_param_channels(batch_x_norm, batch_param)

            # 模型输出 normalized delta
            pred_delta_norm = model(batch_x_param)

            # Delta 恢复成下一帧 normalized state
            pred_y_norm = x_last_norm + pred_delta_norm

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

    print("\n================ Cross-Param M3-Delta-ParamConcat Evaluation ================")
    print(f"{'Field':<12} | {'Rel-L2 (%)':>12} | {'MSE':>12}")
    print("-" * 55)

    for i, field in enumerate(FIELD_ORDER):
        rel_l2 = torch.sqrt(field_sse[i] / (field_target_sq[i] + 1e-12)) * 100.0
        mse = field_sse[i] / field_numel[i]

        rows.append({
            "model": "M3-Delta-ParamConcat",
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
        "model": "M3-Delta-ParamConcat",
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