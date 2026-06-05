import os
import sys
import json
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.rbc_dataset import RBCDataset
from models.operators.fno2d import PlainFNO2d
from training.normalization import FieldWiseNormalizer
from training.metrics import (
    compute_field_wise_rel_l2,
    compute_total_rel_l2,
    compute_field_wise_mse,
    compute_total_mse
)


def load_model(checkpoint_name, device):
    model = PlainFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32
    ).to(device)

    ckpt_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'checkpoints',
            'controlled',
            checkpoint_name
        )
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ 加载权重: {checkpoint_name} | Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    print("🚀 启动 M3-MSE-Controlled 物理空间评估")

    stats_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'data',
            'stats',
            'rbc_field_stats.json'
        )
    )

    if not os.path.exists(stats_file):
        raise FileNotFoundError(
            f"找不到统计量文件: {stats_file}，请先运行 compute_field_stats.py"
        )

    normalizer = FieldWiseNormalizer(
        stats_path=stats_file,
        device=device
    )

    split_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'data',
            'splits',
            'iid_split.json'
        )
    )

    with open(split_file, 'r', encoding='utf-8') as f:
        split_config = json.load(f)

    val_dataset = RBCDataset(
        split_config=split_config["val"],
        normalize=True,
        stats_path=stats_file
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("📦 正在加载 M3-MSE-Controlled 权重...")
    model_m3 = load_model(
        'm3_mse_controlled_best.pth',
        device
    )

    metrics = {
        'M3-MSE-Controlled': {
            'rel': {
                'buoyancy': 0,
                'u_x': 0,
                'u_y': 0,
                'pressure': 0,
                'global': 0
            },
            'mse': {
                'buoyancy': 0,
                'u_x': 0,
                'u_y': 0,
                'pressure': 0,
                'global': 0
            }
        }
    }

    num_batches = len(val_loader)

    print("⏳ 开始 M3-MSE one-step physical-space evaluation...\n")

    with torch.no_grad():
        for batch_x_norm, batch_y_norm in val_loader:
            batch_x_norm = batch_x_norm.to(device)
            batch_y_norm = batch_y_norm.to(device)

            pred_norm = model_m3(batch_x_norm)

            pred_phys = normalizer.denormalize_y(pred_norm)
            target_phys = normalizer.denormalize_y(batch_y_norm)

            assert pred_phys.shape == target_phys.shape == batch_y_norm.shape, (
                "❌ 反归一化后张量维度异常"
            )

            batch_field_rel = compute_field_wise_rel_l2(
                pred_phys,
                target_phys
            )

            batch_total_rel = compute_total_rel_l2(
                pred_phys,
                target_phys
            )

            batch_field_mse = compute_field_wise_mse(
                pred_phys,
                target_phys
            )

            batch_total_mse = compute_total_mse(
                pred_phys,
                target_phys
            )

            for k in ['buoyancy', 'u_x', 'u_y', 'pressure']:
                metrics['M3-MSE-Controlled']['rel'][k] += batch_field_rel[k]
                metrics['M3-MSE-Controlled']['mse'][k] += batch_field_mse[k]

            metrics['M3-MSE-Controlled']['rel']['global'] += batch_total_rel
            metrics['M3-MSE-Controlled']['mse']['global'] += batch_total_mse

    for metric_type in ['rel', 'mse']:
        for k in metrics['M3-MSE-Controlled'][metric_type]:
            metrics['M3-MSE-Controlled'][metric_type][k] /= num_batches

    fields = ['buoyancy', 'u_x', 'u_y', 'pressure']

    print("=" * 75)
    print(" 📊 Quantitative Evaluation: M3-MSE-Controlled Physical Space")
    print("=" * 75)
    print(f"{'Field':<15} | {'Metric':<12} | {'M3-MSE-Controlled':<22}")
    print("-" * 75)

    for field in fields:
        m3_rel = metrics['M3-MSE-Controlled']['rel'][field] * 100
        m3_mse = metrics['M3-MSE-Controlled']['mse'][field]

        print(
            f"{field:<15} | {'Rel-L2 (%)':<12} | "
            f"{m3_rel:>18.2f} %"
        )

        print(
            f"{'':<15} | {'MSE':<12} | "
            f"{m3_mse:>20.6f}"
        )

        print("-" * 75)

    m3_rel_g = metrics['M3-MSE-Controlled']['rel']['global'] * 100
    m3_mse_g = metrics['M3-MSE-Controlled']['mse']['global']

    print(
        f"{'Total(Global)':<15} | {'Rel-L2 (%)':<12} | "
        f"{m3_rel_g:>18.2f} %"
    )

    print(
        f"{'':<15} | {'MSE':<12} | "
        f"{m3_mse_g:>20.6f}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()