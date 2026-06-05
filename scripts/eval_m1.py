import os
import sys
import json
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.rbc_dataset import RBCDataset
from models.operators.fno2d import PlainFNO2d
from training.metrics import (
    compute_field_wise_rel_l2,
    compute_total_rel_l2,
    compute_field_wise_mse,
    compute_total_mse
)


def evaluate_model(model, val_loader, checkpoint_name, device):
    checkpoint_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'checkpoints',
            'controlled',
            checkpoint_name
        )
    )

    if not os.path.exists(checkpoint_path):
        print(f"\n❌ 找不到权重文件: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n✅ 成功加载权重: {checkpoint_name} (Epoch {checkpoint.get('epoch', 'N/A')})")

    total_rel_sum = 0.0
    total_mse_sum = 0.0

    field_rel_sums = {
        'buoyancy': 0.0,
        'u_x': 0.0,
        'u_y': 0.0,
        'pressure': 0.0
    }

    field_mse_sums = {
        'buoyancy': 0.0,
        'u_x': 0.0,
        'u_y': 0.0,
        'pressure': 0.0
    }

    num_batches = len(val_loader)

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)

            batch_total_rel = compute_total_rel_l2(pred, batch_y)
            batch_field_rel = compute_field_wise_rel_l2(pred, batch_y)

            batch_total_mse = compute_total_mse(pred, batch_y)
            batch_field_mse = compute_field_wise_mse(pred, batch_y)

            total_rel_sum += batch_total_rel
            total_mse_sum += batch_total_mse

            for k in field_rel_sums.keys():
                field_rel_sums[k] += batch_field_rel[k]
                field_mse_sums[k] += batch_field_mse[k]

    print("=" * 70)
    print(f" 📊 M1 Controlled 评估报告: {checkpoint_name}")
    print("=" * 70)
    print(f"{'Field':<15} | {'Rel L2 (%)':<15} | {'MSE':<15}")
    print("-" * 70)

    for field in field_rel_sums.keys():
        avg_rel = (field_rel_sums[field] / num_batches) * 100
        avg_mse = field_mse_sums[field] / num_batches
        print(f"{field:<15} | {avg_rel:>11.2f} % | {avg_mse:>13.6f}")

    print("-" * 70)
    print(
        f"{'Total (Global)':<15} | "
        f"{(total_rel_sum / num_batches) * 100:>11.2f} % | "
        f"{total_mse_sum / num_batches:>13.6f}"
    )
    print("=" * 70)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    print(f"🚀 启动 M1 Controlled 对比评估 | 设备: {device}")

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

    val_dataset = RBCDataset(split_config=split_config["val"])

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = PlainFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32
    ).to(device)

    models_to_evaluate = [
        'm1_mse_controlled_best.pth',
        'm1_rel_l2_controlled_best.pth'
    ]

    for ckpt in models_to_evaluate:
        evaluate_model(model, val_loader, ckpt, device)


if __name__ == "__main__":
    main()