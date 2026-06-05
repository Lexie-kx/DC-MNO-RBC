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

    print("🚀 启动 M1 Controlled 三方横评：M0 vs M1-MSE vs M1-RelL2")

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

    print("📦 正在加载 M1 controlled 权重...")
    model_mse = load_model('m1_mse_controlled_best.pth', device)
    model_rel = load_model('m1_rel_l2_controlled_best.pth', device)

    metrics = {
        'M0': {
            'rel': {'buoyancy': 0, 'u_x': 0, 'u_y': 0, 'pressure': 0, 'global': 0},
            'mse': {'buoyancy': 0, 'u_x': 0, 'u_y': 0, 'pressure': 0, 'global': 0}
        },
        'MSE': {
            'rel': {'buoyancy': 0, 'u_x': 0, 'u_y': 0, 'pressure': 0, 'global': 0},
            'mse': {'buoyancy': 0, 'u_x': 0, 'u_y': 0, 'pressure': 0, 'global': 0}
        },
        'REL': {
            'rel': {'buoyancy': 0, 'u_x': 0, 'u_y': 0, 'pressure': 0, 'global': 0},
            'mse': {'buoyancy': 0, 'u_x': 0, 'u_y': 0, 'pressure': 0, 'global': 0}
        }
    }

    num_batches = len(val_loader)

    print("⏳ 开始验证集评估...\n")

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred_m0 = batch_x[:, -4:, :, :]
            pred_mse = model_mse(batch_x)
            pred_rel = model_rel(batch_x)

            preds = {
                'M0': pred_m0,
                'MSE': pred_mse,
                'REL': pred_rel
            }

            for model_name, pred in preds.items():
                batch_field_rel = compute_field_wise_rel_l2(pred, batch_y)
                batch_total_rel = compute_total_rel_l2(pred, batch_y)

                batch_field_mse = compute_field_wise_mse(pred, batch_y)
                batch_total_mse = compute_total_mse(pred, batch_y)

                for k in ['buoyancy', 'u_x', 'u_y', 'pressure']:
                    metrics[model_name]['rel'][k] += batch_field_rel[k]
                    metrics[model_name]['mse'][k] += batch_field_mse[k]

                metrics[model_name]['rel']['global'] += batch_total_rel
                metrics[model_name]['mse']['global'] += batch_total_mse

    for model_name in metrics:
        for metric_type in ['rel', 'mse']:
            for k in metrics[model_name][metric_type]:
                metrics[model_name][metric_type][k] /= num_batches

    fields = ['buoyancy', 'u_x', 'u_y', 'pressure']

    print("=" * 100)
    print(" 📊 M1 Controlled Quantitative Evaluation: M0 vs M1-MSE vs M1-RelL2")
    print("=" * 100)
    print(
        f"{'Field':<14} | {'Metric':<12} | "
        f"{'M0 Persistence':<18} | {'M1-MSE Controlled':<20} | {'M1-RelL2 Controlled':<22}"
    )
    print("-" * 100)

    for field in fields:
        m0_r = metrics['M0']['rel'][field] * 100
        mse_r = metrics['MSE']['rel'][field] * 100
        rel_r = metrics['REL']['rel'][field] * 100

        print(
            f"{field:<14} | {'Rel-L2 (%)':<12} | "
            f"{m0_r:>16.2f} % | {mse_r:>18.2f} % | {rel_r:>20.2f} %"
        )

        m0_m = metrics['M0']['mse'][field]
        mse_m = metrics['MSE']['mse'][field]
        rel_m = metrics['REL']['mse'][field]

        print(
            f"{'':<14} | {'MSE':<12} | "
            f"{m0_m:>18.6f} | {mse_m:>20.6f} | {rel_m:>22.6f}"
        )
        print("-" * 100)

    m0_r_g = metrics['M0']['rel']['global'] * 100
    mse_r_g = metrics['MSE']['rel']['global'] * 100
    rel_r_g = metrics['REL']['rel']['global'] * 100

    m0_m_g = metrics['M0']['mse']['global']
    mse_m_g = metrics['MSE']['mse']['global']
    rel_m_g = metrics['REL']['mse']['global']

    print(
        f"{'Total(Global)':<14} | {'Rel-L2 (%)':<12} | "
        f"{m0_r_g:>16.2f} % | {mse_r_g:>18.2f} % | {rel_r_g:>20.2f} %"
    )

    print(
        f"{'':<14} | {'MSE':<12} | "
        f"{m0_m_g:>18.6f} | {mse_m_g:>20.6f} | {rel_m_g:>22.6f}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()