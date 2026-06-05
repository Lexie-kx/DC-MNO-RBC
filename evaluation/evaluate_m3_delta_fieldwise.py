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


def init_metric_dict():
    return {
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    print("🚀 启动 M3-Delta 对比评估：M0 vs M3-RelL2 vs M3-Delta-RelL2")

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

    # physical-space dataset：用于 M0
    val_dataset_plain = RBCDataset(
        split_config=split_config["val"],
        normalize=False
    )

    val_loader_plain = DataLoader(
        val_dataset_plain,
        batch_size=batch_size,
        shuffle=False
    )

    # normalized dataset：用于 M3 和 M3-Delta
    val_dataset_norm = RBCDataset(
        split_config=split_config["val"],
        normalize=True,
        stats_path=stats_file
    )

    val_loader_norm = DataLoader(
        val_dataset_norm,
        batch_size=batch_size,
        shuffle=False
    )

    print("📦 正在加载 M3 和 M3-Delta 权重...")
    model_m3 = load_model(
        'm3_rel_l2_controlled_best.pth',
        device
    )

    model_delta = load_model(
        'm3_delta_rel_l2_controlled_best.pth',
        device
    )

    metrics = {
        'M0': init_metric_dict(),
        'M3-REL': init_metric_dict(),
        'M3-DELTA': init_metric_dict()
    }

    num_batches = len(val_loader_plain)

    print("\n⏳ 开始验证集 physical-space 对比评估...\n")

    with torch.no_grad():
        for (batch_x_plain, batch_y_plain), (batch_x_norm, batch_y_norm) in zip(
            val_loader_plain,
            val_loader_norm
        ):
            batch_x_plain = batch_x_plain.to(device)
            batch_y_plain = batch_y_plain.to(device)

            batch_x_norm = batch_x_norm.to(device)
            batch_y_norm = batch_y_norm.to(device)

            # 检查 plain target 和 normalized target 反归一化后是否一致
            target_norm_phys = normalizer.denormalize_y(batch_y_norm)

            if not torch.allclose(batch_y_plain, target_norm_phys, atol=1e-5, rtol=1e-4):
                raise RuntimeError("❌ plain loader 和 normalized loader 的 target 不一致")

            # M0 persistence: physical space
            pred_m0_phys = batch_x_plain[:, -4:, :, :]

            # M3 state prediction:
            # pred_y_norm = model_m3(x_norm)
            pred_m3_norm = model_m3(batch_x_norm)
            pred_m3_phys = normalizer.denormalize_y(pred_m3_norm)

            # M3-Delta prediction:
            # pred_delta_norm = model_delta(x_norm)
            # pred_y_norm = x_last_norm + pred_delta_norm
            x_last_norm = batch_x_norm[:, -4:, :, :]
            pred_delta_norm = model_delta(batch_x_norm)
            pred_delta_y_norm = x_last_norm + pred_delta_norm
            pred_delta_phys = normalizer.denormalize_y(pred_delta_y_norm)

            preds = {
                'M0': pred_m0_phys,
                'M3-REL': pred_m3_phys,
                'M3-DELTA': pred_delta_phys
            }

            for model_name, pred_phys in preds.items():
                batch_field_rel = compute_field_wise_rel_l2(
                    pred_phys,
                    batch_y_plain
                )

                batch_total_rel = compute_total_rel_l2(
                    pred_phys,
                    batch_y_plain
                )

                batch_field_mse = compute_field_wise_mse(
                    pred_phys,
                    batch_y_plain
                )

                batch_total_mse = compute_total_mse(
                    pred_phys,
                    batch_y_plain
                )

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

    print("=" * 125)
    print(" 📊 M3-Delta Controlled Quantitative Evaluation: M0 vs M3-RelL2 vs M3-Delta-RelL2")
    print("=" * 125)
    print(
        f"{'Field':<14} | {'Metric':<12} | "
        f"{'M0 Persistence':<18} | {'M3-RelL2 Controlled':<22} | {'M3-Delta Controlled':<24}"
    )
    print("-" * 125)

    for field in fields:
        m0_r = metrics['M0']['rel'][field] * 100
        m3_r = metrics['M3-REL']['rel'][field] * 100
        delta_r = metrics['M3-DELTA']['rel'][field] * 100

        print(
            f"{field:<14} | {'Rel-L2 (%)':<12} | "
            f"{m0_r:>16.2f} % | {m3_r:>20.2f} % | {delta_r:>22.2f} %"
        )

        m0_m = metrics['M0']['mse'][field]
        m3_m = metrics['M3-REL']['mse'][field]
        delta_m = metrics['M3-DELTA']['mse'][field]

        print(
            f"{'':<14} | {'MSE':<12} | "
            f"{m0_m:>18.6f} | {m3_m:>22.6f} | {delta_m:>24.6f}"
        )
        print("-" * 125)

    m0_r_g = metrics['M0']['rel']['global'] * 100
    m3_r_g = metrics['M3-REL']['rel']['global'] * 100
    delta_r_g = metrics['M3-DELTA']['rel']['global'] * 100

    m0_m_g = metrics['M0']['mse']['global']
    m3_m_g = metrics['M3-REL']['mse']['global']
    delta_m_g = metrics['M3-DELTA']['mse']['global']

    print(
        f"{'Total(Global)':<14} | {'Rel-L2 (%)':<12} | "
        f"{m0_r_g:>16.2f} % | {m3_r_g:>20.2f} % | {delta_r_g:>22.2f} %"
    )

    print(
        f"{'':<14} | {'MSE':<12} | "
        f"{m0_m_g:>18.6f} | {m3_m_g:>22.6f} | {delta_m_g:>24.6f}"
    )

    print("=" * 125)


if __name__ == "__main__":
    main()