import os
import sys
import json
import torch
from torch.utils.data import DataLoader, Dataset

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


class ParamNormChannelDataset(Dataset):
    """
    用于 M4-RelL2-ParamNorm-Controlled 评估。

    base_dataset 返回:
        x:     [16, H, W]，已经 field-wise normalize
        y:     [4, H, W]，已经 field-wise normalize
        param: [2] = [log10(Ra), log10(Pr)]

    输出:
        x_with_param: [18, H, W]
        y:            [4, H, W]
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y, param = self.base_dataset[idx]

        param_norm = param.clone()
        param_norm[0] = (param_norm[0] - 7.0) / 1.0
        param_norm[1] = param_norm[1] / 0.30103

        _, H, W = x.shape
        param_map = param_norm[:, None, None].expand(2, H, W)

        x = torch.cat([x, param_map], dim=0)

        return x, y


def load_m3_model(device):
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
            'm3_rel_l2_controlled_best.pth'
        )
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 M3 权重文件: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ 加载 M3-RelL2 权重 | Epoch: {checkpoint.get('epoch', 'N/A')}")
    return model


def load_m4_model(device):
    model = PlainFNO2d(
        in_channels=18,
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
            'm4_rel_l2_paramnorm_controlled_best.pth'
        )
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 M4 权重文件: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ 加载 M4-RelL2-ParamNorm 权重 | Epoch: {checkpoint.get('epoch', 'N/A')}")
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

    print("🚀 启动 M4 对比评估：M0 vs M3-RelL2 vs M4-RelL2-ParamNorm")

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

    # =========================
    # 1. plain physical-space dataset
    # 用于 M0 persistence
    # =========================
    val_dataset_plain = RBCDataset(
        split_config=split_config["val"],
        normalize=False
    )

    val_loader_plain = DataLoader(
        val_dataset_plain,
        batch_size=batch_size,
        shuffle=False
    )

    # =========================
    # 2. normalized 16-channel dataset
    # 用于 M3
    # =========================
    val_dataset_m3 = RBCDataset(
        split_config=split_config["val"],
        normalize=True,
        stats_path=stats_file
    )

    val_loader_m3 = DataLoader(
        val_dataset_m3,
        batch_size=batch_size,
        shuffle=False
    )

    # =========================
    # 3. normalized + param 18-channel dataset
    # 用于 M4
    # =========================
    val_base_dataset_m4 = RBCDataset(
        split_config=split_config["val"],
        normalize=True,
        stats_path=stats_file,
        return_params=True
    )

    val_dataset_m4 = ParamNormChannelDataset(val_base_dataset_m4)

    val_loader_m4 = DataLoader(
        val_dataset_m4,
        batch_size=batch_size,
        shuffle=False
    )

    sample_x, sample_y = next(iter(val_loader_m4))
    print(f"✅ M4 输入检查: X shape = {sample_x.shape}，应为 [B, 18, 256, 64]")
    print(f"✅ M4 目标检查: Y shape = {sample_y.shape}，应为 [B, 4, 256, 64]")
    print(f"👉 参数通道示例: {sample_x[0, -2:, 0, 0].tolist()}，应大致在 [-1, 0, 1]")

    print("\n📦 正在加载 controlled 权重...")
    model_m3 = load_m3_model(device)
    model_m4 = load_m4_model(device)

    metrics = {
        'M0': init_metric_dict(),
        'M3-REL': init_metric_dict(),
        'M4-PARAMNORM': init_metric_dict()
    }

    num_batches = len(val_loader_plain)

    print("\n⏳ 开始验证集 physical-space 对比评估...\n")

    with torch.no_grad():
        for (
            (batch_x_plain, batch_y_plain),
            (batch_x_m3, batch_y_m3),
            (batch_x_m4, batch_y_m4)
        ) in zip(val_loader_plain, val_loader_m3, val_loader_m4):

            batch_x_plain = batch_x_plain.to(device)
            batch_y_plain = batch_y_plain.to(device)

            batch_x_m3 = batch_x_m3.to(device)
            batch_y_m3 = batch_y_m3.to(device)

            batch_x_m4 = batch_x_m4.to(device)
            batch_y_m4 = batch_y_m4.to(device)

            # 检查三个 dataloader 是否对齐
            target_m3_phys = normalizer.denormalize_y(batch_y_m3)
            target_m4_phys = normalizer.denormalize_y(batch_y_m4)

            if not torch.allclose(batch_y_plain, target_m3_phys, atol=1e-5, rtol=1e-4):
                raise RuntimeError("❌ plain loader 和 M3 loader 的 target 不一致")

            if not torch.allclose(batch_y_plain, target_m4_phys, atol=1e-5, rtol=1e-4):
                raise RuntimeError("❌ plain loader 和 M4 loader 的 target 不一致")

            # M0 persistence: physical space
            pred_m0_phys = batch_x_plain[:, -4:, :, :]

            # M3 prediction: normalized -> physical
            pred_m3_norm = model_m3(batch_x_m3)
            pred_m3_phys = normalizer.denormalize_y(pred_m3_norm)

            # M4 prediction: normalized -> physical
            pred_m4_norm = model_m4(batch_x_m4)
            pred_m4_phys = normalizer.denormalize_y(pred_m4_norm)

            preds = {
                'M0': pred_m0_phys,
                'M3-REL': pred_m3_phys,
                'M4-PARAMNORM': pred_m4_phys
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
    print(" 📊 M4 Controlled Quantitative Evaluation: M0 vs M3-RelL2 vs M4-ParamNorm-RelL2")
    print("=" * 125)
    print(
        f"{'Field':<14} | {'Metric':<12} | "
        f"{'M0 Persistence':<18} | {'M3-RelL2 Controlled':<22} | {'M4-ParamNorm Controlled':<26}"
    )
    print("-" * 125)

    for field in fields:
        m0_r = metrics['M0']['rel'][field] * 100
        m3_r = metrics['M3-REL']['rel'][field] * 100
        m4_r = metrics['M4-PARAMNORM']['rel'][field] * 100

        print(
            f"{field:<14} | {'Rel-L2 (%)':<12} | "
            f"{m0_r:>16.2f} % | {m3_r:>20.2f} % | {m4_r:>24.2f} %"
        )

        m0_m = metrics['M0']['mse'][field]
        m3_m = metrics['M3-REL']['mse'][field]
        m4_m = metrics['M4-PARAMNORM']['mse'][field]

        print(
            f"{'':<14} | {'MSE':<12} | "
            f"{m0_m:>18.6f} | {m3_m:>22.6f} | {m4_m:>26.6f}"
        )
        print("-" * 125)

    m0_r_g = metrics['M0']['rel']['global'] * 100
    m3_r_g = metrics['M3-REL']['rel']['global'] * 100
    m4_r_g = metrics['M4-PARAMNORM']['rel']['global'] * 100

    m0_m_g = metrics['M0']['mse']['global']
    m3_m_g = metrics['M3-REL']['mse']['global']
    m4_m_g = metrics['M4-PARAMNORM']['mse']['global']

    print(
        f"{'Total(Global)':<14} | {'Rel-L2 (%)':<12} | "
        f"{m0_r_g:>16.2f} % | {m3_r_g:>20.2f} % | {m4_r_g:>24.2f} %"
    )

    print(
        f"{'':<14} | {'MSE':<12} | "
        f"{m0_m_g:>18.6f} | {m3_m_g:>22.6f} | {m4_m_g:>26.6f}"
    )

    print("=" * 125)


if __name__ == "__main__":
    main()