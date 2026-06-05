import os
import sys
import json
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.rbc_dataset import RBCDataset
from models.operators.fno2d import PlainFNO2d
from training.metrics import (
    compute_field_wise_rel_l2,
    compute_total_rel_l2,
    compute_field_wise_mse,
    compute_total_mse
)


class ParamNormChannelDataset(Dataset):
    """
    用于 M2-RelL2-ParamNorm-Controlled 评估。

    RBCDataset 原始返回:
        x:     [16, H, W]
        y:     [4, H, W]
        param: [2] = [log10(Ra), log10(Pr)]

    这里把 param 归一化后拼接到输入通道:

        Ra = 1e6, 1e7, 1e8
        log10(Ra) = 6, 7, 8
        ra_norm = (log10(Ra) - 7.0) / 1.0
        对应 -1, 0, 1

        Pr = 0.5, 1, 2
        log10(Pr) = -0.30103, 0, 0.30103
        pr_norm = log10(Pr) / 0.30103
        对应 -1, 0, 1

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

        # param[0] = log10(Ra)
        # param[1] = log10(Pr)
        param_norm = param.clone()
        param_norm[0] = (param_norm[0] - 7.0) / 1.0
        param_norm[1] = param_norm[1] / 0.30103

        _, H, W = x.shape

        # [2] -> [2, H, W]
        param_map = param_norm[:, None, None].expand(2, H, W)

        # [16, H, W] + [2, H, W] -> [18, H, W]
        x = torch.cat([x, param_map], dim=0)

        return x, y


def load_m2_paramnorm_model(checkpoint_name, device):
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
            checkpoint_name
        )
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ 加载 M2-ParamNorm 权重: {checkpoint_name} | Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model


def load_m1_rel_model(device):
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
            'm1_rel_l2_controlled_best.pth'
        )
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 M1 权重文件: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ 加载 M1-RelL2 权重 | Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    print("🚀 启动 M2-ParamNorm Controlled 评估：M0 vs M1-RelL2 vs M2-ParamNorm-RelL2")

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
    # 1. 普通 16 通道数据：用于 M0 和 M1
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
    # 2. 带归一化参数通道的 18 通道数据：用于 M2-ParamNorm
    # =========================
    val_base_dataset_param = RBCDataset(
        split_config=split_config["val"],
        normalize=False,
        return_params=True
    )

    val_dataset_param = ParamNormChannelDataset(val_base_dataset_param)

    val_loader_param = DataLoader(
        val_dataset_param,
        batch_size=batch_size,
        shuffle=False
    )

    # 检查一个 batch
    sample_x, sample_y = next(iter(val_loader_param))
    print(f"✅ M2-ParamNorm 输入检查: X shape = {sample_x.shape}，应为 [B, 18, 256, 64]")
    print(f"✅ M2-ParamNorm 目标检查: Y shape = {sample_y.shape}，应为 [B, 4, 256, 64]")
    print(f"👉 参数通道示例: {sample_x[0, -2:, 0, 0].tolist()}，应大致在 [-1, 0, 1]")

    print("\n📦 正在加载 controlled 权重...")
    model_m1_rel = load_m1_rel_model(device)
    model_m2_paramnorm = load_m2_paramnorm_model(
        'm2_rel_l2_paramnorm_controlled_best.pth',
        device
    )

    metrics = {
        'M0': {
            'rel': {'buoyancy': 0, 'u_x': 0, 'u_y': 0, 'pressure': 0, 'global': 0},
            'mse': {'buoyancy': 0, 'u_x': 0, 'u_y': 0, 'pressure': 0, 'global': 0}
        },
        'M1-REL': {
            'rel': {'buoyancy': 0, 'u_x': 0, 'u_y': 0, 'pressure': 0, 'global': 0},
            'mse': {'buoyancy': 0, 'u_x': 0, 'u_y': 0, 'pressure': 0, 'global': 0}
        },
        'M2-PARAMNORM': {
            'rel': {'buoyancy': 0, 'u_x': 0, 'u_y': 0, 'pressure': 0, 'global': 0},
            'mse': {'buoyancy': 0, 'u_x': 0, 'u_y': 0, 'pressure': 0, 'global': 0}
        }
    }

    num_batches = len(val_loader_plain)

    print("\n⏳ 开始验证集评估...\n")

    with torch.no_grad():
        for (batch_x_plain, batch_y_plain), (batch_x_param, batch_y_param) in zip(
            val_loader_plain,
            val_loader_param
        ):
            batch_x_plain = batch_x_plain.to(device)
            batch_y_plain = batch_y_plain.to(device)

            batch_x_param = batch_x_param.to(device)
            batch_y_param = batch_y_param.to(device)

            # 防止两个 dataloader 顺序错位
            if not torch.allclose(batch_y_plain, batch_y_param):
                raise RuntimeError(
                    "❌ plain loader 和 param loader 的 batch_y 不一致，请检查 shuffle 是否为 False。"
                )

            pred_m0 = batch_x_plain[:, -4:, :, :]
            pred_m1_rel = model_m1_rel(batch_x_plain)
            pred_m2_paramnorm = model_m2_paramnorm(batch_x_param)

            preds = {
                'M0': pred_m0,
                'M1-REL': pred_m1_rel,
                'M2-PARAMNORM': pred_m2_paramnorm
            }

            for model_name, pred in preds.items():
                batch_field_rel = compute_field_wise_rel_l2(pred, batch_y_plain)
                batch_total_rel = compute_total_rel_l2(pred, batch_y_plain)

                batch_field_mse = compute_field_wise_mse(pred, batch_y_plain)
                batch_total_mse = compute_total_mse(pred, batch_y_plain)

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

    print("=" * 120)
    print(" 📊 M2-ParamNorm Controlled Quantitative Evaluation: M0 vs M1-RelL2 vs M2-ParamNorm-RelL2")
    print("=" * 120)
    print(
        f"{'Field':<14} | {'Metric':<12} | "
        f"{'M0 Persistence':<18} | {'M1-RelL2 Controlled':<22} | {'M2-ParamNorm Controlled':<26}"
    )
    print("-" * 120)

    for field in fields:
        m0_r = metrics['M0']['rel'][field] * 100
        m1_r = metrics['M1-REL']['rel'][field] * 100
        m2_r = metrics['M2-PARAMNORM']['rel'][field] * 100

        print(
            f"{field:<14} | {'Rel-L2 (%)':<12} | "
            f"{m0_r:>16.2f} % | {m1_r:>20.2f} % | {m2_r:>24.2f} %"
        )

        m0_m = metrics['M0']['mse'][field]
        m1_m = metrics['M1-REL']['mse'][field]
        m2_m = metrics['M2-PARAMNORM']['mse'][field]

        print(
            f"{'':<14} | {'MSE':<12} | "
            f"{m0_m:>18.6f} | {m1_m:>22.6f} | {m2_m:>26.6f}"
        )
        print("-" * 120)

    m0_r_g = metrics['M0']['rel']['global'] * 100
    m1_r_g = metrics['M1-REL']['rel']['global'] * 100
    m2_r_g = metrics['M2-PARAMNORM']['rel']['global'] * 100

    m0_m_g = metrics['M0']['mse']['global']
    m1_m_g = metrics['M1-REL']['mse']['global']
    m2_m_g = metrics['M2-PARAMNORM']['mse']['global']

    print(
        f"{'Total(Global)':<14} | {'Rel-L2 (%)':<12} | "
        f"{m0_r_g:>16.2f} % | {m1_r_g:>20.2f} % | {m2_r_g:>24.2f} %"
    )

    print(
        f"{'':<14} | {'MSE':<12} | "
        f"{m0_m_g:>18.6f} | {m1_m_g:>22.6f} | {m2_m_g:>26.6f}"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()