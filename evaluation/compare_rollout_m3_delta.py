import os
import sys
import json
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.operators.fno2d import PlainFNO2d
from training.normalization import FieldWiseNormalizer
from constants import DATA_PATH, FIELD_ORDER


class AutoregressiveRolloutM3State:
    """M3 state prediction：模型直接预测下一帧 normalized state"""

    def __init__(self, model_path, device, normalizer):
        self.device = device
        self.normalizer = normalizer

        self.model = PlainFNO2d(
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到 checkpoint: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"✅ 成功加载 M3-State 权重: {os.path.basename(model_path)}")

    @torch.no_grad()
    def rollout(self, initial_x_phys, steps=8):
        predictions_phys = []
        current_x_phys = initial_x_phys.clone().to(self.device)

        for _ in range(steps):
            current_x_norm = self.normalizer.normalize_x(current_x_phys)

            pred_y_norm = self.model(current_x_norm)
            pred_y_phys = self.normalizer.denormalize_y(pred_y_norm)

            predictions_phys.append(pred_y_phys.cpu())

            retained_history_phys = current_x_phys[:, 4:, :, :]
            current_x_phys = torch.cat(
                [retained_history_phys, pred_y_phys],
                dim=1
            )

        return predictions_phys


class AutoregressiveRolloutM3Delta:
    """M3 delta prediction：模型预测 delta，rollout 时加回当前最后一帧"""

    def __init__(self, model_path, device, normalizer):
        self.device = device
        self.normalizer = normalizer

        self.model = PlainFNO2d(
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到 checkpoint: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"✅ 成功加载 M3-Delta 权重: {os.path.basename(model_path)}")

    @torch.no_grad()
    def rollout(self, initial_x_phys, steps=8):
        predictions_phys = []
        current_x_phys = initial_x_phys.clone().to(self.device)

        for _ in range(steps):
            # 当前 4 帧历史：physical -> normalized
            current_x_norm = self.normalizer.normalize_x(current_x_phys)

            # 预测 normalized delta
            pred_delta_norm = self.model(current_x_norm)

            # 当前最后一帧 normalized state
            x_last_norm = current_x_norm[:, -4:, :, :]

            # delta 还原成下一帧 normalized state
            pred_y_norm = x_last_norm + pred_delta_norm

            # normalized state -> physical state
            pred_y_phys = self.normalizer.denormalize_y(pred_y_norm)

            predictions_phys.append(pred_y_phys.cpu())

            # 用预测结果进入下一步自回归
            retained_history_phys = current_x_phys[:, 4:, :, :]
            current_x_phys = torch.cat(
                [retained_history_phys, pred_y_phys],
                dim=1
            )

        return predictions_phys


def plot_publication_diagnostic(
    gt_seq,
    pred_seqs_dict,
    field_idx,
    field_name,
    time_steps=(1, 4, 8),
    save_dir=None,
):
    if save_dir is None:
        save_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "outputs",
                "figures",
                "rollout_m3_delta"
            )
        )

    os.makedirs(save_dir, exist_ok=True)

    model_names = list(pred_seqs_dict.keys())
    num_models = len(model_names)
    total_rows = 1 + num_models * 2

    row_labels = ["GT"] + model_names + [f"Err({name})" for name in model_names]

    fig, axes = plt.subplots(
        nrows=total_rows,
        ncols=len(time_steps),
        figsize=(5.2 * len(time_steps), 2.8 * total_rows),
        squeeze=False
    )

    fig.suptitle(
        f"M3 State vs Delta Autoregressive Diagnosis: {field_name}",
        fontsize=20,
        y=0.98,
        fontweight="bold"
    )

    # 物理场统一色标：基于 GT 全时序
    all_gt = np.concatenate([
        gt_seq[i][0, field_idx].cpu().numpy().flatten()
        for i in range(len(gt_seq))
    ])

    vmin, vmax = all_gt.min(), all_gt.max()

    # 每个模型单独 error 色标：显示各自误差结构
    err_vmax_by_model = {}

    for model_name, seq in pred_seqs_dict.items():
        model_errors = []

        for i in range(len(gt_seq)):
            gt_img = gt_seq[i][0, field_idx].cpu().numpy()
            pred_img = seq[i][0, field_idx].cpu().numpy()
            err_img = np.abs(gt_img - pred_img)
            model_errors.extend(err_img.flatten())

        if len(model_errors) > 0:
            err_vmax = np.percentile(model_errors, 95)
            if err_vmax <= 1e-12:
                err_vmax = 1.0
        else:
            err_vmax = 1.0

        err_vmax_by_model[model_name] = err_vmax

    img_shape_y = gt_seq[0][0, field_idx].shape[0]

    for col_idx, t in enumerate(time_steps):
        t_idx = t - 1
        gt_img = gt_seq[t_idx][0, field_idx].cpu().numpy()

        # Row 0: GT
        ax = axes[0, col_idx]
        ax.imshow(
            gt_img,
            cmap="RdBu_r",
            origin="lower",
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(f"Horizon t + {t}", fontsize=14, fontweight="bold")

        # Prediction rows
        for m_idx, model_name in enumerate(model_names):
            row_idx = 1 + m_idx
            ax = axes[row_idx, col_idx]

            pred_img = pred_seqs_dict[model_name][t_idx][0, field_idx].cpu().numpy()

            ax.imshow(
                pred_img,
                cmap="RdBu_r",
                origin="lower",
                vmin=vmin,
                vmax=vmax
            )

        # Error rows
        for m_idx, model_name in enumerate(model_names):
            row_idx = 1 + num_models + m_idx
            ax = axes[row_idx, col_idx]

            pred_img = pred_seqs_dict[model_name][t_idx][0, field_idx].cpu().numpy()
            err_img = np.abs(gt_img - pred_img)

            ax.imshow(
                err_img,
                cmap="inferno",
                origin="lower",
                vmin=0,
                vmax=err_vmax_by_model[model_name]
            )

    for row_idx in range(total_rows):
        for col_idx in range(len(time_steps)):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 1 or row_idx == 1 + num_models:
                ax.axhline(
                    y=img_shape_y + 2,
                    color="black",
                    linewidth=2.5,
                    clip_on=False
                )

    for row_idx, label in enumerate(row_labels):
        y_pos = 1.0 - (row_idx + 0.5) / total_rows

        fig.text(
            0.045,
            y_pos,
            label,
            fontsize=13,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center"
        )

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.94,
        left=0.22,
        hspace=0.05,
        wspace=0.05
    )

    save_path = os.path.join(
        save_dir,
        f"m3_state_vs_delta_rollout_{field_name}.png"
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📉 【诊断完毕】{field_name} rollout 图已存至: {save_path}")
    plt.close()


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    STEPS = 8

    print("🚀 启动 M3-RelL2 vs M3-Delta-RelL2 自回归 rollout 评估...")

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    ckpt_dir = os.path.join(root_dir, "checkpoints", "controlled")
    save_dir = os.path.join(root_dir, "outputs", "figures", "rollout_m3_delta")
    os.makedirs(save_dir, exist_ok=True)

    required_ckpts = [
        "m3_rel_l2_controlled_best.pth",
        "m3_delta_rel_l2_controlled_best.pth",
    ]

    for name in required_ckpts:
        path = os.path.join(ckpt_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ 缺少 checkpoint: {path}")

    split_file = os.path.join(root_dir, "data", "splits", "iid_split.json")
    with open(split_file, "r", encoding="utf-8") as f:
        split_config = json.load(f)

    val_first_group = split_config["val"][0]
    group_name = val_first_group["group"]
    traj_idx = val_first_group["trajectories"][0]

    print(f"📦 直接从 HDF5 读取 Group={group_name}, Traj={traj_idx}")

    with h5py.File(DATA_PATH, "r") as f:
        group = f[group_name]

        traj_data = []
        for field in FIELD_ORDER:
            traj_data.append(group[field][traj_idx, :STEPS + 4, :, :])

        traj_data = np.stack(traj_data, axis=0)
        traj_data = traj_data.transpose(1, 0, 2, 3)

        x0_np = traj_data[0:4, :, :, :].reshape(
            -1,
            traj_data.shape[2],
            traj_data.shape[3]
        )

        initial_x = torch.tensor(
            x0_np,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        gt_sequence = []
        for step in range(STEPS):
            y_np = traj_data[4 + step, :, :, :]
            gt_sequence.append(
                torch.tensor(y_np, dtype=torch.float32).unsqueeze(0)
            )

    print("✅ 物理数据切片提取完成")

    stats_file = os.path.join(root_dir, "data", "stats", "rbc_field_stats.json")
    normalizer = FieldWiseNormalizer(stats_path=stats_file, device=DEVICE)

    print("\n⚙️ 正在执行 M3-RelL2-Controlled state rollout...")
    m3_state = AutoregressiveRolloutM3State(
        os.path.join(ckpt_dir, "m3_rel_l2_controlled_best.pth"),
        DEVICE,
        normalizer
    )
    seq_m3_state = m3_state.rollout(initial_x, steps=STEPS)

    print("\n⚙️ 正在执行 M3-Delta-RelL2-Controlled delta rollout...")
    m3_delta = AutoregressiveRolloutM3Delta(
        os.path.join(ckpt_dir, "m3_delta_rel_l2_controlled_best.pth"),
        DEVICE,
        normalizer
    )
    seq_m3_delta = m3_delta.rollout(initial_x, steps=STEPS)

    pred_seqs_dict = {
        "M3-State-RelL2": seq_m3_state,
        "M3-Delta-RelL2": seq_m3_delta,
    }

    fields = [
        (0, "buoyancy"),
        (1, "u_x"),
        (2, "u_y"),
        (3, "pressure"),
    ]

    print("\n🎨 正在绘制 M3 State vs Delta rollout 诊断图...")
    for idx, name in fields:
        plot_publication_diagnostic(
            gt_seq=gt_sequence,
            pred_seqs_dict=pred_seqs_dict,
            field_idx=idx,
            field_name=name,
            time_steps=(1, 4, 8),
            save_dir=save_dir
        )

    print(f"\n✅ M3 State vs Delta rollout 图已全部保存到: {save_dir}")


if __name__ == "__main__":
    main()