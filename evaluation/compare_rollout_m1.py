import os
import sys
import json
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.operators.fno2d import PlainFNO2d
from constants import DATA_PATH, FIELD_ORDER


class AutoregressiveRollout:
    def __init__(self, model_path, device):
        self.device = device

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

        print(f"✅ 成功加载权重: {os.path.basename(model_path)}")

    @torch.no_grad()
    def rollout(self, initial_x, steps=8):
        predictions = []

        current_x = initial_x.clone().to(self.device)

        for _ in range(steps):
            pred_y = self.model(current_x)

            predictions.append(pred_y.cpu())

            retained_history = current_x[:, 4:, :, :]
            current_x = torch.cat([retained_history, pred_y], dim=1)

        return predictions


def load_initial_and_gt_from_h5(split_config, steps=8, device="cpu"):
    val_first_group = split_config["val"][0]

    group_name = val_first_group["group"]
    traj_idx = val_first_group["trajectories"][0]

    print(f"📦 从 HDF5 读取 Group={group_name}, Traj={traj_idx}")

    with h5py.File(DATA_PATH, "r") as f:
        group = f[group_name]

        traj_data = []

        for field in FIELD_ORDER:
            traj_data.append(
                group[field][traj_idx, :steps + 4, :, :]
            )

        traj_data = np.stack(traj_data, axis=0)

        # [4,T,H,W] -> [T,4,H,W]
        traj_data = traj_data.transpose(1, 0, 2, 3)

    x0_np = traj_data[0:4, :, :, :].reshape(
        -1,
        traj_data.shape[2],
        traj_data.shape[3]
    )

    initial_x = torch.tensor(
        x0_np,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    gt_sequence = []

    for step in range(steps):
        y_np = traj_data[4 + step, :, :, :]

        gt_sequence.append(
            torch.tensor(y_np, dtype=torch.float32).unsqueeze(0)
        )

    print("✅ initial_x 与 GT sequence 构建完成")

    return initial_x, gt_sequence


def plot_publication_diagnostic(
    gt_seq,
    pred_seqs_dict,
    field_idx,
    field_name,
    time_steps=(1, 4, 8),
    save_dir="outputs/figures/rollout_2x2"
):
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(pred_seqs_dict.keys())

    num_models = len(model_names)

    total_rows = 1 + num_models * 2

    row_labels = (
        ["Ground Truth"]
        + model_names
        + [f"Error: {name}" for name in model_names]
    )

    fig, axes = plt.subplots(
        nrows=total_rows,
        ncols=len(time_steps),
        figsize=(5.2 * len(time_steps), 3.0 * total_rows),
        squeeze=False
    )

    fig.suptitle(
        f"Autoregressive Physical Diagnosis: {field_name}",
        fontsize=20,
        fontweight="bold",
        y=0.98
    )

    all_gt = np.concatenate([
        gt_seq[i][0, field_idx].cpu().numpy().flatten()
        for i in range(len(gt_seq))
    ])

    vmin = all_gt.min()
    vmax = all_gt.max()

    all_errors = []

    for seq in pred_seqs_dict.values():
        for i in range(len(gt_seq)):
            gt_img = gt_seq[i][0, field_idx].cpu().numpy()
            pred_img = seq[i][0, field_idx].cpu().numpy()

            all_errors.extend(
                np.abs(gt_img - pred_img).flatten()
            )

    vmax_err = np.percentile(all_errors, 95) if len(all_errors) > 0 else 1.0

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

        ax.set_title(
            f"Horizon t + {t}",
            fontsize=14,
            fontweight="bold"
        )

        ax.axis("off")

        # Prediction rows
        for m_idx, model_name in enumerate(model_names):
            row_idx = 1 + m_idx

            pred_img = pred_seqs_dict[model_name][t_idx][0, field_idx].cpu().numpy()

            ax = axes[row_idx, col_idx]

            ax.imshow(
                pred_img,
                cmap="RdBu_r",
                origin="lower",
                vmin=vmin,
                vmax=vmax
            )

            ax.axis("off")

        # Error rows
        for m_idx, model_name in enumerate(model_names):
            row_idx = 1 + num_models + m_idx

            pred_img = pred_seqs_dict[model_name][t_idx][0, field_idx].cpu().numpy()

            err_img = np.abs(gt_img - pred_img)

            ax = axes[row_idx, col_idx]

            ax.imshow(
                err_img,
                cmap="inferno",
                origin="lower",
                vmin=0,
                vmax=vmax_err
            )

            ax.axis("off")

    # 行标签
    for row_idx, label in enumerate(row_labels):
        y_pos = 1.0 - (row_idx + 0.5) / total_rows

        fig.text(
            0.045,
            y_pos,
            label,
            fontsize=15,
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
        f"phase1_m1_controlled_rollout_{field_name}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"📉 已保存: {save_path}")


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    STEPS = 8

    print(f"🚀 启动 M1 Controlled Rollout 对比评估 | 设备: {DEVICE}")

    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    ckpt_dir = os.path.join(
        root_dir,
        "checkpoints",
        "controlled"
    )

    save_dir = os.path.join(
        root_dir,
        "outputs",
        "figures",
        "rollout_2x2"
    )

    mse_ckpt = os.path.join(
        ckpt_dir,
        "m1_mse_controlled_best.pth"
    )

    rel_ckpt = os.path.join(
        ckpt_dir,
        "m1_rel_l2_controlled_best.pth"
    )

    for path in [mse_ckpt, rel_ckpt]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ 缺少 checkpoint: {path}")

    split_file = os.path.join(
        root_dir,
        "data",
        "splits",
        "iid_split.json"
    )

    with open(split_file, "r", encoding="utf-8") as f:
        split_config = json.load(f)

    initial_x, gt_sequence = load_initial_and_gt_from_h5(
        split_config=split_config,
        steps=STEPS,
        device=DEVICE
    )

    print("\n⚙️ 正在执行 M1-MSE-Controlled rollout...")
    m1_mse_engine = AutoregressiveRollout(mse_ckpt, DEVICE)

    seq_m1_mse = m1_mse_engine.rollout(
        initial_x,
        steps=STEPS
    )

    print("\n⚙️ 正在执行 M1-RelL2-Controlled rollout...")
    m1_rel_engine = AutoregressiveRollout(rel_ckpt, DEVICE)

    seq_m1_rel = m1_rel_engine.rollout(
        initial_x,
        steps=STEPS
    )

    pred_seqs_dict = {
        "M1-MSE-Controlled": seq_m1_mse,
        "M1-RelL2-Controlled": seq_m1_rel
    }

    fields = [
        (0, "buoyancy"),
        (1, "u_x"),
        (2, "u_y"),
        (3, "pressure")
    ]

    print("\n🎨 正在绘制 M1 Controlled rollout 图...")

    for idx, name in fields:
        plot_publication_diagnostic(
            gt_seq=gt_sequence,
            pred_seqs_dict=pred_seqs_dict,
            field_idx=idx,
            field_name=name,
            time_steps=(1, 4, 8),
            save_dir=save_dir
        )

    print(f"\n✅ M1 Controlled rollout 图已全部保存到: {save_dir}")


if __name__ == "__main__":
    main()