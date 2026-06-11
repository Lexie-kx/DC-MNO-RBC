import os
import sys
import json
import math
import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import DATA_PATH, FIELD_ORDER, CONTEXT_LENGTH
from training.normalization import FieldWiseNormalizer
from models.operators.fno2d import PlainFNO2d
from models.operators.fno2d_film import FiLMFNO2d


FIELD_TO_IDX = {
    "buoyancy": 0,
    "u_x": 1,
    "u_y": 2,
    "pressure": 3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize cross-parameter rollout fields for M0 / M3-Delta / M3-Delta-FiLM."
    )

    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--stats", type=str, required=True)

    parser.add_argument(
        "--m3_delta_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--film_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--field",
        type=str,
        default="u_x",
        choices=["buoyancy", "u_x", "u_y", "pressure"],
    )

    parser.add_argument(
        "--time_steps",
        type=str,
        default="1,4,8,16",
        help="Comma-separated rollout horizons to visualize."
    )

    parser.add_argument(
        "--group_index",
        type=int,
        default=0,
        help="Which group item in split['test'] to visualize."
    )

    parser.add_argument(
        "--traj_index_in_group",
        type=int,
        default=0,
        help="Which trajectory index inside selected split group list."
    )

    parser.add_argument(
        "--t0",
        type=int,
        default=0,
        help="Rollout start time."
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Name used in output filename, e.g. unseen_pr or unseen_ra."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/figures/rollout_fields",
    )

    return parser.parse_args()


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


def parse_ra_pr_from_group(group_name):
    import re

    ra_match = re.search(r"[Rr]a_?([0-9.eE+-]+)", group_name)
    pr_match = re.search(r"[Pp]r_?([0-9.eE+-]+)", group_name)

    if ra_match is None or pr_match is None:
        raise ValueError(f"Cannot parse Ra/Pr from group name: {group_name}")

    ra = float(ra_match.group(1))
    pr = float(pr_match.group(1))

    return ra, pr


def make_param_tensor(group_name, device):
    ra, pr = parse_ra_pr_from_group(group_name)

    param = torch.tensor(
        [[math.log10(float(ra)), math.log10(float(pr))]],
        dtype=torch.float32,
        device=device,
    )

    return param


def load_model_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def read_rollout_sample(split_path, group_index, traj_index_in_group, t0, max_horizon, device):
    with open(split_path, "r", encoding="utf-8") as f:
        split_config = json.load(f)

    test_config = split_config["test"]

    if group_index >= len(test_config):
        raise IndexError(
            f"group_index={group_index} out of range; test has {len(test_config)} groups."
        )

    item = test_config[group_index]
    group_name = item["group"]
    traj_list = item["trajectories"]

    if traj_index_in_group >= len(traj_list):
        raise IndexError(
            f"traj_index_in_group={traj_index_in_group} out of range; "
            f"group has {len(traj_list)} trajectories."
        )

    traj_idx = traj_list[traj_index_in_group]

    print(f"📦 Reading HDF5 sample:")
    print(f"   group = {group_name}")
    print(f"   trajectory = {traj_idx}")
    print(f"   t0 = {t0}")

    with h5py.File(DATA_PATH, "r") as f:
        group = f[group_name]

        # 每个 field: [num_traj, T, H, W]
        field_arrays = []

        needed_steps = t0 + CONTEXT_LENGTH + max_horizon

        for field in FIELD_ORDER:
            arr = group[field][traj_idx, :needed_steps, :, :]
            field_arrays.append(arr)

        # [4, T, H, W] -> [T, 4, H, W]
        traj_data = np.stack(field_arrays, axis=0).transpose(1, 0, 2, 3)

    if needed_steps > traj_data.shape[0]:
        raise ValueError(
            f"Not enough time steps. Need {needed_steps}, got {traj_data.shape[0]}"
        )

    history = traj_data[t0: t0 + CONTEXT_LENGTH]
    future = traj_data[
        t0 + CONTEXT_LENGTH:
        t0 + CONTEXT_LENGTH + max_horizon
    ]

    # history: [4, 4, H, W] -> x0: [1, 16, H, W]
    _, _, H, W = history.shape
    x0_phys = history.reshape(CONTEXT_LENGTH * 4, H, W)
    x0_phys = torch.tensor(x0_phys, dtype=torch.float32).unsqueeze(0).to(device)

    # future: [max_horizon, 4, H, W]
    future_phys = torch.tensor(future, dtype=torch.float32)

    param = make_param_tensor(group_name, device=device)

    return x0_phys, future_phys, param, group_name, traj_idx


@torch.no_grad()
def rollout_m0(x0_phys, max_horizon):
    current = x0_phys[:, -4:, :, :].detach().cpu()
    preds = []

    for _ in range(max_horizon):
        preds.append(current.clone())

    return preds


@torch.no_grad()
def rollout_m3_delta(model, x0_phys, normalizer, max_horizon):
    preds = []

    current_x_norm = normalizer.normalize_x(x0_phys)

    for _ in range(max_horizon):
        x_last_norm = current_x_norm[:, -4:, :, :]

        pred_delta_norm = model(current_x_norm)
        pred_y_norm = x_last_norm + pred_delta_norm

        pred_y_phys = normalizer.denormalize_y(pred_y_norm)

        preds.append(pred_y_phys.detach().cpu())

        current_x_norm = torch.cat(
            [
                current_x_norm[:, 4:, :, :],
                pred_y_norm,
            ],
            dim=1,
        )

    return preds


@torch.no_grad()
def rollout_film(model, x0_phys, param, normalizer, max_horizon):
    preds = []

    current_x_norm = normalizer.normalize_x(x0_phys)

    for _ in range(max_horizon):
        x_last_norm = current_x_norm[:, -4:, :, :]

        pred_delta_norm = model(current_x_norm, param)
        pred_y_norm = x_last_norm + pred_delta_norm

        pred_y_phys = normalizer.denormalize_y(pred_y_norm)

        preds.append(pred_y_phys.detach().cpu())

        current_x_norm = torch.cat(
            [
                current_x_norm[:, 4:, :, :],
                pred_y_norm,
            ],
            dim=1,
        )

    return preds


def plot_rollout_field(
    gt_seq,
    pred_dict,
    field,
    time_steps,
    output_path,
    title,
):
    field_idx = FIELD_TO_IDX[field]

    model_names = list(pred_dict.keys())

    # 行：
    # GT
    # M0
    # M3-Delta
    # M3-Delta-FiLM
    # Err(M0)
    # Err(M3-Delta)
    # Err(M3-Delta-FiLM)
    row_labels = ["GT"] + model_names + [f"Err({name})" for name in model_names]

    num_rows = len(row_labels)
    num_cols = len(time_steps)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(3.2 * num_cols, 2.4 * num_rows),
        squeeze=False,
    )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)

    # 统一物理场色标：用 GT + predictions 的 2%~98% 分位，防止极端值毁图
    state_values = []

    for t in time_steps:
        idx = t - 1
        gt_img = gt_seq[idx][field_idx].cpu().numpy()
        state_values.append(gt_img.flatten())

        for name in model_names:
            pred_img = pred_dict[name][idx][0, field_idx].cpu().numpy()
            state_values.append(pred_img.flatten())

    state_values = np.concatenate(state_values)
    vmin = np.percentile(state_values, 2)
    vmax = np.percentile(state_values, 98)

    # 统一 error 色标，便于不同模型横向比较
    err_values = []

    for t in time_steps:
        idx = t - 1
        gt_img = gt_seq[idx][field_idx].cpu().numpy()

        for name in model_names:
            pred_img = pred_dict[name][idx][0, field_idx].cpu().numpy()
            err_img = np.abs(gt_img - pred_img)
            err_values.append(err_img.flatten())

    err_values = np.concatenate(err_values)
    err_vmax = np.percentile(err_values, 95)

    if err_vmax <= 1e-12:
        err_vmax = 1.0

    for col_idx, t in enumerate(time_steps):
        idx = t - 1

        gt_img = gt_seq[idx][field_idx].cpu().numpy()

        # GT row
        ax = axes[0, col_idx]
        im_state = ax.imshow(
            gt_img,
            cmap="RdBu_r",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_title(f"t + {t}", fontsize=12, fontweight="bold")

        # prediction rows
        for m_idx, model_name in enumerate(model_names):
            row_idx = 1 + m_idx
            ax = axes[row_idx, col_idx]

            pred_img = pred_dict[model_name][idx][0, field_idx].cpu().numpy()

            ax.imshow(
                pred_img,
                cmap="RdBu_r",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )

        # error rows
        for m_idx, model_name in enumerate(model_names):
            row_idx = 1 + len(model_names) + m_idx
            ax = axes[row_idx, col_idx]

            pred_img = pred_dict[model_name][idx][0, field_idx].cpu().numpy()
            err_img = np.abs(gt_img - pred_img)

            ax.imshow(
                err_img,
                cmap="inferno",
                origin="lower",
                vmin=0,
                vmax=err_vmax,
                aspect="equal",
            )

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0:
                ax.set_ylabel(
                    row_labels[row_idx],
                    rotation=0,
                    fontsize=10,
                    fontweight="bold",
                    labelpad=45,
                    va="center",
                )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, left=0.26, hspace=0.12, wspace=0.08)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved field rollout visualization: {output_path}")


def main():
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    split_path = resolve_path(project_root, args.split)
    stats_path = resolve_path(project_root, args.stats)
    m3_delta_ckpt = resolve_path(project_root, args.m3_delta_checkpoint)
    film_ckpt = resolve_path(project_root, args.film_checkpoint)
    output_dir = resolve_path(project_root, args.output_dir)

    time_steps = [int(x) for x in args.time_steps.split(",")]
    max_horizon = max(time_steps)

    if args.tag is None:
        tag = os.path.splitext(os.path.basename(split_path))[0].replace("_split", "")
    else:
        tag = args.tag

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 [Visualize Cross-Parameter Rollout Fields]")
    print(f"📌 Device: {device}")
    print(f"📌 Split: {split_path}")
    print(f"📌 Stats: {stats_path}")
    print(f"📌 M3-Delta checkpoint: {m3_delta_ckpt}")
    print(f"📌 FiLM checkpoint: {film_ckpt}")
    print(f"📌 Field: {args.field}")
    print(f"📌 Time steps: {time_steps}")
    print(f"📌 Output dir: {output_dir}")

    normalizer = FieldWiseNormalizer(stats_path=stats_path, device=device)

    m3_delta = PlainFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
    ).to(device)

    m3_delta = load_model_state(m3_delta, m3_delta_ckpt, device)

    film = FiLMFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        film_hidden_dim=64,
    ).to(device)

    film = load_model_state(film, film_ckpt, device)

    x0_phys, gt_seq, param, group_name, traj_idx = read_rollout_sample(
        split_path=split_path,
        group_index=args.group_index,
        traj_index_in_group=args.traj_index_in_group,
        t0=args.t0,
        max_horizon=max_horizon,
        device=device,
    )

    print("⚙️ Running rollouts...")

    seq_m0 = rollout_m0(
        x0_phys=x0_phys,
        max_horizon=max_horizon,
    )

    seq_m3_delta = rollout_m3_delta(
        model=m3_delta,
        x0_phys=x0_phys,
        normalizer=normalizer,
        max_horizon=max_horizon,
    )

    seq_film = rollout_film(
        model=film,
        x0_phys=x0_phys,
        param=param,
        normalizer=normalizer,
        max_horizon=max_horizon,
    )

    pred_dict = {
        
        "M3-Delta": seq_m3_delta,
        "M3-Delta-FiLM": seq_film,
    }

    output_path = os.path.join(
        output_dir,
        f"rollout_fields_{tag}_{args.field}_g{args.group_index}_traj{traj_idx}_t{args.t0}.png"
    )

    title = (
        f"Cross-parameter rollout field visualization: {tag}, {args.field}\n"
        f"group={group_name}, traj={traj_idx}, t0={args.t0}"
    )

    plot_rollout_field(
        gt_seq=gt_seq,
        pred_dict=pred_dict,
        field=args.field,
        time_steps=time_steps,
        output_path=output_path,
        title=title,
    )

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
