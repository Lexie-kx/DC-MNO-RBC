import os
import sys
import re
import csv
import json
import math
import argparse
from collections import defaultdict

import h5py
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from training.normalization import FieldWiseNormalizer
from models.operators.fno2d import PlainFNO2d
from models.operators.fno2d_film import FiLMFNO2d


DATA_PATH = constants.DATA_PATH
FIELD_ORDER = constants.FIELD_ORDER
CONTEXT_LENGTH = getattr(constants, "CONTEXT_LENGTH", 4)

FIELD_TO_IDX = {
    "buoyancy": FIELD_ORDER.index("buoyancy"),
    "u_x": FIELD_ORDER.index("u_x"),
    "u_y": FIELD_ORDER.index("u_y"),
    "pressure": FIELD_ORDER.index("pressure"),
}

UX_IDX = FIELD_TO_IDX["u_x"]
UY_IDX = FIELD_TO_IDX["u_y"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate rollout physics diagnostics: divergence and vorticity."
    )

    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--stats", type=str, required=True)

    parser.add_argument("--m3_delta_checkpoint", type=str, required=True)
    parser.add_argument("--film_checkpoint", type=str, required=True)

    parser.add_argument("--output", type=str, required=True)

    parser.add_argument(
        "--horizons",
        type=str,
        default="1,4,8,16",
        help="Comma-separated rollout horizons, e.g. 1,4,8,16"
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=4)

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional debug limit. If omitted, use all rollout samples."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Keep 0 for h5py safety unless you know your setup supports workers."
    )

    parser.add_argument(
        "--dx",
        type=float,
        default=1.0,
        help="Grid spacing along x direction. Default uses index-space spacing."
    )

    parser.add_argument(
        "--dy",
        type=float,
        default=1.0,
        help="Grid spacing along y direction. Default uses index-space spacing."
    )

    parser.add_argument(
        "--axis_convention",
        type=str,
        default="height_width",
        choices=["height_width", "width_height"],
        help=(
            "height_width: tensor [H,W], x derivative along H, y derivative along W. "
            "width_height: tensor [H,W], x derivative along W, y derivative along H."
        )
    )

    return parser.parse_args()


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


def parse_ra_pr_from_group(group_name):
    """
    Expected group examples:
        ra_1e6_pr_0.5
        ra_1e8_pr_2
    """
    match = re.search(r"ra_([0-9.eE+-]+)_pr_([0-9.eE+-]+)", group_name)
    if match is None:
        raise ValueError(f"Cannot parse Ra/Pr from group name: {group_name}")

    ra = float(match.group(1))
    pr = float(match.group(2))
    return ra, pr


def make_param_from_group(group_name):
    ra, pr = parse_ra_pr_from_group(group_name)

    return torch.tensor(
        [math.log10(float(ra)), math.log10(float(pr))],
        dtype=torch.float32,
    )


class CrossParamRolloutDataset(Dataset):
    def __init__(
        self,
        split_path,
        split_key="test",
        max_horizon=16,
        stride=4,
        max_samples=None,
    ):
        super().__init__()

        self.split_path = split_path
        self.split_key = split_key
        self.max_horizon = max_horizon
        self.stride = stride

        with open(split_path, "r", encoding="utf-8") as f:
            split_config = json.load(f)

        if split_key not in split_config:
            raise KeyError(f"Split file does not contain key: {split_key}")

        split_items = split_config[split_key]

        self.samples = []

        with h5py.File(DATA_PATH, "r") as f:
            for item in split_items:
                group_name = item["group"]
                trajectories = item["trajectories"]

                if group_name not in f:
                    raise KeyError(f"Group {group_name} not found in HDF5")

                group = f[group_name]

                # Use the first field to infer trajectory time length.
                first_field = FIELD_ORDER[0]

                for traj_idx in trajectories:
                    total_T = group[first_field].shape[1]

                    max_t0 = total_T - CONTEXT_LENGTH - max_horizon

                    if max_t0 < 0:
                        continue

                    for t0 in range(0, max_t0 + 1, stride):
                        self.samples.append(
                            {
                                "group": group_name,
                                "trajectory": int(traj_idx),
                                "t0": int(t0),
                            }
                        )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"📦 Built rollout physics dataset:")
        print(f"   split = {split_key}")
        print(f"   samples = {len(self.samples)}")
        print(f"   max_horizon = {self.max_horizon}")
        print(f"   stride = {self.stride}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]

        group_name = meta["group"]
        traj_idx = meta["trajectory"]
        t0 = meta["t0"]

        start = t0
        end = t0 + CONTEXT_LENGTH + self.max_horizon

        with h5py.File(DATA_PATH, "r") as f:
            group = f[group_name]

            field_arrays = []
            for field in FIELD_ORDER:
                arr = group[field][traj_idx, start:end, :, :]
                field_arrays.append(arr)

        # [C, T, H, W] -> [T, C, H, W]
        traj_data = torch.tensor(
            field_arrays,
            dtype=torch.float32,
        ).permute(1, 0, 2, 3)

        history = traj_data[:CONTEXT_LENGTH]                 # [4, 4, H, W]
        future = traj_data[CONTEXT_LENGTH:]                  # [max_horizon, 4, H, W]

        _, _, H, W = history.shape

        x0_phys = history.reshape(CONTEXT_LENGTH * len(FIELD_ORDER), H, W)
        param = make_param_from_group(group_name)

        return {
            "x0_phys": x0_phys,
            "future_phys": future,
            "param": param,
        }


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f"✅ Loaded checkpoint: {checkpoint_path}")
    return model


@torch.no_grad()
def rollout_m0(x0_phys, max_horizon):
    """
    Persistence baseline:
    Always predict the last frame in the initial context.
    """
    last_frame = x0_phys[:, -len(FIELD_ORDER):, :, :]
    preds = []

    for _ in range(max_horizon):
        preds.append(last_frame.clone())

    return preds


@torch.no_grad()
def rollout_m3_delta(model, x0_phys, normalizer, max_horizon):
    """
    M3-Delta:
    model predicts normalized delta.
    pred_y_norm = x_last_norm + pred_delta_norm
    """
    preds = []

    current_x_norm = normalizer.normalize_x(x0_phys)

    for _ in range(max_horizon):
        x_last_norm = current_x_norm[:, -len(FIELD_ORDER):, :, :]

        pred_delta_norm = model(current_x_norm)
        pred_y_norm = x_last_norm + pred_delta_norm

        pred_y_phys = normalizer.denormalize_y(pred_y_norm)

        preds.append(pred_y_phys)

        current_x_norm = torch.cat(
            [
                current_x_norm[:, len(FIELD_ORDER):, :, :],
                pred_y_norm,
            ],
            dim=1,
        )

    return preds


@torch.no_grad()
def rollout_m3_delta_film(model, x0_phys, param, normalizer, max_horizon):
    """
    FiLM-conditioned M3-Delta:
    model predicts normalized delta with param conditioning.
    """
    preds = []

    current_x_norm = normalizer.normalize_x(x0_phys)

    for _ in range(max_horizon):
        x_last_norm = current_x_norm[:, -len(FIELD_ORDER):, :, :]

        pred_delta_norm = model(current_x_norm, param)
        pred_y_norm = x_last_norm + pred_delta_norm

        pred_y_phys = normalizer.denormalize_y(pred_y_norm)

        preds.append(pred_y_phys)

        current_x_norm = torch.cat(
            [
                current_x_norm[:, len(FIELD_ORDER):, :, :],
                pred_y_norm,
            ],
            dim=1,
        )

    return preds


def spatial_derivatives_height_width(ux, uy, dx=1.0, dy=1.0):
    """
    Input:
        ux, uy: [B, H, W]

    Convention:
        x derivative along H dimension
        y derivative along W dimension

    Output all tensors are [B, H-2, W-2]
    """
    dux_dx = (ux[:, 2:, 1:-1] - ux[:, :-2, 1:-1]) / (2.0 * dx)
    duy_dy = (uy[:, 1:-1, 2:] - uy[:, 1:-1, :-2]) / (2.0 * dy)

    duy_dx = (uy[:, 2:, 1:-1] - uy[:, :-2, 1:-1]) / (2.0 * dx)
    dux_dy = (ux[:, 1:-1, 2:] - ux[:, 1:-1, :-2]) / (2.0 * dy)

    divergence = dux_dx + duy_dy
    vorticity = duy_dx - dux_dy

    return divergence, vorticity


def spatial_derivatives_width_height(ux, uy, dx=1.0, dy=1.0):
    """
    Input:
        ux, uy: [B, H, W]

    Convention:
        x derivative along W dimension
        y derivative along H dimension

    Output all tensors are [B, H-2, W-2]
    """
    dux_dx = (ux[:, 1:-1, 2:] - ux[:, 1:-1, :-2]) / (2.0 * dx)
    duy_dy = (uy[:, 2:, 1:-1] - uy[:, :-2, 1:-1]) / (2.0 * dy)

    duy_dx = (uy[:, 1:-1, 2:] - uy[:, 1:-1, :-2]) / (2.0 * dx)
    dux_dy = (ux[:, 2:, 1:-1] - ux[:, :-2, 1:-1]) / (2.0 * dy)

    divergence = dux_dx + duy_dy
    vorticity = duy_dx - dux_dy

    return divergence, vorticity


def compute_div_vort(state, axis_convention, dx, dy):
    """
    state: [B, 4, H, W]
    """
    ux = state[:, UX_IDX, :, :]
    uy = state[:, UY_IDX, :, :]

    if axis_convention == "height_width":
        return spatial_derivatives_height_width(ux, uy, dx=dx, dy=dy)
    elif axis_convention == "width_height":
        return spatial_derivatives_width_height(ux, uy, dx=dx, dy=dy)
    else:
        raise ValueError(f"Unknown axis_convention: {axis_convention}")


def new_accumulator():
    return {
        "num_samples": 0,
        "num_points": 0,

        "div_pred_abs_sum": 0.0,
        "div_gt_abs_sum": 0.0,
        "div_error_abs_sum": 0.0,

        "div_pred_sq_sum": 0.0,
        "div_gt_sq_sum": 0.0,
        "div_error_sq_sum": 0.0,

        "vort_error_sq_sum": 0.0,
        "vort_gt_sq_sum": 0.0,
        "vort_mse_sum": 0.0,
    }


def update_metrics(acc, pred_state, gt_state, axis_convention, dx, dy):
    """
    pred_state, gt_state: [B, 4, H, W]
    """
    div_pred, vort_pred = compute_div_vort(
        pred_state,
        axis_convention=axis_convention,
        dx=dx,
        dy=dy,
    )

    div_gt, vort_gt = compute_div_vort(
        gt_state,
        axis_convention=axis_convention,
        dx=dx,
        dy=dy,
    )

    div_error = div_pred - div_gt
    vort_error = vort_pred - vort_gt

    num_samples = pred_state.shape[0]
    num_points = div_pred.numel()

    acc["num_samples"] += num_samples
    acc["num_points"] += num_points

    acc["div_pred_abs_sum"] += torch.sum(torch.abs(div_pred)).item()
    acc["div_gt_abs_sum"] += torch.sum(torch.abs(div_gt)).item()
    acc["div_error_abs_sum"] += torch.sum(torch.abs(div_error)).item()

    acc["div_pred_sq_sum"] += torch.sum(div_pred ** 2).item()
    acc["div_gt_sq_sum"] += torch.sum(div_gt ** 2).item()
    acc["div_error_sq_sum"] += torch.sum(div_error ** 2).item()

    acc["vort_error_sq_sum"] += torch.sum(vort_error ** 2).item()
    acc["vort_gt_sq_sum"] += torch.sum(vort_gt ** 2).item()
    acc["vort_mse_sum"] += torch.sum(vort_error ** 2).item()


def finalize_row(split_name, model_name, horizon, acc):
    eps = 1e-12
    num_points = max(acc["num_points"], 1)

    div_pred_mae = acc["div_pred_abs_sum"] / num_points
    div_gt_mae = acc["div_gt_abs_sum"] / num_points
    div_error_mae = acc["div_error_abs_sum"] / num_points

    div_pred_mse = acc["div_pred_sq_sum"] / num_points
    div_gt_mse = acc["div_gt_sq_sum"] / num_points
    div_error_mse = acc["div_error_sq_sum"] / num_points

    vorticity_mse = acc["vort_mse_sum"] / num_points
    vorticity_rel_l2_percent = math.sqrt(
        acc["vort_error_sq_sum"] / (acc["vort_gt_sq_sum"] + eps)
    ) * 100.0

    return {
        "split_name": split_name,
        "model_name": model_name,
        "horizon": horizon,
        "num_samples": acc["num_samples"],

        "div_pred_mae": div_pred_mae,
        "div_gt_mae": div_gt_mae,
        "div_error_mae": div_error_mae,

        "div_pred_mse": div_pred_mse,
        "div_gt_mse": div_gt_mse,
        "div_error_mse": div_error_mse,

        "vorticity_rel_l2_percent": vorticity_rel_l2_percent,
        "vorticity_mse": vorticity_mse,
    }


def infer_split_name(split_path):
    name = os.path.basename(split_path)
    name = name.replace("_split.json", "")
    name = name.replace(".json", "")
    return name


def main():
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    split_path = resolve_path(project_root, args.split)
    stats_path = resolve_path(project_root, args.stats)
    m3_delta_ckpt = resolve_path(project_root, args.m3_delta_checkpoint)
    film_ckpt = resolve_path(project_root, args.film_checkpoint)
    output_path = resolve_path(project_root, args.output)

    horizons = [int(x) for x in args.horizons.split(",")]
    max_horizon = max(horizons)
    split_name = infer_split_name(split_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 [Cross-Parameter Rollout Physics Diagnostics]")
    print(f"📌 Device: {device}")
    print(f"📌 Split: {split_path}")
    print(f"📌 Stats: {stats_path}")
    print(f"📌 Split name: {split_name}")
    print(f"📌 Horizons: {horizons}")
    print(f"📌 Stride: {args.stride}")
    print(f"📌 Batch size: {args.batch_size}")
    print(f"📌 Axis convention: {args.axis_convention}")
    print(f"📌 dx={args.dx}, dy={args.dy}")
    print(f"📌 Output: {output_path}")

    normalizer = FieldWiseNormalizer(stats_path=stats_path, device=device)

    dataset = CrossParamRolloutDataset(
        split_path=split_path,
        split_key="test",
        max_horizon=max_horizon,
        stride=args.stride,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    m3_delta = PlainFNO2d(
        in_channels=CONTEXT_LENGTH * len(FIELD_ORDER),
        out_channels=len(FIELD_ORDER),
        modes1=16,
        modes2=16,
        width=32,
    ).to(device)

    m3_delta = load_checkpoint(m3_delta, m3_delta_ckpt, device)

    film = FiLMFNO2d(
        in_channels=CONTEXT_LENGTH * len(FIELD_ORDER),
        out_channels=len(FIELD_ORDER),
        modes1=16,
        modes2=16,
        width=32,
        film_hidden_dim=64,
    ).to(device)

    film = load_checkpoint(film, film_ckpt, device)

    model_names = [
        "M0",
        "M3-Delta",
        "M3-Delta-FiLM",
    ]

    accum = {
        model_name: {
            h: new_accumulator()
            for h in horizons
        }
        for model_name in model_names
    }

    m3_delta.eval()
    film.eval()

    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        x0_phys = batch["x0_phys"].to(device, non_blocking=True)
        future_phys = batch["future_phys"].to(device, non_blocking=True)
        param = batch["param"].to(device, non_blocking=True)

        with torch.no_grad():
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

            seq_film = rollout_m3_delta_film(
                model=film,
                x0_phys=x0_phys,
                param=param,
                normalizer=normalizer,
                max_horizon=max_horizon,
            )

        pred_dict = {
            "M0": seq_m0,
            "M3-Delta": seq_m3_delta,
            "M3-Delta-FiLM": seq_film,
        }

        for h in horizons:
            idx = h - 1
            gt_state = future_phys[:, idx, :, :, :]

            for model_name in model_names:
                pred_state = pred_dict[model_name][idx]

                update_metrics(
                    acc=accum[model_name][h],
                    pred_state=pred_state,
                    gt_state=gt_state,
                    axis_convention=args.axis_convention,
                    dx=args.dx,
                    dy=args.dy,
                )

        if batch_idx % 10 == 0 or batch_idx == total_batches:
            print(f"   processed batch {batch_idx}/{total_batches}")

    rows = []

    for model_name in model_names:
        for h in horizons:
            rows.append(
                finalize_row(
                    split_name=split_name,
                    model_name=model_name,
                    horizon=h,
                    acc=accum[model_name][h],
                )
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "split_name",
        "model_name",
        "horizon",
        "num_samples",

        "div_pred_mae",
        "div_gt_mae",
        "div_error_mae",

        "div_pred_mse",
        "div_gt_mse",
        "div_error_mse",

        "vorticity_rel_l2_percent",
        "vorticity_mse",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print("\n✅ Physics diagnostics saved:")
    print(f"   {output_path}")

    print("\n📊 Summary:")
    header = (
        f"{'model':<18} {'h':>4} "
        f"{'div_pred_mae':>14} {'div_err_mae':>14} "
        f"{'vort_rel_l2%':>14}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['model_name']:<18} {row['horizon']:>4} "
            f"{row['div_pred_mae']:>14.6e} "
            f"{row['div_error_mae']:>14.6e} "
            f"{row['vorticity_rel_l2_percent']:>14.6f}"
        )


if __name__ == "__main__":
    main()
