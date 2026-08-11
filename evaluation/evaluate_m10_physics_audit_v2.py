import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from constants import (
    DATA_PATH,
    FIELD_ORDER,
    CONTEXT_LENGTH,
    DTYPE,
)

from training.normalization import FieldWiseNormalizer


FIELD_TO_IDX = {
    name: FIELD_ORDER.index(name)
    for name in FIELD_ORDER
}

B_IDX = FIELD_TO_IDX["buoyancy"]
UX_IDX = FIELD_TO_IDX["u_x"]
UY_IDX = FIELD_TO_IDX["u_y"]
P_IDX = FIELD_TO_IDX["pressure"]


# ============================================================
# Reuse CLOSED M10 evaluators.
#
# IMPORTANT:
# model_dx / model_dy belong to the already-trained M10 model.
# physics_dx / physics_dy belong ONLY to physics diagnostics.
# ============================================================

BASE_EVAL_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_cross_param_rollout_m10_2path.py",
)

RMSCAP_EVAL_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_m10_1a_stateparam_rmscap_rollout.py",
)


def load_python_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


base = load_python_module(
    "m10_closed_rollout_eval",
    BASE_EVAL_PATH,
)

rmscap_eval = load_python_module(
    "m10_1a_closed_rollout_eval",
    RMSCAP_EVAL_PATH,
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "M10 Physics Audit v2: "
            "M6 vs M10-StateParam vs M10-1a-StateParam-RMSCap. "
            "No training. "
            "Model discretization and physics-audit discretization "
            "are explicitly separated. PDE quantities are treated as "
            "FD-based comparative proxies, not solver-level PDE violation."
        )
    )

    parser.add_argument("--split", required=True)
    parser.add_argument("--stats", required=True)

    parser.add_argument(
        "--split_label",
        required=True,
        choices=["unseen_pr", "unseen_ra"],
    )

    parser.add_argument(
        "--split_key",
        default="test",
        choices=["train", "val", "valid", "validation", "test"],
    )

    parser.add_argument("--seed", required=True, type=int)

    parser.add_argument("--m6_checkpoint", required=True)

    parser.add_argument(
        "--m10_stateparam_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--m10_1a_checkpoint",
        required=True,
    )

    # --------------------------------------------------------
    # Frozen model discretization.
    #
    # DO NOT change these when evaluating old M10 checkpoints.
    # --------------------------------------------------------

    parser.add_argument(
        "--model_dx",
        type=float,
        default=1.0 / 64.0,
    )

    parser.add_argument(
        "--model_dy",
        type=float,
        default=1.0 / 64.0,
    )

    # --------------------------------------------------------
    # Physical diagnostic discretization.
    # --------------------------------------------------------

    parser.add_argument(
        "--physics_dx",
        type=float,
        default=1.0 / 64.0,
        help=(
            "Declared X spacing for FD-based physics proxies only. "
            "Default 1/64 is the internally consistent periodic 256-point "
            "proxy grid on the x-domain of length 4; it is NOT claimed to "
            "reproduce the original solver discretization."
        ),
    )

    parser.add_argument(
        "--physics_dy",
        type=float,
        default=1.0 / 63.0,
        help=(
            "Declared Y spacing for FD-based physics proxies only. "
            "Default 1/63 corresponds to 64 uniform points including "
            "both y=0 and y=1 boundaries."
        ),
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.25,
        help=(
            "Nominal one-frame time increment used by the FD-based "
            "transition proxy. The same dt is applied to prediction and GT. "
            "This is not interpreted as a solver-level residual calibration."
        ),
    )

    parser.add_argument(
        "--path_b_rms_cap",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--path_b_rms_eps",
        type=float,
        default=1.0e-12,
    )

    parser.add_argument(
        "--alpha_max",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--conditioner_hidden",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--horizons",
        default="1,4,8,16",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help=(
            "Temporal-start stride for physics audit. "
            "Default 4 follows earlier physics-evaluation practice."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/tables/m10_physics_audit_v2",
    )

    return parser.parse_args()


# ============================================================
# Utilities / provenance
# ============================================================

def resolve_path(path):
    if os.path.isabs(path):
        return path

    return os.path.abspath(
        os.path.join(PROJECT_ROOT, path)
    )


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def canonical_split_key(config, requested):
    if requested in config:
        return requested

    aliases = {
        "val": ["valid", "validation"],
        "valid": ["val", "validation"],
        "validation": ["val", "valid"],
    }

    for candidate in aliases.get(requested, []):
        if candidate in config:
            return candidate

    raise KeyError(
        f"Split key '{requested}' not found. "
        f"Available keys: {list(config.keys())}"
    )


# ============================================================
# Dataset
# ============================================================

class PhysicsAuditDataset(Dataset):
    """
    Final tensor convention:

        [trajectory, time, field, X, Y]

    X is periodic horizontal direction.
    Y is vertical non-periodic direction.
    """

    def __init__(
        self,
        split_path,
        split_key,
        max_horizon,
        stride,
        max_samples=None,
    ):
        super().__init__()

        self.max_horizon = int(max_horizon)
        self.stride = int(stride)

        if self.stride < 1:
            raise ValueError("stride must be >= 1")

        with open(
            split_path,
            "r",
            encoding="utf-8",
        ) as f:
            split_config = json.load(f)

        actual_key = canonical_split_key(
            split_config,
            split_key,
        )

        split_items = split_config[actual_key]

        self.data_store = {}
        self.index = []

        print(
            f"📦 Loading Physics-Audit data | "
            f"split_key={actual_key}"
        )

        with h5py.File(DATA_PATH, "r") as f:

            for item in split_items:

                group_name = item["group"]
                traj_indices = item["trajectories"]

                if group_name not in f:
                    raise KeyError(
                        f"Missing HDF5 group: {group_name}"
                    )

                group = f[group_name]

                fields_data = [
                    group[field][:]
                    for field in FIELD_ORDER
                ]

                stacked = torch.tensor(
                    np.stack(
                        fields_data,
                        axis=0,
                    ),
                    dtype=DTYPE,
                )

                # [field, traj, time, X, Y]
                selected = stacked[
                    :,
                    traj_indices,
                ]

                # [traj, time, field, X, Y]
                data = selected.permute(
                    1, 2, 0, 3, 4
                ).contiguous()

                if data.shape[-2:] != (256, 64):
                    raise RuntimeError(
                        f"{group_name}: expected "
                        f"[X,Y]=[256,64], got "
                        f"{tuple(data.shape[-2:])}"
                    )

                self.data_store[group_name] = data

                param = base.RolloutDataset._make_param(
                    group_name
                )

                num_traj = data.shape[0]
                num_steps = data.shape[1]

                max_start = (
                    num_steps
                    - CONTEXT_LENGTH
                    - self.max_horizon
                    + 1
                )

                if max_start <= 0:
                    raise RuntimeError(
                        f"{group_name}: not enough "
                        f"time steps for horizon "
                        f"{self.max_horizon}"
                    )

                for local_traj_idx in range(
                    num_traj
                ):
                    for t0 in range(
                        0,
                        max_start,
                        self.stride,
                    ):
                        self.index.append(
                            {
                                "group": group_name,
                                "traj": local_traj_idx,
                                "t0": t0,
                                "param": param,
                            }
                        )

        if max_samples is not None:
            self.index = self.index[
                : int(max_samples)
            ]

        print(
            f"✅ Physics-Audit samples: "
            f"{len(self.index)}"
        )
        print(
            f"📌 stride={self.stride}, "
            f"max_horizon={self.max_horizon}"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        meta = self.index[idx]

        data = self.data_store[
            meta["group"]
        ]

        t0 = meta["t0"]

        history = data[
            meta["traj"],
            t0 : t0 + CONTEXT_LENGTH,
        ]

        future = data[
            meta["traj"],
            t0 + CONTEXT_LENGTH :
            t0 + CONTEXT_LENGTH
            + self.max_horizon,
        ]

        _, _, X, Y = history.shape

        x0_phys = history.reshape(
            CONTEXT_LENGTH * 4,
            X,
            Y,
        )

        return (
            x0_phys,
            future,
            meta["param"],
        )


# ============================================================
# Physical derivatives
#
# Tensor:
#     [B, X, Y]
#
# X:
#     periodic central difference
#
# Y:
#     non-periodic;
#     physics metrics evaluated on interior y=1..Y-2.
#
# This avoids contaminating FD-based physics proxies with
# one-sided boundary finite differences.
# ============================================================

def ddx_periodic(f, dx):
    return (
        torch.roll(
            f,
            shifts=-1,
            dims=-2,
        )
        - torch.roll(
            f,
            shifts=1,
            dims=-2,
        )
    ) / (2.0 * dx)


def ddxx_periodic(f, dx):
    return (
        torch.roll(
            f,
            shifts=-1,
            dims=-2,
        )
        - 2.0 * f
        + torch.roll(
            f,
            shifts=1,
            dims=-2,
        )
    ) / (dx * dx)


def ddy_interior(f, dy):
    return (
        f[..., 2:]
        - f[..., :-2]
    ) / (2.0 * dy)


def ddyy_interior(f, dy):
    return (
        f[..., 2:]
        - 2.0 * f[..., 1:-1]
        + f[..., :-2]
    ) / (dy * dy)


def grad_interior(f, dx, dy):
    fx = ddx_periodic(
        f,
        dx,
    )[..., 1:-1]

    fy = ddy_interior(
        f,
        dy,
    )

    return fx, fy


def laplacian_interior(f, dx, dy):
    f_xx = ddxx_periodic(
        f,
        dx,
    )[..., 1:-1]

    f_yy = ddyy_interior(
        f,
        dy,
    )

    return f_xx + f_yy


# ============================================================
# State physics
# ============================================================

def compute_state_physics(
    state,
    dx,
    dy,
):
    """
    state:
        [B, 4, X, Y]
    """

    b = state[:, B_IDX]
    ux = state[:, UX_IDX]
    uy = state[:, UY_IDX]
    p = state[:, P_IDX]

    ux_x, ux_y = grad_interior(
        ux, dx, dy
    )

    uy_x, uy_y = grad_interior(
        uy, dx, dy
    )

    b_x, b_y = grad_interior(
        b, dx, dy
    )

    p_x, p_y = grad_interior(
        p, dx, dy
    )

    ux_i = ux[..., 1:-1]
    uy_i = uy[..., 1:-1]

    divergence = ux_x + uy_y

    vorticity = uy_x - ux_y

    adv_b = (
        ux_i * b_x
        + uy_i * b_y
    )

    grad_p = torch.stack(
        [p_x, p_y],
        dim=1,
    )

    # Exact Path-A-shaped state quantity:
    # b' = b - <b>_x
    b_prime = (
        b
        - b.mean(
            dim=-2,
            keepdim=True,
        )
    )

    return {
        "divergence": divergence,
        "vorticity": vorticity,
        "grad_p": grad_p,
        "adv_b": adv_b,
        "b_prime": b_prime,
    }


# ============================================================
# FD-based PDE proxy
#
# IMPORTANT INTERPRETATION:
#
# These quantities are NOT solver-level PDE violation metrics.
# Native official GT calibration showed that ordinary second-order
# finite differences on the released fields do not produce a
# near-zero vertical-momentum residual. Therefore:
#
#   1) absolute GT FD residual RMS is reported as calibration;
#   2) the primary comparative quantity is
#          ||R_pred^FD - R_GT^FD||;
#   3) a dimensionless normalized mismatch is computed with the
#      same-unit GT PDE-term scale:
#
#      NR = sqrt(
#          sum (R_pred^FD - R_GT^FD)^2
#          / sum_j sum (T_j,GT^FD)^2
#      )
#
# where T_j are terms from the SAME equation and therefore have
# consistent physical units. This preserves dimensional consistency.
#
# Discrete transition n -> n+1:
#
# R_b
#   = (b_{n+1}-b_n)/dt
#     - kappa Lap(b_n)
#     + u_n · grad(b_n)
#
# R_ux
#   = (ux_{n+1}-ux_n)/dt
#     - nu Lap(ux_n)
#     + dp/dx
#     + u_n · grad(ux_n)
#
# R_uy
#   = (uy_{n+1}-uy_n)/dt
#     - nu Lap(uy_n)
#     + dp/dy
#     - b_n
#     + u_n · grad(uy_n)
#
# The SAME FD operator is applied to prediction and GT.
# ============================================================

def compute_pde_residuals(
    prev_state,
    next_state,
    param,
    dx,
    dy,
    dt,
):
    b0 = prev_state[:, B_IDX]
    ux0 = prev_state[:, UX_IDX]
    uy0 = prev_state[:, UY_IDX]
    p0 = prev_state[:, P_IDX]

    b1 = next_state[:, B_IDX]
    ux1 = next_state[:, UX_IDX]
    uy1 = next_state[:, UY_IDX]

    b_x, b_y = grad_interior(
        b0, dx, dy
    )

    ux_x, ux_y = grad_interior(
        ux0, dx, dy
    )

    uy_x, uy_y = grad_interior(
        uy0, dx, dy
    )

    p_x, p_y = grad_interior(
        p0, dx, dy
    )

    lap_b = laplacian_interior(
        b0, dx, dy
    )

    lap_ux = laplacian_interior(
        ux0, dx, dy
    )

    lap_uy = laplacian_interior(
        uy0, dx, dy
    )

    b_i = b0[..., 1:-1]
    ux_i = ux0[..., 1:-1]
    uy_i = uy0[..., 1:-1]

    db_dt = (
        b1[..., 1:-1]
        - b0[..., 1:-1]
    ) / dt

    dux_dt = (
        ux1[..., 1:-1]
        - ux0[..., 1:-1]
    ) / dt

    duy_dt = (
        uy1[..., 1:-1]
        - uy0[..., 1:-1]
    ) / dt

    adv_b = (
        ux_i * b_x
        + uy_i * b_y
    )

    adv_ux = (
        ux_i * ux_x
        + uy_i * ux_y
    )

    adv_uy = (
        ux_i * uy_x
        + uy_i * uy_y
    )

    log_ra = param[:, 0]
    log_pr = param[:, 1]

    ra = torch.pow(
        torch.tensor(
            10.0,
            device=param.device,
            dtype=param.dtype,
        ),
        log_ra,
    )

    pr = torch.pow(
        torch.tensor(
            10.0,
            device=param.device,
            dtype=param.dtype,
        ),
        log_pr,
    )

    kappa = torch.pow(
        ra * pr,
        -0.5,
    ).view(-1, 1, 1)

    nu = torch.pow(
        ra / pr,
        -0.5,
    ).view(-1, 1, 1)

    # Same-unit equation terms.
    t_b_dt = db_dt
    t_b_diff = -kappa * lap_b
    t_b_adv = adv_b

    t_ux_dt = dux_dt
    t_ux_diff = -nu * lap_ux
    t_ux_p = p_x
    t_ux_adv = adv_ux

    t_uy_dt = duy_dt
    t_uy_diff = -nu * lap_uy
    t_uy_p = p_y
    t_uy_b = -b_i
    t_uy_adv = adv_uy

    r_b = (
        t_b_dt
        + t_b_diff
        + t_b_adv
    )

    r_ux = (
        t_ux_dt
        + t_ux_diff
        + t_ux_p
        + t_ux_adv
    )

    r_uy = (
        t_uy_dt
        + t_uy_diff
        + t_uy_p
        + t_uy_b
        + t_uy_adv
    )

    r_u = torch.stack(
        [r_ux, r_uy],
        dim=1,
    )

    # Per-point sum of squared same-unit GT terms.
    # These tensors are later used only from the GT branch to build
    # a dimensionally consistent scale for normalized mismatch.
    term_energy_b = (
        t_b_dt * t_b_dt
        + t_b_diff * t_b_diff
        + t_b_adv * t_b_adv
    )

    term_energy_ux = (
        t_ux_dt * t_ux_dt
        + t_ux_diff * t_ux_diff
        + t_ux_p * t_ux_p
        + t_ux_adv * t_ux_adv
    )

    term_energy_uy = (
        t_uy_dt * t_uy_dt
        + t_uy_diff * t_uy_diff
        + t_uy_p * t_uy_p
        + t_uy_b * t_uy_b
        + t_uy_adv * t_uy_adv
    )

    term_energy_u = torch.stack(
        [term_energy_ux, term_energy_uy],
        dim=1,
    )

    return {
        "r_b": r_b,
        "r_ux": r_ux,
        "r_uy": r_uy,
        "r_u": r_u,
        "term_energy_b": term_energy_b,
        "term_energy_ux": term_energy_ux,
        "term_energy_uy": term_energy_uy,
        "term_energy_u": term_energy_u,
    }


# ============================================================
# Accumulators
# ============================================================

def new_bucket():
    return {}


def add_value(bucket, key, value):
    bucket[key] = (
        bucket.get(key, 0.0)
        + float(value)
    )


def update_rel(
    bucket,
    prefix,
    pred,
    gt,
):
    pred64 = pred.double()
    gt64 = gt.double()

    diff = pred64 - gt64

    add_value(
        bucket,
        f"{prefix}_err_sq",
        torch.sum(diff * diff).item(),
    )

    add_value(
        bucket,
        f"{prefix}_gt_sq",
        torch.sum(gt64 * gt64).item(),
    )

    add_value(
        bucket,
        f"{prefix}_n",
        diff.numel(),
    )


def update_mae_triplet(
    bucket,
    prefix,
    pred,
    gt,
):
    pred64 = pred.double()
    gt64 = gt.double()

    add_value(
        bucket,
        f"{prefix}_pred_abs",
        torch.sum(
            torch.abs(pred64)
        ).item(),
    )

    add_value(
        bucket,
        f"{prefix}_gt_abs",
        torch.sum(
            torch.abs(gt64)
        ).item(),
    )

    add_value(
        bucket,
        f"{prefix}_err_abs",
        torch.sum(
            torch.abs(
                pred64 - gt64
            )
        ).item(),
    )

    add_value(
        bucket,
        f"{prefix}_n",
        pred64.numel(),
    )


def update_rms_triplet(
    bucket,
    prefix,
    pred,
    gt,
):
    pred64 = pred.double()
    gt64 = gt.double()

    diff = pred64 - gt64

    add_value(
        bucket,
        f"{prefix}_pred_sq",
        torch.sum(
            pred64 * pred64
        ).item(),
    )

    add_value(
        bucket,
        f"{prefix}_gt_sq",
        torch.sum(
            gt64 * gt64
        ).item(),
    )

    add_value(
        bucket,
        f"{prefix}_err_sq",
        torch.sum(
            diff * diff
        ).item(),
    )

    add_value(
        bucket,
        f"{prefix}_n",
        pred64.numel(),
    )


def update_pde_proxy(
    bucket,
    prefix,
    pred_residual,
    gt_residual,
    gt_term_energy,
):
    """
    Accumulate an FD-based PDE comparison.

    Primary comparative numerator:
        ||R_pred^FD - R_GT^FD||

    Dimensionally consistent denominator:
        sqrt(sum_j ||T_j,GT^FD||^2)

    All T_j belong to the same equation and therefore have
    compatible physical units.
    """

    update_rms_triplet(
        bucket,
        prefix,
        pred_residual,
        gt_residual,
    )

    gt_term_energy64 = gt_term_energy.double()

    add_value(
        bucket,
        f"{prefix}_gt_term_energy_sum",
        torch.sum(
            gt_term_energy64
        ).item(),
    )


# ============================================================
# Main metric update
# ============================================================

def update_physics_metrics(
    bucket,
    pred_prev,
    pred_next,
    gt_prev,
    gt_next,
    param,
    dx,
    dy,
    dt,
):
    # --------------------------------------------------------
    # State / field errors
    # --------------------------------------------------------

    update_rel(
        bucket,
        "global",
        pred_next,
        gt_next,
    )

    for field_name, idx in FIELD_TO_IDX.items():
        update_rel(
            bucket,
            field_name,
            pred_next[:, idx],
            gt_next[:, idx],
        )

    # Pressure gauge:
    # remove each frame's spatial mean independently.
    pred_p = pred_next[:, P_IDX]
    gt_p = gt_next[:, P_IDX]

    pred_p0 = (
        pred_p
        - pred_p.mean(
            dim=(-2, -1),
            keepdim=True,
        )
    )

    gt_p0 = (
        gt_p
        - gt_p.mean(
            dim=(-2, -1),
            keepdim=True,
        )
    )

    update_rel(
        bucket,
        "pressure_gauge",
        pred_p0,
        gt_p0,
    )

    # --------------------------------------------------------
    # State physics mechanisms
    # --------------------------------------------------------

    pred_phys = compute_state_physics(
        pred_next,
        dx,
        dy,
    )

    gt_phys = compute_state_physics(
        gt_next,
        dx,
        dy,
    )

    update_mae_triplet(
        bucket,
        "div",
        pred_phys["divergence"],
        gt_phys["divergence"],
    )

    update_rel(
        bucket,
        "vorticity",
        pred_phys["vorticity"],
        gt_phys["vorticity"],
    )

    update_rel(
        bucket,
        "grad_p",
        pred_phys["grad_p"],
        gt_phys["grad_p"],
    )

    update_rel(
        bucket,
        "adv_b",
        pred_phys["adv_b"],
        gt_phys["adv_b"],
    )

    update_rel(
        bucket,
        "b_prime",
        pred_phys["b_prime"],
        gt_phys["b_prime"],
    )

    # --------------------------------------------------------
    # Transition PDE residuals
    # --------------------------------------------------------

    pred_res = compute_pde_residuals(
        pred_prev,
        pred_next,
        param,
        dx,
        dy,
        dt,
    )

    gt_res = compute_pde_residuals(
        gt_prev,
        gt_next,
        param,
        dx,
        dy,
        dt,
    )

    pde_proxy_specs = [
        (
            "r_b",
            "term_energy_b",
        ),
        (
            "r_ux",
            "term_energy_ux",
        ),
        (
            "r_uy",
            "term_energy_uy",
        ),
        (
            "r_u",
            "term_energy_u",
        ),
    ]

    for residual_key, energy_key in pde_proxy_specs:
        update_pde_proxy(
            bucket,
            residual_key,
            pred_res[residual_key],
            gt_res[residual_key],
            gt_res[energy_key],
        )


# ============================================================
# Finalization
# ============================================================

def rel_percent(bucket, prefix):
    return (
        math.sqrt(
            bucket[f"{prefix}_err_sq"]
            / (
                bucket[f"{prefix}_gt_sq"]
                + 1.0e-30
            )
        )
        * 100.0
    )


def rms_value(bucket, prefix, kind):
    n = max(
        bucket[f"{prefix}_n"],
        1.0,
    )

    return math.sqrt(
        bucket[f"{prefix}_{kind}_sq"]
        / n
    )


def finalize_bucket(
    split_label,
    seed,
    model_name,
    horizon,
    bucket,
):
    row = {
        "split": split_label,
        "seed": seed,
        "model": model_name,
        "horizon": horizon,
    }

    # --------------------------------------------------------
    # Field prediction
    # --------------------------------------------------------

    for prefix in [
        "global",
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
        "pressure_gauge",
        "vorticity",
        "grad_p",
        "adv_b",
        "b_prime",
    ]:
        row[
            f"{prefix}_rel_l2_percent"
        ] = rel_percent(
            bucket,
            prefix,
        )

    # --------------------------------------------------------
    # Divergence
    # --------------------------------------------------------

    div_n = max(
        bucket["div_n"],
        1.0,
    )

    row["div_pred_mae"] = (
        bucket["div_pred_abs"]
        / div_n
    )

    row["div_gt_mae"] = (
        bucket["div_gt_abs"]
        / div_n
    )

    row["div_error_mae"] = (
        bucket["div_err_abs"]
        / div_n
    )

    # --------------------------------------------------------
    # FD-based PDE proxy
    #
    # Absolute pred/GT residual RMS are calibration quantities.
    # The primary comparative quantities are:
    #   *_error_rms
    #   *_normalized_mismatch
    #
    # We intentionally DO NOT report pred/GT "floor ratios":
    # native official GT does not provide a near-zero FD floor,
    # so such ratios are easy to misinterpret.
    # --------------------------------------------------------

    for prefix in [
        "r_b",
        "r_ux",
        "r_uy",
        "r_u",
    ]:
        pred_rms = rms_value(
            bucket,
            prefix,
            "pred",
        )

        gt_rms = rms_value(
            bucket,
            prefix,
            "gt",
        )

        err_rms = rms_value(
            bucket,
            prefix,
            "err",
        )

        gt_term_energy_sum = max(
            bucket[
                f"{prefix}_gt_term_energy_sum"
            ],
            1.0e-30,
        )

        normalized_mismatch = math.sqrt(
            bucket[f"{prefix}_err_sq"]
            / gt_term_energy_sum
        )

        gt_residual_to_term_scale = math.sqrt(
            bucket[f"{prefix}_gt_sq"]
            / gt_term_energy_sum
        )

        row[
            f"{prefix}_pred_rms"
        ] = pred_rms

        row[
            f"{prefix}_gt_rms"
        ] = gt_rms

        row[
            f"{prefix}_error_rms"
        ] = err_rms

        row[
            f"{prefix}_normalized_mismatch"
        ] = normalized_mismatch

        row[
            f"{prefix}_gt_residual_to_term_scale"
        ] = gt_residual_to_term_scale

    return row


# ============================================================
# Difference table
# ============================================================

LOWER_IS_BETTER = [
    "global_rel_l2_percent",
    "buoyancy_rel_l2_percent",
    "u_x_rel_l2_percent",
    "u_y_rel_l2_percent",
    "pressure_gauge_rel_l2_percent",
    "div_error_mae",
    "vorticity_rel_l2_percent",
    "grad_p_rel_l2_percent",
    "adv_b_rel_l2_percent",
    "b_prime_rel_l2_percent",
    "r_b_error_rms",
    "r_ux_error_rms",
    "r_uy_error_rms",
    "r_u_error_rms",
    "r_b_normalized_mismatch",
    "r_ux_normalized_mismatch",
    "r_uy_normalized_mismatch",
    "r_u_normalized_mismatch",
]


def make_difference_table(summary_df):
    pairs = [
        (
            "M10-StateParam",
            "M6",
        ),
        (
            "M10-1a-StateParam-RMSCap",
            "M10-StateParam",
        ),
        (
            "M10-1a-StateParam-RMSCap",
            "M6",
        ),
    ]

    rows = []

    for model_a, model_b in pairs:

        a = summary_df[
            summary_df["model"] == model_a
        ]

        b = summary_df[
            summary_df["model"] == model_b
        ]

        merged = a.merge(
            b,
            on=[
                "split",
                "seed",
                "horizon",
            ],
            suffixes=("_a", "_b"),
        )

        for _, r in merged.iterrows():

            row = {
                "split": r["split"],
                "seed": int(r["seed"]),
                "comparison": (
                    f"{model_a} - {model_b}"
                ),
                "horizon": int(
                    r["horizon"]
                ),
            }

            for metric in LOWER_IS_BETTER:
                row[
                    f"{metric}_diff"
                ] = (
                    r[f"{metric}_a"]
                    - r[f"{metric}_b"]
                )

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# One model rollout
# ============================================================

@torch.no_grad()
def run_model_rollout(
    model_name,
    model,
    x0_phys,
    future_phys,
    param,
    normalizer,
    max_horizon,
    physics_dx,
    physics_dy,
    dt,
    stats,
):
    x_norm = normalizer.normalize_x(
        x0_phys
    )

    # Last observed physical frame.
    pred_prev_phys = x0_phys[
        :,
        -len(FIELD_ORDER):,
        :,
        :,
    ]

    gt_prev_phys = pred_prev_phys

    for step in range(
        1,
        max_horizon + 1,
    ):

        current_norm = x_norm[
            :,
            -len(FIELD_ORDER):,
            :,
            :,
        ]

        if model_name == "M6":
            pred_delta_norm = model(
                x_norm
            )
        else:
            pred_delta_norm = model(
                x_norm,
                params=param,
            )

        pred_next_norm = (
            current_norm
            + pred_delta_norm
        )

        pred_next_phys = (
            normalizer.denormalize_y(
                pred_next_norm
            )
        )

        gt_next_phys = future_phys[
            :,
            step - 1,
        ]

        key = (
            model_name,
            step,
        )

        if key not in stats:
            stats[key] = new_bucket()

        update_physics_metrics(
            stats[key],
            pred_prev=pred_prev_phys,
            pred_next=pred_next_phys,
            gt_prev=gt_prev_phys,
            gt_next=gt_next_phys,
            param=param,
            dx=physics_dx,
            dy=physics_dy,
            dt=dt,
        )

        x_norm = torch.cat(
            [
                x_norm[
                    :,
                    len(FIELD_ORDER):,
                    :,
                    :,
                ],
                pred_next_norm,
            ],
            dim=1,
        )

        pred_prev_phys = pred_next_phys
        gt_prev_phys = gt_next_phys


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    requested_horizons = sorted(
        {
            int(x)
            for x in args.horizons.split(",")
            if x.strip()
        }
    )

    if not requested_horizons:
        raise ValueError(
            "No horizons requested."
        )

    max_horizon = max(
        requested_horizons
    )

    # --------------------------------------------------------
    # Frozen-model / physics-grid guardrails
    # --------------------------------------------------------

    if not math.isclose(
        args.model_dx,
        1.0 / 64.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        print(
            "⚠️ model_dx differs from "
            "historical M10 discretization."
        )

    if not math.isclose(
        args.model_dy,
        1.0 / 64.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        print(
            "⚠️ model_dy differs from "
            "historical M10 discretization."
        )

    split_path = resolve_path(
        args.split
    )

    stats_path = resolve_path(
        args.stats
    )

    m6_path = resolve_path(
        args.m6_checkpoint
    )

    m10_path = resolve_path(
        args.m10_stateparam_checkpoint
    )

    m10_1a_path = resolve_path(
        args.m10_1a_checkpoint
    )

    data_path = resolve_path(
        DATA_PATH
    )

    for path in [
        split_path,
        stats_path,
        m6_path,
        m10_path,
        m10_1a_path,
        data_path,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                path
            )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "🚀 [M10-Physics-Audit-v2]"
    )
    print(
        "📌 Type: diagnostic / "
        "control audit; NO TRAINING"
    )
    print(
        "📌 PDE semantics: FD-based comparative proxy; "
        "NOT solver-level PDE violation"
    )
    print(
        "📌 Primary PDE proxy: residual mismatch "
        "normalized by same-unit GT term scale"
    )
    print(
        f"📌 Device: {device}"
    )
    print(
        f"📌 Split: "
        f"{args.split_label}/{args.split_key}"
    )
    print(
        f"📌 Seed: {args.seed}"
    )
    print(
        f"📌 Horizons: "
        f"{requested_horizons}"
    )
    print(
        f"📌 Full curve: "
        f"h=1..{max_horizon}"
    )

    print()
    print(
        "========== DISCRETIZATION =========="
    )
    print(
        "Frozen M10 model grid:"
    )
    print(
        f"  model_dx = {args.model_dx}"
    )
    print(
        f"  model_dy = {args.model_dy}"
    )

    print(
        "Physics-Audit grid:"
    )
    print(
        f"  physics_dx = "
        f"{args.physics_dx}"
    )
    print(
        f"  physics_dy = "
        f"{args.physics_dy}"
    )
    print(
        f"  dt = {args.dt}"
    )
    print(
        "  note: physics grid/dt define the FD proxy only; "
        "they do not alter frozen M10 checkpoints."
    )

    print()
    print(
        "========== PROVENANCE =========="
    )
    print(
        "DATA_SHA256:",
        sha256_file(
            data_path
        ),
    )
    print(
        "SPLIT_SHA256:",
        sha256_file(
            split_path
        ),
    )
    print(
        "STATS_SHA256:",
        sha256_file(
            stats_path
        ),
    )
    print(
        "M6_SHA256:",
        sha256_file(
            m6_path
        ),
    )
    print(
        "M10_STATEPARAM_SHA256:",
        sha256_file(
            m10_path
        ),
    )
    print(
        "M10_1A_SHA256:",
        sha256_file(
            m10_1a_path
        ),
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    dataset = PhysicsAuditDataset(
        split_path=split_path,
        split_key=args.split_key,
        max_horizon=max_horizon,
        stride=args.stride,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    normalizer = FieldWiseNormalizer(
        stats_path
    ).to(device)

    (
        field_mean,
        field_std,
    ) = base.build_field_stats(
        stats_path
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # separate namespace for MODEL metadata.
    #
    # Existing checkpoints must see historical dx/dy.
    # --------------------------------------------------------

    model_args = SimpleNamespace(
        seed=args.seed,
        dx=args.model_dx,
        dy=args.model_dy,
        alpha_max=args.alpha_max,
        conditioner_hidden=(
            args.conditioner_hidden
        ),
        path_b_rms_cap=(
            args.path_b_rms_cap
        ),
        path_b_rms_eps=(
            args.path_b_rms_eps
        ),
    )

    # --------------------------------------------------------
    # M6
    # --------------------------------------------------------

    m6, _ = base.build_m6(
        m6_path,
        device,
    )

    # --------------------------------------------------------
    # Original M10-StateParam
    # --------------------------------------------------------

    (
        m10_stateparam,
        m10_payload,
    ) = base.build_m10(
        mode="stateparam",
        checkpoint_path=m10_path,
        field_mean=field_mean,
        field_std=field_std,
        args=model_args,
        device=device,
    )

    base.assert_same_m6(
        m6,
        m10_stateparam,
        "M10-StateParam",
    )

    # --------------------------------------------------------
    # M10-1a StateParam RMSCap
    # --------------------------------------------------------

    (
        m10_1a,
        m10_1a_payload,
    ) = (
        rmscap_eval.build_rmscap_stateparam(
            checkpoint_path=m10_1a_path,
            field_mean=field_mean,
            field_std=field_std,
            args=model_args,
            device=device,
        )
    )

    base.assert_same_m6(
        m6,
        m10_1a,
        "M10-1a-StateParam-RMSCap",
    )

    if isinstance(
        m10_payload,
        dict,
    ):
        print(
            "📌 M10-StateParam best_val:",
            m10_payload.get(
                "best_val_loss",
                m10_payload.get(
                    "val_loss",
                    "NA",
                ),
            ),
        )

    if isinstance(
        m10_1a_payload,
        dict,
    ):
        print(
            "📌 M10-1a best_val:",
            m10_1a_payload.get(
                "best_val_loss",
                m10_1a_payload.get(
                    "val_loss",
                    "NA",
                ),
            ),
        )

    models = {
        "M6": m6,
        "M10-StateParam": (
            m10_stateparam
        ),
        "M10-1a-StateParam-RMSCap": (
            m10_1a
        ),
    }

    model_order = [
        "M6",
        "M10-StateParam",
        "M10-1a-StateParam-RMSCap",
    ]

    stats = {}

    # --------------------------------------------------------
    # Rollout
    # --------------------------------------------------------

    print()
    print(
        "🔥 Starting Physics-Audit "
        "free-autoregressive rollout..."
    )

    with torch.no_grad():

        for (
            batch_idx,
            (
                x0_phys,
                future_phys,
                param,
            ),
        ) in enumerate(
            loader,
            start=1,
        ):

            x0_phys = x0_phys.to(
                device,
                non_blocking=True,
            )

            future_phys = future_phys.to(
                device,
                non_blocking=True,
            )

            param = param.to(
                device,
                non_blocking=True,
            )

            for model_name in model_order:

                run_model_rollout(
                    model_name=model_name,
                    model=models[
                        model_name
                    ],
                    x0_phys=x0_phys,
                    future_phys=(
                        future_phys
                    ),
                    param=param,
                    normalizer=normalizer,
                    max_horizon=(
                        max_horizon
                    ),
                    physics_dx=(
                        args.physics_dx
                    ),
                    physics_dy=(
                        args.physics_dy
                    ),
                    dt=args.dt,
                    stats=stats,
                )

            if (
                batch_idx % 10 == 0
                or batch_idx
                == len(loader)
            ):
                print(
                    f"  processed batch "
                    f"{batch_idx}/"
                    f"{len(loader)}"
                )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    rows = []

    for model_name in model_order:
        for horizon in range(
            1,
            max_horizon + 1,
        ):

            rows.append(
                finalize_bucket(
                    split_label=(
                        args.split_label
                    ),
                    seed=args.seed,
                    model_name=(
                        model_name
                    ),
                    horizon=horizon,
                    bucket=stats[
                        (
                            model_name,
                            horizon,
                        )
                    ],
                )
            )

    curve_df = pd.DataFrame(
        rows
    )

    summary_df = curve_df[
        curve_df["horizon"].isin(
            requested_horizons
        )
    ].copy()

    diff_df = make_difference_table(
        summary_df
    )

    output_dir = resolve_path(
        args.output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    prefix = (
        f"m10_physics_audit_v2_"
        f"{args.split_label}_"
        f"seed{args.seed}"
    )

    curve_path = os.path.join(
        output_dir,
        f"{prefix}_curve_h1_h"
        f"{max_horizon}.csv",
    )

    summary_path = os.path.join(
        output_dir,
        f"{prefix}_summary.csv",
    )

    diff_path = os.path.join(
        output_dir,
        f"{prefix}_differences.csv",
    )

    metadata_path = os.path.join(
        output_dir,
        f"{prefix}_metadata.json",
    )

    audit_metadata = {
        "audit_version": "M10-Physics-Audit-v2",
        "pde_semantics": (
            "FD-based comparative proxy; not solver-level PDE violation"
        ),
        "primary_pde_metric": (
            "dimensionless residual mismatch normalized by "
            "same-unit GT PDE-term scale"
        ),
        "model_dx": args.model_dx,
        "model_dy": args.model_dy,
        "physics_dx": args.physics_dx,
        "physics_dy": args.physics_dy,
        "dt": args.dt,
        "data_sha256": sha256_file(data_path),
        "split_sha256": sha256_file(split_path),
        "stats_sha256": sha256_file(stats_path),
        "m6_sha256": sha256_file(m6_path),
        "m10_stateparam_sha256": sha256_file(m10_path),
        "m10_1a_sha256": sha256_file(m10_1a_path),
        "native_gt_calibration_note": (
            "Official released GT does not yield a near-zero vertical-"
            "momentum residual under ordinary second-order FD; therefore "
            "absolute FD residual is calibration only."
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit_metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    curve_df.to_csv(
        curve_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    diff_df.to_csv(
        diff_path,
        index=False,
    )

    # --------------------------------------------------------
    # Compact terminal table
    # --------------------------------------------------------

    compact_cols = [
        "model",
        "horizon",
        "global_rel_l2_percent",
        "u_y_rel_l2_percent",
        "adv_b_rel_l2_percent",
        "div_error_mae",
        "vorticity_rel_l2_percent",
        "grad_p_rel_l2_percent",
        "r_b_normalized_mismatch",
        "r_uy_normalized_mismatch",
        "r_u_normalized_mismatch",
    ]

    print()
    print(
        "================ "
        "M10 PHYSICS AUDIT v2 SUMMARY "
        "================"
    )

    print(
        summary_df[
            compact_cols
        ].to_string(
            index=False
        )
    )

    # GT calibration is model-independent for a fixed sample/horizon.
    # Print one representative copy (M6 bucket) to make the non-zero
    # FD residual baseline visible without duplicating three identical rows.
    gt_calibration_cols = [
        "horizon",
        "r_b_gt_rms",
        "r_uy_gt_rms",
        "r_u_gt_rms",
        "r_b_gt_residual_to_term_scale",
        "r_uy_gt_residual_to_term_scale",
        "r_u_gt_residual_to_term_scale",
    ]

    gt_calibration_df = summary_df[
        summary_df["model"] == "M6"
    ][
        gt_calibration_cols
    ].copy()

    print()
    print(
        "========== GT FD-PROXY CALIBRATION =========="
    )
    print(
        "Non-zero values here are NOT model errors; "
        "they document the GT behavior of the chosen FD proxy."
    )
    print(
        gt_calibration_df.to_string(
            index=False
        )
    )

    print()
    print(
        "========== KEY DIFFERENCES =========="
    )
    print(
        "Negative = model A better "
        "for all reported difference metrics."
    )

    key_diff_cols = [
        "comparison",
        "horizon",
        "global_rel_l2_percent_diff",
        "u_y_rel_l2_percent_diff",
        "adv_b_rel_l2_percent_diff",
        "div_error_mae_diff",
        "vorticity_rel_l2_percent_diff",
        "grad_p_rel_l2_percent_diff",
        "r_b_error_rms_diff",
        "r_uy_error_rms_diff",
        "r_b_normalized_mismatch_diff",
        "r_uy_normalized_mismatch_diff",
        "r_u_normalized_mismatch_diff",
    ]

    print(
        diff_df[
            key_diff_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "✅ Saved:"
    )
    print(
        "  curve:",
        curve_path,
    )
    print(
        "  summary:",
        summary_path,
    )
    print(
        "  differences:",
        diff_path,
    )
    print(
        "  metadata:",
        metadata_path,
    )


if __name__ == "__main__":
    main()
