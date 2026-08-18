import os
import sys
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Project imports
# ============================================================

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.rbc_dataset import RBCDataset
from constants import (
    FIELD_ORDER,
    B_IDX,
    UX_IDX,
    UY_IDX,
)


# ============================================================
# D1-0 locked dimensional conventions
#
# IMPORTANT:
# These are NOT tuning parameters.
#
# OLD:
#   exact M10 screening convention.
#
# GRID:
#   corrected canonical spatial grid.
#
# CANONICAL:
#   corrected spatial grid + finite-frame dt conversion.
# ============================================================

DX_OLD = 1.0 / 64.0
DY_OLD = 1.0 / 64.0

DX_CANONICAL = 1.0 / 64.0
DY_CANONICAL = 1.0 / 63.0

DT_CANONICAL = 0.25

CONTEXT_LENGTH = 4
NUM_FIELDS = 4

VARIANT_ORDER = [
    "old",
    "grid",
    "canonical",
]


# ============================================================
# Dataset wrapper
# ============================================================

class D1OneStepDataset(Dataset):
    """
    D1-0 GT-state dimensional-interface dataset.

    Expected output
    ---------------
    context_norm:
        [4, 4, X, Y]

    y_seq_norm:
        [1, 4, X, Y]

    param:
        [2] = [log10(Ra), log10(Pr)]

    No model rollout is used.
    No prediction is used.
    """

    def __init__(
        self,
        base_dataset,
        context_length=4,
    ):
        self.base_dataset = base_dataset
        self.context_length = int(
            context_length
        )

    def __len__(self):
        return len(self.base_dataset)

    def _parse_sample(
        self,
        sample,
    ):
        x_norm = None
        y_seq_norm = None
        param = None

        if isinstance(
            sample,
            dict,
        ):
            for key in [
                "x_norm",
                "x",
                "context",
                "context_norm",
            ]:
                if key in sample:
                    x_norm = sample[key]
                    break

            for key in [
                "y_seq_norm",
                "y_seq",
                "target_sequence",
                "sequence",
            ]:
                if key in sample:
                    y_seq_norm = sample[key]
                    break

            for key in [
                "param",
                "params",
                "parameter",
                "parameters",
            ]:
                if key in sample:
                    param = sample[key]
                    break

        elif isinstance(
            sample,
            (tuple, list),
        ):
            for obj in sample:

                if torch.is_tensor(obj):

                    if (
                        obj.ndim == 3
                        and obj.shape[0]
                        == self.context_length
                        * NUM_FIELDS
                    ):
                        x_norm = obj

                    elif (
                        obj.ndim == 4
                        and obj.shape[1]
                        == NUM_FIELDS
                    ):
                        y_seq_norm = obj

                    elif (
                        obj.ndim == 1
                        and obj.numel() == 2
                    ):
                        param = obj

                elif (
                    isinstance(
                        obj,
                        (tuple, list),
                    )
                    and len(obj) == 2
                ):
                    try:
                        maybe_param = (
                            torch.tensor(
                                obj,
                                dtype=torch.float32,
                            )
                        )

                        if (
                            maybe_param.ndim == 1
                            and maybe_param.numel()
                            == 2
                        ):
                            param = maybe_param

                    except Exception:
                        pass

        else:
            raise TypeError(
                "Unsupported RBCDataset "
                f"sample type: {type(sample)}"
            )

        if (
            x_norm is None
            or y_seq_norm is None
            or param is None
        ):
            raise RuntimeError(
                "Could not parse "
                "x_norm / y_seq_norm / param "
                "from RBCDataset output.\n"
                "D1 requires "
                "return_sequence=True and "
                "return_params=True.\n"
                f"sample type: {type(sample)}\n"
                f"sample repr: "
                f"{repr(sample)[:500]}"
            )

        if not torch.is_tensor(param):
            param = torch.tensor(
                param,
                dtype=torch.float32,
            )

        return (
            x_norm,
            y_seq_norm,
            param.float(),
        )

    def __getitem__(
        self,
        idx,
    ):
        sample = self.base_dataset[idx]

        (
            x_norm,
            y_seq_norm,
            param,
        ) = self._parse_sample(
            sample
        )

        if x_norm.ndim != 3:
            raise RuntimeError(
                "x_norm must be "
                "[16, X, Y], got "
                f"{tuple(x_norm.shape)}"
            )

        c_total, nx, ny = (
            x_norm.shape
        )

        expected_channels = (
            self.context_length
            * NUM_FIELDS
        )

        if (
            c_total
            != expected_channels
        ):
            raise RuntimeError(
                "Unexpected context "
                "channel count: "
                f"expected "
                f"{expected_channels}, "
                f"got {c_total}"
            )

        context_norm = (
            x_norm.view(
                self.context_length,
                NUM_FIELDS,
                nx,
                ny,
            )
        )

        if (
            y_seq_norm.ndim != 4
            or y_seq_norm.shape[1]
            != NUM_FIELDS
            or y_seq_norm.shape[0] < 1
        ):
            raise RuntimeError(
                "y_seq_norm must be "
                "[S, 4, X, Y], got "
                f"{tuple(y_seq_norm.shape)}"
            )

        return (
            context_norm,
            y_seq_norm,
            param,
        )


# ============================================================
# Field statistics
# ============================================================

def load_field_stats(
    stats_path,
):
    with open(
        stats_path,
        "r",
        encoding="utf-8",
    ) as f:
        stats = json.load(f)

    mean = torch.tensor(
        [
            float(
                stats[field]["mean"]
            )
            for field
            in FIELD_ORDER
        ],
        dtype=torch.float32,
    )

    std = torch.tensor(
        [
            float(
                stats[field]["std"]
            )
            for field
            in FIELD_ORDER
        ],
        dtype=torch.float32,
    )

    if torch.any(std <= 0):
        raise RuntimeError(
            "All field std values "
            "must be positive."
        )

    return mean, std


# ============================================================
# Exact M10-style finite differences
# ============================================================

def grad_x_periodic(
    f,
    dx,
):
    """
    Second-order central difference
    in periodic X direction.

    f:
        [B, X, Y]
    """

    return (
        torch.roll(
            f,
            shifts=-1,
            dims=-2,
        )
        -
        torch.roll(
            f,
            shifts=1,
            dims=-2,
        )
    ) / (
        2.0 * dx
    )


def grad_y_nonperiodic(
    f,
    dy,
):
    """
    M10-style non-periodic Y derivative.

    Interior:
        second-order central difference.

    Boundaries:
        second-order one-sided difference.
    """

    if f.shape[-1] < 3:
        raise ValueError(
            "Need at least 3 Y points."
        )

    grad = torch.empty_like(f)

    # Interior
    grad[..., 1:-1] = (
        f[..., 2:]
        -
        f[..., :-2]
    ) / (
        2.0 * dy
    )

    # Lower boundary
    grad[..., 0] = (
        -3.0 * f[..., 0]
        + 4.0 * f[..., 1]
        - f[..., 2]
    ) / (
        2.0 * dy
    )

    # Upper boundary
    grad[..., -1] = (
        3.0 * f[..., -1]
        - 4.0 * f[..., -2]
        + f[..., -3]
    ) / (
        2.0 * dy
    )

    return grad


# ============================================================
# Path-B construction
# ============================================================

def compute_path_b_norm(
    current_norm,
    field_mean,
    field_std,
    dx,
    dy,
    dt_factor,
):
    """
    Reproduce the M10 representation chain:

        current_norm
          -> inverse Z-score-like reconstruction
          -> native/HDF5 field representation
          -> -u dot grad(b)
          -> divide by buoyancy std
          -> optional finite-step dt factor

    IMPORTANT:
    For fidelity with existing M10,
    reconstruction uses raw field_std,
    matching the M10 model buffer convention.
    """

    if current_norm.ndim != 4:
        raise ValueError(
            "current_norm must be "
            "[B,4,X,Y]."
        )

    mean = field_mean.view(
        1,
        NUM_FIELDS,
        1,
        1,
    )

    std = field_std.view(
        1,
        NUM_FIELDS,
        1,
        1,
    )

    current_native = (
        current_norm * std
        + mean
    )

    b = current_native[
        :,
        B_IDX,
        :,
        :,
    ]

    ux = current_native[
        :,
        UX_IDX,
        :,
        :,
    ]

    uy = current_native[
        :,
        UY_IDX,
        :,
        :,
    ]

    db_dx = grad_x_periodic(
        b,
        dx,
    )

    db_dy = grad_y_nonperiodic(
        b,
        dy,
    )

    minus_u_dot_grad_b = -(
        ux * db_dx
        +
        uy * db_dy
    )

    b_std = field_std[B_IDX]

    path_b_norm = (
        minus_u_dot_grad_b
        / b_std
    )

    path_b_norm = (
        float(dt_factor)
        * path_b_norm
    )

    return path_b_norm


# ============================================================
# Streaming metric accumulator
# ============================================================

class PairAccumulator:
    """
    Streaming statistics for:

        x = Path-B candidate
        y = GT normalized buoyancy delta
    """

    def __init__(self):
        self.samples = 0
        self.points = 0

        self.sum_x = 0.0
        self.sum_y = 0.0

        self.sum_x2 = 0.0
        self.sum_y2 = 0.0

        self.sum_xy = 0.0
        self.sum_diff2 = 0.0

    def update(
        self,
        x,
        y,
    ):
        if x.shape != y.shape:
            raise RuntimeError(
                "Metric shapes do not match: "
                f"x={tuple(x.shape)}, "
                f"y={tuple(y.shape)}"
            )

        x64 = (
            x.detach()
            .double()
        )

        y64 = (
            y.detach()
            .double()
        )

        self.samples += int(
            x.shape[0]
        )

        self.points += int(
            x64.numel()
        )

        self.sum_x += float(
            x64.sum().item()
        )

        self.sum_y += float(
            y64.sum().item()
        )

        self.sum_x2 += float(
            torch.sum(
                x64 * x64
            ).item()
        )

        self.sum_y2 += float(
            torch.sum(
                y64 * y64
            ).item()
        )

        self.sum_xy += float(
            torch.sum(
                x64 * y64
            ).item()
        )

        diff = x64 - y64

        self.sum_diff2 += float(
            torch.sum(
                diff * diff
            ).item()
        )

    def finalize(self):
        if self.points <= 0:
            raise RuntimeError(
                "Empty metric bucket."
            )

        n = float(
            self.points
        )

        gt_rms = math.sqrt(
            max(
                self.sum_y2 / n,
                0.0,
            )
        )

        rb_rms = math.sqrt(
            max(
                self.sum_x2 / n,
                0.0,
            )
        )

        if gt_rms > 0:
            scale_ratio = (
                rb_rms / gt_rms
            )
        else:
            scale_ratio = float("nan")

        cosine_denom = math.sqrt(
            max(
                self.sum_x2
                * self.sum_y2,
                0.0,
            )
        )

        if cosine_denom > 0:
            cosine = (
                self.sum_xy
                / cosine_denom
            )
        else:
            cosine = float("nan")

        centered_x2 = (
            self.sum_x2
            -
            self.sum_x
            * self.sum_x
            / n
        )

        centered_y2 = (
            self.sum_y2
            -
            self.sum_y
            * self.sum_y
            / n
        )

        centered_xy = (
            self.sum_xy
            -
            self.sum_x
            * self.sum_y
            / n
        )

        pearson_denom = math.sqrt(
            max(
                centered_x2
                * centered_y2,
                0.0,
            )
        )

        if pearson_denom > 0:
            pearson = (
                centered_xy
                / pearson_denom
            )
        else:
            pearson = float("nan")

        if self.sum_y2 > 0:
            rel_l2_diag = math.sqrt(
                self.sum_diff2
                / self.sum_y2
            )
        else:
            rel_l2_diag = float("nan")

        return {
            "samples": self.samples,
            "points": self.points,
            "gt_delta_rms": gt_rms,
            "rb_rms": rb_rms,
            "scale_ratio": scale_ratio,
            "cosine": cosine,
            "pearson": pearson,
            "rel_l2_diag": rel_l2_diag,
        }


# ============================================================
# Helpers
# ============================================================

def param_key(
    param_row,
):
    log_ra = round(
        float(param_row[0]),
        6,
    )

    log_pr = round(
        float(param_row[1]),
        6,
    )

    return (
        log_ra,
        log_pr,
    )


def physical_param_values(
    log_ra,
    log_pr,
):
    ra = 10.0 ** float(log_ra)
    pr = 10.0 ** float(log_pr)

    return (
        ra,
        pr,
    )


def variant_metadata(
    variant,
):
    if variant == "old":
        return {
            "dx": DX_OLD,
            "dy": DY_OLD,
            "dt_factor": 1.0,
            "description": (
                "M10 old convention: "
                "dx=1/64, dy=1/64, "
                "no explicit dt"
            ),
        }

    if variant == "grid":
        return {
            "dx": DX_CANONICAL,
            "dy": DY_CANONICAL,
            "dt_factor": 1.0,
            "description": (
                "grid corrected only: "
                "dx=1/64, dy=1/63, "
                "no explicit dt"
            ),
        }

    if variant == "canonical":
        return {
            "dx": DX_CANONICAL,
            "dy": DY_CANONICAL,
            "dt_factor": DT_CANONICAL,
            "description": (
                "canonical finite-step: "
                "dx=1/64, dy=1/63, "
                "dt=0.25"
            ),
        }

    raise ValueError(
        f"Unknown variant: {variant}"
    )


# ============================================================
# One subset
# ============================================================

@torch.no_grad()
def audit_subset(
    subset_name,
    dataset,
    field_mean,
    field_std,
    device,
    batch_size,
    num_workers,
    max_batches,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    print()
    print(
        "========================================"
    )
    print(
        f"D1-0 SUBSET: {subset_name.upper()}"
    )
    print(
        "========================================"
    )
    print(
        "Samples:",
        len(dataset),
    )

    sample = dataset[0]

    (
        sample_context,
        sample_y_seq,
        sample_param,
    ) = sample

    print(
        "Context shape:",
        tuple(sample_context.shape),
    )
    print(
        "Y-seq shape:",
        tuple(sample_y_seq.shape),
    )
    print(
        "Param example:",
        sample_param.tolist(),
    )

    # key:
    # (
    #   scope,
    #   log_ra,
    #   log_pr,
    #   variant
    # )
    accumulators = defaultdict(
        PairAccumulator
    )

    mean_dev = field_mean.to(
        device
    )

    std_dev = field_std.to(
        device
    )

    processed_batches = 0

    for batch_index, (
        context_norm,
        y_seq_norm,
        param,
    ) in enumerate(
        loader,
        start=1,
    ):
        if (
            max_batches is not None
            and batch_index
            > max_batches
        ):
            break

        context_norm = (
            context_norm.to(
                device,
                non_blocking=False,
            )
        )

        y_seq_norm = (
            y_seq_norm.to(
                device,
                non_blocking=False,
            )
        )

        param_cpu = (
            param.detach()
            .cpu()
        )

        current_norm = (
            context_norm[:, -1]
        )

        next_norm = (
            y_seq_norm[:, 0]
        )

        # -----------------------------------------
        # Ground-truth finite-frame buoyancy delta
        # already in network-normalized space.
        # -----------------------------------------

        gt_delta_b_norm = (
            next_norm[
                :,
                B_IDX,
                :,
                :,
            ]
            -
            current_norm[
                :,
                B_IDX,
                :,
                :,
            ]
        )

        # -----------------------------------------
        # Three D1 variants
        # -----------------------------------------

        rb_old = compute_path_b_norm(
            current_norm=current_norm,
            field_mean=mean_dev,
            field_std=std_dev,
            dx=DX_OLD,
            dy=DY_OLD,
            dt_factor=1.0,
        )

        rb_grid = compute_path_b_norm(
            current_norm=current_norm,
            field_mean=mean_dev,
            field_std=std_dev,
            dx=DX_CANONICAL,
            dy=DY_CANONICAL,
            dt_factor=1.0,
        )

        rb_canonical = (
            DT_CANONICAL
            * rb_grid
        )

        rb_map = {
            "old": rb_old,
            "grid": rb_grid,
            "canonical": rb_canonical,
        }

        # -----------------------------------------
        # Overall subset buckets
        # -----------------------------------------

        for variant in VARIANT_ORDER:
            accumulators[
                (
                    "overall",
                    None,
                    None,
                    variant,
                )
            ].update(
                rb_map[variant],
                gt_delta_b_norm,
            )

        # -----------------------------------------
        # Condition buckets
        # -----------------------------------------

        condition_to_indices = (
            defaultdict(list)
        )

        for sample_index in range(
            param_cpu.shape[0]
        ):
            key = param_key(
                param_cpu[
                    sample_index
                ]
            )

            condition_to_indices[
                key
            ].append(
                sample_index
            )

        for (
            log_ra,
            log_pr,
        ), indices in (
            condition_to_indices.items()
        ):
            idx = torch.tensor(
                indices,
                dtype=torch.long,
                device=device,
            )

            gt_condition = (
                gt_delta_b_norm.index_select(
                    0,
                    idx,
                )
            )

            for variant in VARIANT_ORDER:
                rb_condition = (
                    rb_map[
                        variant
                    ].index_select(
                        0,
                        idx,
                    )
                )

                accumulators[
                    (
                        "condition",
                        log_ra,
                        log_pr,
                        variant,
                    )
                ].update(
                    rb_condition,
                    gt_condition,
                )

        processed_batches += 1

        if (
            batch_index == 1
            or batch_index % 50 == 0
        ):
            print(
                f"Processed batches: "
                f"{batch_index}"
            )

    if processed_batches == 0:
        raise RuntimeError(
            "No batches were processed."
        )

    print(
        "Processed batches total:",
        processed_batches,
    )

    rows = []

    for (
        scope,
        log_ra,
        log_pr,
        variant,
    ), accumulator in sorted(
        accumulators.items(),
        key=lambda item: (
            item[0][0],
            -999.0
            if item[0][1] is None
            else item[0][1],
            -999.0
            if item[0][2] is None
            else item[0][2],
            VARIANT_ORDER.index(
                item[0][3]
            ),
        ),
    ):
        metrics = (
            accumulator.finalize()
        )

        meta = variant_metadata(
            variant
        )

        if (
            log_ra is None
            or log_pr is None
        ):
            ra = float("nan")
            pr = float("nan")
        else:
            ra, pr = (
                physical_param_values(
                    log_ra,
                    log_pr,
                )
            )

        row = {
            "subset": subset_name,
            "scope": scope,
            "log10_ra": log_ra,
            "log10_pr": log_pr,
            "ra": ra,
            "pr": pr,
            "variant": variant,
            "dx": meta["dx"],
            "dy": meta["dy"],
            "dt_factor": (
                meta["dt_factor"]
            ),
            "description": (
                meta["description"]
            ),
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

    return rows


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "D1-0 GT-state dimensional-interface "
            "diagnostic for M10 Path B. "
            "No training, no model checkpoint, "
            "no gate, no RMSCap."
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help=(
            "Split JSON, e.g. "
            "data/splits/unseen_pr_split.json"
        ),
    )

    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help=(
            "Matching field-wise stats JSON."
        ),
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="both",
        choices=[
            "train",
            "val",
            "both",
        ],
        help=(
            "D1 intentionally excludes TEST."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=[
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help=(
            "Smoke-test only. "
            "Omit for full D1 audit."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    print(
        "========================================"
    )
    print(
        "D1-0 DIMENSIONAL-INTERFACE DIAGNOSTIC"
    )
    print(
        "========================================"
    )
    print(
        "Stage: CONTROL / DIAGNOSTIC"
    )
    print(
        "Training: OFF"
    )
    print(
        "M6/M10 model: NOT LOADED"
    )
    print(
        "Utility Gate: OFF"
    )
    print(
        "alpha_B: OFF"
    )
    print(
        "RMSCap: OFF"
    )
    print(
        "TEST access: FORBIDDEN"
    )

    print()
    print(
        "========== LOCKED CONVENTIONS =========="
    )
    print(
        "OLD:"
    )
    print(
        "  dx =",
        DX_OLD,
    )
    print(
        "  dy =",
        DY_OLD,
    )
    print(
        "  dt factor = 1.0"
    )

    print(
        "GRID:"
    )
    print(
        "  dx =",
        DX_CANONICAL,
    )
    print(
        "  dy =",
        DY_CANONICAL,
    )
    print(
        "  dt factor = 1.0"
    )

    print(
        "CANONICAL:"
    )
    print(
        "  dx =",
        DX_CANONICAL,
    )
    print(
        "  dy =",
        DY_CANONICAL,
    )
    print(
        "  dt factor =",
        DT_CANONICAL,
    )

    if not os.path.exists(
        args.split
    ):
        raise FileNotFoundError(
            args.split
        )

    if not os.path.exists(
        args.stats
    ):
        raise FileNotFoundError(
            args.stats
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if (
        args.max_batches is not None
        and args.max_batches <= 0
    ):
        raise ValueError(
            "max_batches must be positive."
        )

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    device = torch.device(
        args.device
    )

    print()
    print(
        "Device:",
        device,
    )
    print(
        "Split:",
        args.split,
    )
    print(
        "Stats:",
        args.stats,
    )
    print(
        "Subset:",
        args.subset,
    )
    print(
        "Max batches:",
        args.max_batches,
    )

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as f:
        split_config = json.load(f)

    field_mean, field_std = (
        load_field_stats(
            args.stats
        )
    )

    print()
    print(
        "========== FIELD STATS =========="
    )

    for i, field in enumerate(
        FIELD_ORDER
    ):
        print(
            f"{field:10s}",
            "mean=",
            f"{field_mean[i].item():.8e}",
            "std=",
            f"{field_std[i].item():.8e}",
        )

    if args.subset == "both":
        subset_names = [
            "train",
            "val",
        ]
    else:
        subset_names = [
            args.subset
        ]

    all_rows = []

    for subset_name in subset_names:

        if subset_name not in split_config:
            raise KeyError(
                f"Split JSON has no "
                f"'{subset_name}' section."
            )

        base_dataset = RBCDataset(
            split_config=(
                split_config[
                    subset_name
                ]
            ),
            normalize=True,
            stats_path=args.stats,
            return_sequence=True,
            target_steps=1,
            return_params=True,
        )

        dataset = D1OneStepDataset(
            base_dataset,
            context_length=(
                CONTEXT_LENGTH
            ),
        )

        rows = audit_subset(
            subset_name=subset_name,
            dataset=dataset,
            field_mean=field_mean,
            field_std=field_std,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=(
                args.max_batches
            ),
        )

        all_rows.extend(
            rows
        )

    df = pd.DataFrame(
        all_rows
    )

    # -----------------------------------------
    # Output path
    # -----------------------------------------

    if args.output is None:
        split_stem = Path(
            args.split
        ).stem

        output = (
            "outputs/tables/"
            f"d1_dimensional_interface_"
            f"{split_stem}.csv"
        )
    else:
        output = args.output

    output_dir = os.path.dirname(
        output
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    df.to_csv(
        output,
        index=False,
    )

    # ========================================================
    # Compact terminal summary
    # ========================================================

    print()
    print(
        "========================================"
    )
    print(
        "D1-0 OVERALL SUMMARY"
    )
    print(
        "========================================"
    )

    overall = (
        df[
            df["scope"]
            == "overall"
        ][
            [
                "subset",
                "variant",
                "samples",
                "gt_delta_rms",
                "rb_rms",
                "scale_ratio",
                "cosine",
                "pearson",
                "rel_l2_diag",
            ]
        ]
        .copy()
    )

    overall["variant"] = (
        pd.Categorical(
            overall["variant"],
            categories=(
                VARIANT_ORDER
            ),
            ordered=True,
        )
    )

    overall = overall.sort_values(
        [
            "subset",
            "variant",
        ]
    )

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6e}"
            ),
        )
    )

    print()
    print(
        "========================================"
    )
    print(
        "PER-CONDITION SCALE RATIO"
    )
    print(
        "RMS(PathB) / RMS(GT delta_b_norm)"
    )
    print(
        "========================================"
    )

    condition = df[
        df["scope"]
        == "condition"
    ].copy()

    if not condition.empty:

        pivot = (
            condition.pivot_table(
                index=[
                    "subset",
                    "ra",
                    "pr",
                ],
                columns="variant",
                values="scale_ratio",
                aggfunc="first",
            )
            .reset_index()
        )

        desired_cols = [
            "subset",
            "ra",
            "pr",
        ] + [
            v
            for v in VARIANT_ORDER
            if v in pivot.columns
        ]

        pivot = pivot[
            desired_cols
        ]

        print(
            pivot.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.6e}"
                ),
            )
        )

    # ========================================================
    # Deterministic sanity checks
    # ========================================================

    print()
    print(
        "========================================"
    )
    print(
        "DETERMINISTIC SANITY CHECK"
    )
    print(
        "========================================"
    )

    for subset_name in subset_names:
        sub = overall[
            overall["subset"]
            == subset_name
        ]

        grid_row = sub[
            sub["variant"]
            == "grid"
        ]

        canonical_row = sub[
            sub["variant"]
            == "canonical"
        ]

        if (
            len(grid_row) == 1
            and len(canonical_row) == 1
        ):
            grid_rms = float(
                grid_row[
                    "rb_rms"
                ].iloc[0]
            )

            canonical_rms = float(
                canonical_row[
                    "rb_rms"
                ].iloc[0]
            )

            grid_cos = float(
                grid_row[
                    "cosine"
                ].iloc[0]
            )

            canonical_cos = float(
                canonical_row[
                    "cosine"
                ].iloc[0]
            )

            ratio = (
                canonical_rms
                / grid_rms
                if grid_rms > 0
                else float("nan")
            )

            cos_diff = (
                canonical_cos
                - grid_cos
            )

            print(
                f"{subset_name}: "
                "canonical/grid Path-B RMS "
                f"= {ratio:.8f} "
                f"(expected {DT_CANONICAL:.8f})"
            )

            print(
                f"{subset_name}: "
                "canonical-grid cosine diff "
                f"= {cos_diff:.8e} "
                "(expected ~0)"
            )

    print()
    print(
        "NOTE:"
    )
    print(
        "rel_l2_diag is diagnostic only."
    )
    print(
        "Path B is one PDE contribution, "
        "not the complete buoyancy update."
    )

    print()
    print(
        "CSV saved to:"
    )
    print(
        output
    )

    print()
    print(
        "✅ D1-0 diagnostic finished."
    )


if __name__ == "__main__":
    main()
