import os
import sys
import re
import json
import math
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from constants import (
    DATA_PATH,
    FIELD_ORDER,
    B_IDX,
    UX_IDX,
    UY_IDX,
    EPS,
)

# Reuse the already-audited D1-1A checkpoint loader.
from evaluation.audit_m10_d1_1_alpha_gtstate import (
    build_model_from_checkpoint,
    load_stats,
    parse_ra_pr,
    evenly_spaced_indices,
    temporal_phase,
)

# Reuse the exact D1-0 dimensional-interface definitions.
from evaluation.audit_m10_dimensional_interface_d1 import (
    compute_path_b_norm,
    grad_x_periodic,
    grad_y_nonperiodic,
    DX_CANONICAL,
    DY_CANONICAL,
    DT_CANONICAL,
)


# ============================================================
# Locked D1-1B protocol
# ============================================================

CONTEXT_LENGTH = 4
MAX_HORIZON = 16

REPORT_HORIZONS = tuple(
    range(
        1,
        MAX_HORIZON + 1,
    )
)

NUM_FIELDS = 4


# ============================================================
# Dataset
# ============================================================

class H16StratifiedGTSequenceDataset(Dataset):
    """
    Deterministic stratified GT sequences.

    Each sample contains:

        context:
            t0 ... t0+3

        future:
            t0+4 ... t0+19

    Therefore 16-step rollout is fully supported.

    D1-1B then creates:

        1. GT branch:
               append true next state

        2. rollout branch:
               append model-predicted next state

    Both start from the exact same normalized GT context.
    """

    def __init__(
        self,
        *,
        split_items,
        data_path,
        field_mean,
        field_std,
        time_samples_per_trajectory,
    ):
        self.data_path = str(
            data_path
        )

        self.field_mean = np.asarray(
            field_mean,
            dtype=np.float32,
        )

        self.field_std = np.asarray(
            field_std,
            dtype=np.float32,
        )

        self.samples = []

        self._h5 = None

        with h5py.File(
            self.data_path,
            "r",
        ) as f:

            for item in split_items:

                group_name = item[
                    "group"
                ]

                trajectories = item[
                    "trajectories"
                ]

                if group_name not in f:
                    raise KeyError(
                        f"Missing group: "
                        f"{group_name}"
                    )

                group = f[
                    group_name
                ]

                total_t = int(
                    group[
                        FIELD_ORDER[0]
                    ].shape[1]
                )

                # context:
                #   t0 ... t0+3
                #
                # H16 future:
                #   t0+4 ... t0+19
                #
                # Last index <= T-1.
                max_t0 = (
                    total_t
                    - CONTEXT_LENGTH
                    - MAX_HORIZON
                )

                if max_t0 < 0:
                    continue

                selected_t0 = (
                    evenly_spaced_indices(
                        start=0,
                        end=max_t0,
                        count=(
                            time_samples_per_trajectory
                        ),
                    )
                )

                ra, pr = parse_ra_pr(
                    group_name
                )

                for trajectory in trajectories:

                    for t0 in selected_t0:

                        self.samples.append(
                            {
                                "group":
                                    group_name,

                                "trajectory":
                                    int(
                                        trajectory
                                    ),

                                "t0":
                                    int(t0),

                                "max_t0":
                                    int(max_t0),

                                "phase":
                                    temporal_phase(
                                        t0,
                                        max_t0,
                                    ),

                                "ra":
                                    float(ra),

                                "pr":
                                    float(pr),
                            }
                        )

        if not self.samples:
            raise RuntimeError(
                "No D1-1B samples."
            )

    def __len__(self):
        return len(
            self.samples
        )

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(
                self.data_path,
                "r",
            )

        return self._h5

    def __getitem__(
        self,
        idx,
    ):
        meta = self.samples[idx]

        f = self._get_h5()

        group = f[
            meta["group"]
        ]

        trajectory = meta[
            "trajectory"
        ]

        t0 = meta[
            "t0"
        ]

        full_fields = []

        for field in FIELD_ORDER:

            # 4 context + 16 future = 20 frames
            values = np.asarray(
                group[field][
                    trajectory,
                    t0:
                    t0
                    + CONTEXT_LENGTH
                    + MAX_HORIZON,
                ],
                dtype=np.float32,
            )

            if (
                values.shape[0]
                !=
                CONTEXT_LENGTH
                + MAX_HORIZON
            ):
                raise RuntimeError(
                    "Unexpected sequence length."
                )

            full_fields.append(
                values
            )

        # 4 × [T,X,Y]
        # -> [T,C,X,Y]
        full_raw = np.stack(
            full_fields,
            axis=1,
        )

        mean = (
            self.field_mean[
                None,
                :,
                None,
                None,
            ]
        )

        std = (
            self.field_std[
                None,
                :,
                None,
                None,
            ]
        )

        # Exact project normalization.
        full_norm = (
            full_raw
            - mean
        ) / (
            std
            + float(EPS)
        )

        context_norm = (
            full_norm[
                :CONTEXT_LENGTH
            ]
        )

        future_norm = (
            full_norm[
                CONTEXT_LENGTH:
            ]
        )

        param = np.asarray(
            [
                math.log10(
                    meta["ra"]
                ),
                math.log10(
                    meta["pr"]
                ),
            ],
            dtype=np.float32,
        )

        phase_id = {
            "early": 0,
            "middle": 1,
            "late": 2,
            "all": 3,
        }[
            meta["phase"]
        ]

        return (
            torch.from_numpy(
                context_norm
            ),
            torch.from_numpy(
                future_norm
            ),
            torch.from_numpy(
                param
            ),
            torch.tensor(
                meta["ra"],
                dtype=torch.float64,
            ),
            torch.tensor(
                meta["pr"],
                dtype=torch.float64,
            ),
            torch.tensor(
                trajectory,
                dtype=torch.long,
            ),
            torch.tensor(
                t0,
                dtype=torch.long,
            ),
            torch.tensor(
                phase_id,
                dtype=torch.long,
            ),
        )


# ============================================================
# Diagnostics
# ============================================================

def per_sample_rms(
    x,
):
    return torch.sqrt(
        torch.mean(
            x.double()
            * x.double(),
            dim=(-2, -1),
        )
        + 1.0e-30
    )


def per_sample_rel_l2(
    pred,
    target,
):
    pred64 = pred.double()
    target64 = target.double()

    diff2 = torch.sum(
        (
            pred64
            - target64
        ) ** 2,
        dim=tuple(
            range(
                1,
                pred64.ndim,
            )
        ),
    )

    target2 = torch.sum(
        target64 ** 2,
        dim=tuple(
            range(
                1,
                target64.ndim,
            )
        ),
    )

    return torch.sqrt(
        diff2
        / (
            target2
            + 1.0e-30
        )
    )


def state_decomposition(
    latest_norm,
    field_mean,
    field_std,
):
    """
    Diagnostics only.

    Reconstruct the same native/canonical HDF5 representation
    used by M10 before constructing PDE signals.

    Returns per-sample:
        velocity magnitude RMS
        buoyancy-gradient magnitude RMS
    """

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

    native = (
        latest_norm
        * std
        + mean
    )

    b = native[
        :,
        B_IDX,
    ]

    ux = native[
        :,
        UX_IDX,
    ]

    uy = native[
        :,
        UY_IDX,
    ]

    velocity_rms = torch.sqrt(
        torch.mean(
            ux.double() ** 2
            +
            uy.double() ** 2,
            dim=(-2, -1),
        )
        + 1.0e-30
    )

    db_dx = grad_x_periodic(
        b,
        DX_CANONICAL,
    )

    db_dy = grad_y_nonperiodic(
        b,
        DY_CANONICAL,
    )

    grad_b_rms = torch.sqrt(
        torch.mean(
            db_dx.double() ** 2
            +
            db_dy.double() ** 2,
            dim=(-2, -1),
        )
        + 1.0e-30
    )

    return (
        velocity_rms,
        grad_b_rms,
    )


@torch.no_grad()
def extract_state_metrics(
    *,
    comp,
    latest_norm,
    gt_latest_norm,
    field_mean,
    field_std,
):
    """
    Extract existing-M10 metrics plus exact D1 canonical Path B
    on the SAME current state.
    """

    alpha_b = (
        comp[
            "alpha_b"
        ]
        .detach()
        .double()
    )

    path_b_raw = (
        comp[
            "path_b_rms_raw"
        ]
        .detach()
        .double()
    )

    path_b_safe = (
        comp[
            "path_b_rms_safe"
        ]
        .detach()
        .double()
    )

    path_b_scale = (
        comp[
            "path_b_scale"
        ]
        .detach()
        .double()
    )

    cap_active = (
        comp[
            "path_b_cap_active"
        ]
        .detach()
    )

    # Existing M10 effective residual dose.
    old_injection_rms = (
        torch.abs(
            alpha_b
        )
        * path_b_safe
    )

    # --------------------------------------------------------
    # Exact D1 canonical Path B:
    #
    #   dx = 1/64
    #   dy = 1/63
    #   dt = 0.25
    #
    # This is DIAGNOSTIC ONLY.
    # It is NOT injected into the existing M10 model.
    # --------------------------------------------------------

    canonical_path_b = (
        compute_path_b_norm(
            current_norm=latest_norm,
            field_mean=field_mean,
            field_std=field_std,
            dx=DX_CANONICAL,
            dy=DY_CANONICAL,
            dt_factor=DT_CANONICAL,
        )
    )

    canonical_path_b_rms = (
        per_sample_rms(
            canonical_path_b
        )
    )

    canonical_to_old_ratio = (
        canonical_path_b_rms
        /
        (
            path_b_raw
            + 1.0e-30
        )
    )

    # --------------------------------------------------------
    # Current-state contamination
    # --------------------------------------------------------

    state_rel_l2 = (
        per_sample_rel_l2(
            latest_norm,
            gt_latest_norm,
        )
    )

    b_rel_l2 = (
        per_sample_rel_l2(
            latest_norm[
                :,
                B_IDX,
            ],
            gt_latest_norm[
                :,
                B_IDX,
            ],
        )
    )

    velocity_rel_l2 = (
        per_sample_rel_l2(
            latest_norm[
                :,
                [
                    UX_IDX,
                    UY_IDX,
                ],
            ],
            gt_latest_norm[
                :,
                [
                    UX_IDX,
                    UY_IDX,
                ],
            ],
        )
    )

    (
        velocity_mag_rms,
        grad_b_mag_rms,
    ) = state_decomposition(
        latest_norm,
        field_mean,
        field_std,
    )

    return {
        "alpha_b":
            alpha_b,

        "abs_alpha_b":
            torch.abs(
                alpha_b
            ),

        "alpha_negative":
            (
                alpha_b
                < -1.0e-12
            ),

        "alpha_positive":
            (
                alpha_b
                > 1.0e-12
            ),

        "path_b_rms_raw":
            path_b_raw,

        "path_b_rms_safe":
            path_b_safe,

        "path_b_scale":
            path_b_scale,

        "path_b_cap_active":
            cap_active,

        "old_injection_rms":
            old_injection_rms,

        "canonical_path_b_rms":
            canonical_path_b_rms,

        "canonical_to_old_raw_ratio":
            canonical_to_old_ratio,

        "state_rel_l2":
            state_rel_l2,

        "b_state_rel_l2":
            b_rel_l2,

        "velocity_state_rel_l2":
            velocity_rel_l2,

        "velocity_mag_rms":
            velocity_mag_rms,

        "grad_b_mag_rms":
            grad_b_mag_rms,
    }


def tensors_to_rows(
    *,
    metrics,
    split_label,
    seed,
    subset_name,
    state_type,
    horizon,
    ra,
    pr,
    trajectory,
    t0,
    phase_id,
):
    phase_name = {
        0: "early",
        1: "middle",
        2: "late",
        3: "all",
    }

    batch_size = int(
        ra.shape[0]
    )

    cpu_metrics = {}

    for key, value in metrics.items():

        if torch.is_tensor(
            value
        ):
            cpu_metrics[
                key
            ] = (
                value.detach()
                .cpu()
                .numpy()
            )

        else:
            cpu_metrics[
                key
            ] = value

    ra_np = (
        ra.detach()
        .cpu()
        .double()
        .numpy()
    )

    pr_np = (
        pr.detach()
        .cpu()
        .double()
        .numpy()
    )

    trajectory_np = (
        trajectory.detach()
        .cpu()
        .numpy()
    )

    t0_np = (
        t0.detach()
        .cpu()
        .numpy()
    )

    phase_np = (
        phase_id.detach()
        .cpu()
        .numpy()
    )

    rows = []

    for i in range(
        batch_size
    ):

        row = {
            "split_label":
                split_label,

            "seed":
                int(seed),

            "subset":
                subset_name,

            "state_type":
                state_type,

            "horizon":
                int(horizon),

            "ra":
                float(
                    ra_np[i]
                ),

            "pr":
                float(
                    pr_np[i]
                ),

            "trajectory":
                int(
                    trajectory_np[i]
                ),

            "t0":
                int(
                    t0_np[i]
                ),

            "phase":
                phase_name[
                    int(
                        phase_np[i]
                    )
                ],
        }

        for key in [
            "alpha_b",
            "abs_alpha_b",
            "path_b_rms_raw",
            "path_b_rms_safe",
            "path_b_scale",
            "old_injection_rms",
            "canonical_path_b_rms",
            "canonical_to_old_raw_ratio",
            "state_rel_l2",
            "b_state_rel_l2",
            "velocity_state_rel_l2",
            "velocity_mag_rms",
            "grad_b_mag_rms",
        ]:
            row[
                key
            ] = float(
                cpu_metrics[
                    key
                ][i]
            )

        for key in [
            "alpha_negative",
            "alpha_positive",
            "path_b_cap_active",
        ]:
            row[
                key
            ] = bool(
                cpu_metrics[
                    key
                ][i]
            )

        rows.append(
            row
        )

    return rows


# ============================================================
# One checkpoint / subset
# ============================================================

@torch.no_grad()
def audit_rollout(
    *,
    model,
    dataset,
    split_label,
    seed,
    subset_name,
    field_mean_np,
    field_std_np,
    device,
    batch_size,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    field_mean = torch.tensor(
        field_mean_np,
        dtype=torch.float32,
        device=device,
    )

    field_std = torch.tensor(
        field_std_np,
        dtype=torch.float32,
        device=device,
    )

    rows = []

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        (
            context_norm,
            future_norm,
            param,
            ra,
            pr,
            trajectory,
            t0,
            phase_id,
        ) = batch

        context_norm = (
            context_norm
            .to(
                device
            )
            .float()
        )

        future_norm = (
            future_norm
            .to(
                device
            )
            .float()
        )

        param = (
            param
            .to(
                device
            )
            .float()
        )

        if (
            future_norm.shape[1]
            != MAX_HORIZON
        ):
            raise RuntimeError(
                "Expected H16 future."
            )

        # ----------------------------------------------------
        # Two branches begin from the SAME GT context.
        # ----------------------------------------------------

        gt_context = (
            context_norm.clone()
        )

        rollout_context = (
            context_norm.clone()
        )

        batch_actual = (
            context_norm.shape[0]
        )

        nx = (
            context_norm.shape[-2]
        )

        ny = (
            context_norm.shape[-1]
        )

        for step in range(
            1,
            MAX_HORIZON + 1,
        ):
            # =================================================
            # Rollout branch
            # =================================================

            rollout_input = (
                rollout_context.reshape(
                    batch_actual,
                    CONTEXT_LENGTH
                    * NUM_FIELDS,
                    nx,
                    ny,
                )
            )

            (
                rollout_delta,
                rollout_comp,
            ) = model(
                rollout_input,
                params=param,
                return_components=True,
            )

            if not torch.isfinite(
                rollout_delta
            ).all():
                raise RuntimeError(
                    "Non-finite rollout delta: "
                    f"seed={seed}, "
                    f"subset={subset_name}, "
                    f"batch={batch_index}, "
                    f"h={step}"
                )

            gt_latest = (
                gt_context[
                    :,
                    -1,
                ]
            )

            rollout_latest = (
                rollout_context[
                    :,
                    -1,
                ]
            )

            # =================================================
            # Record only h = 1,4,8,16
            # =================================================

            if step in REPORT_HORIZONS:

                rollout_metrics = (
                    extract_state_metrics(
                        comp=rollout_comp,
                        latest_norm=(
                            rollout_latest
                        ),
                        gt_latest_norm=(
                            gt_latest
                        ),
                        field_mean=(
                            field_mean
                        ),
                        field_std=(
                            field_std
                        ),
                    )
                )

                rows.extend(
                    tensors_to_rows(
                        metrics=(
                            rollout_metrics
                        ),
                        split_label=(
                            split_label
                        ),
                        seed=seed,
                        subset_name=(
                            subset_name
                        ),
                        state_type=(
                            "rollout"
                        ),
                        horizon=step,
                        ra=ra,
                        pr=pr,
                        trajectory=(
                            trajectory
                        ),
                        t0=t0,
                        phase_id=(
                            phase_id
                        ),
                    )
                )

                # ---------------------------------------------
                # GT branch model components.
                #
                # At h=1 both branches are identical, so reuse
                # the exact same model output/components.
                # ---------------------------------------------

                if step == 1:
                    gt_comp = (
                        rollout_comp
                    )

                else:
                    gt_input = (
                        gt_context.reshape(
                            batch_actual,
                            CONTEXT_LENGTH
                            * NUM_FIELDS,
                            nx,
                            ny,
                        )
                    )

                    (
                        _,
                        gt_comp,
                    ) = model(
                        gt_input,
                        params=param,
                        return_components=True,
                    )

                gt_metrics = (
                    extract_state_metrics(
                        comp=gt_comp,
                        latest_norm=(
                            gt_latest
                        ),
                        gt_latest_norm=(
                            gt_latest
                        ),
                        field_mean=(
                            field_mean
                        ),
                        field_std=(
                            field_std
                        ),
                    )
                )

                rows.extend(
                    tensors_to_rows(
                        metrics=(
                            gt_metrics
                        ),
                        split_label=(
                            split_label
                        ),
                        seed=seed,
                        subset_name=(
                            subset_name
                        ),
                        state_type="gt",
                        horizon=step,
                        ra=ra,
                        pr=pr,
                        trajectory=(
                            trajectory
                        ),
                        t0=t0,
                        phase_id=(
                            phase_id
                        ),
                    )
                )

            # =================================================
            # Advance both branches
            # =================================================

            current_rollout = (
                rollout_context[
                    :,
                    -1,
                ]
            )

            pred_next = (
                current_rollout
                + rollout_delta
            )

            gt_next = (
                future_norm[
                    :,
                    step - 1,
                ]
            )

            rollout_context = torch.cat(
                [
                    rollout_context[
                        :,
                        1:,
                    ],
                    pred_next.unsqueeze(
                        1
                    ),
                ],
                dim=1,
            )

            gt_context = torch.cat(
                [
                    gt_context[
                        :,
                        1:,
                    ],
                    gt_next.unsqueeze(
                        1
                    ),
                ],
                dim=1,
            )

        if (
            batch_index == 1
            or batch_index % 10 == 0
        ):
            print(
                f"  {subset_name}: "
                f"processed batches "
                f"{batch_index}"
            )

    return rows


# ============================================================
# Summary
# ============================================================

def summarize_group(
    group,
):
    def mean_col(
        name,
    ):
        return float(
            group[
                name
            ].mean()
        )

    def quantile_col(
        name,
        q,
    ):
        return float(
            group[
                name
            ].quantile(
                q
            )
        )

    return pd.Series(
        {
            "samples":
                len(group),

            "state_rel_l2_mean":
                mean_col(
                    "state_rel_l2"
                ),

            "b_state_rel_l2_mean":
                mean_col(
                    "b_state_rel_l2"
                ),

            "velocity_state_rel_l2_mean":
                mean_col(
                    "velocity_state_rel_l2"
                ),

            "abs_alpha_mean":
                mean_col(
                    "abs_alpha_b"
                ),

            "abs_alpha_p95":
                quantile_col(
                    "abs_alpha_b",
                    0.95,
                ),

            "alpha_negative_fraction":
                mean_col(
                    "alpha_negative"
                ),

            "path_b_rms_raw_mean":
                mean_col(
                    "path_b_rms_raw"
                ),

            "path_b_rms_raw_p95":
                quantile_col(
                    "path_b_rms_raw",
                    0.95,
                ),

            "path_b_rms_raw_max":
                float(
                    group[
                        "path_b_rms_raw"
                    ].max()
                ),

            "path_b_rms_safe_mean":
                mean_col(
                    "path_b_rms_safe"
                ),

            "path_b_cap_active_fraction":
                mean_col(
                    "path_b_cap_active"
                ),

            "path_b_scale_mean":
                mean_col(
                    "path_b_scale"
                ),

            "path_b_scale_min":
                float(
                    group[
                        "path_b_scale"
                    ].min()
                ),

            "old_injection_rms_mean":
                mean_col(
                    "old_injection_rms"
                ),

            "old_injection_rms_p95":
                quantile_col(
                    "old_injection_rms",
                    0.95,
                ),

            "canonical_path_b_rms_mean":
                mean_col(
                    "canonical_path_b_rms"
                ),

            "canonical_path_b_rms_p95":
                quantile_col(
                    "canonical_path_b_rms",
                    0.95,
                ),

            "canonical_to_old_raw_ratio_mean":
                mean_col(
                    "canonical_to_old_raw_ratio"
                ),

            "velocity_mag_rms_mean":
                mean_col(
                    "velocity_mag_rms"
                ),

            "grad_b_mag_rms_mean":
                mean_col(
                    "grad_b_mag_rms"
                ),
        }
    )


def make_summary(
    rows_df,
):
    blocks = []

    overall = (
        rows_df.groupby(
            [
                "split_label",
                "seed",
                "subset",
                "state_type",
                "horizon",
            ],
            sort=True,
        )
        .apply(
            summarize_group
        )
        .reset_index()
    )

    overall[
        "scope"
    ] = "overall"

    overall[
        "phase"
    ] = None

    blocks.append(
        overall
    )

    phase = (
        rows_df.groupby(
            [
                "split_label",
                "seed",
                "subset",
                "state_type",
                "horizon",
                "phase",
            ],
            sort=True,
        )
        .apply(
            summarize_group
        )
        .reset_index()
    )

    phase[
        "scope"
    ] = "phase"

    blocks.append(
        phase
    )

    return pd.concat(
        blocks,
        ignore_index=True,
        sort=False,
    )


def print_compact(
    summary_df,
):
    overall = summary_df[
        summary_df[
            "scope"
        ]
        == "overall"
    ].copy()

    print()
    print(
        "========================================"
    )
    print(
        "D1-1B OVERALL GT vs ROLLOUT"
    )
    print(
        "========================================"
    )

    cols = [
        "split_label",
        "seed",
        "subset",
        "state_type",
        "horizon",
        "samples",
        "state_rel_l2_mean",
        "abs_alpha_mean",
        "alpha_negative_fraction",
        "path_b_rms_raw_mean",
        "path_b_rms_raw_p95",
        "path_b_rms_raw_max",
        "path_b_cap_active_fraction",
        "path_b_scale_min",
        "old_injection_rms_mean",
        "canonical_path_b_rms_mean",
        "velocity_mag_rms_mean",
        "grad_b_mag_rms_mean",
    ]

    print(
        overall[
            cols
        ].to_string(
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
        "ROLLOUT / GT PATH-B RATIO"
    )
    print(
        "========================================"
    )

    metric = (
        overall[
            [
                "split_label",
                "seed",
                "subset",
                "state_type",
                "horizon",
                "path_b_rms_raw_mean",
                "canonical_path_b_rms_mean",
                "old_injection_rms_mean",
                "velocity_mag_rms_mean",
                "grad_b_mag_rms_mean",
                "path_b_cap_active_fraction",
            ]
        ]
        .copy()
    )

    gt = metric[
        metric[
            "state_type"
        ]
        == "gt"
    ].copy()

    rollout = metric[
        metric[
            "state_type"
        ]
        == "rollout"
    ].copy()

    key_cols = [
        "split_label",
        "seed",
        "subset",
        "horizon",
    ]

    merged = rollout.merge(
        gt,
        on=key_cols,
        suffixes=(
            "_rollout",
            "_gt",
        ),
    )

    for name in [
        "path_b_rms_raw_mean",
        "canonical_path_b_rms_mean",
        "old_injection_rms_mean",
        "velocity_mag_rms_mean",
        "grad_b_mag_rms_mean",
    ]:
        merged[
            name
            + "_ratio_rollout_gt"
        ] = (
            merged[
                name
                + "_rollout"
            ]
            /
            (
                merged[
                    name
                    + "_gt"
                ]
                + 1.0e-30
            )
        )

    ratio_cols = [
        "split_label",
        "seed",
        "subset",
        "horizon",
        "path_b_rms_raw_mean_ratio_rollout_gt",
        "canonical_path_b_rms_mean_ratio_rollout_gt",
        "old_injection_rms_mean_ratio_rollout_gt",
        "velocity_mag_rms_mean_ratio_rollout_gt",
        "grad_b_mag_rms_mean_ratio_rollout_gt",
        "path_b_cap_active_fraction_rollout",
        "path_b_cap_active_fraction_gt",
    ]

    print(
        merged[
            ratio_cols
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6e}"
            ),
        )
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "D1-1B existing-M10 rollout-state audit. "
            "Compare GT manifold and autoregressive predicted "
            "state using formal M10-2 B-only checkpoints."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--stats",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--split_label",
        required=True,
        choices=[
            "unseen_pr",
            "unseen_ra",
        ],
    )

    parser.add_argument(
        "--seeds",
        default="42,123,2026",
        type=str,
    )

    parser.add_argument(
        "--subset",
        default="both",
        choices=[
            "train",
            "val",
            "both",
        ],
    )

    parser.add_argument(
        "--time_samples_per_trajectory",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=[
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/m10_2",
        type=str,
    )

    parser.add_argument(
        "--output_prefix",
        required=True,
        type=str,
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
        "D1-1B ROLLOUT-STATE AUDIT"
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
        "Checkpoint modification: OFF"
    )

    print(
        "Utility Gate: OFF"
    )

    print(
        "TEST access: FORBIDDEN"
    )

    print(
        "Branches: GT vs autoregressive rollout"
    )

    print(
        "Report horizons:",
        REPORT_HORIZONS,
    )

    print()
    print(
        "Canonical diagnostic interface:"
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
        "  dt =",
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

    if not os.path.exists(
        DATA_PATH
    ):
        raise FileNotFoundError(
            DATA_PATH
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

    seeds = [
        int(
            x.strip()
        )
        for x in args.seeds.split(",")
        if x.strip()
    ]

    if not seeds:
        raise ValueError(
            "No seeds."
        )

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as f:
        split_config = json.load(f)

    (
        field_mean_np,
        field_std_np,
    ) = load_stats(
        args.stats
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

    # ========================================================
    # Build exactly the same stratified states for all seeds.
    # ========================================================

    datasets = {}

    for subset_name in subset_names:

        dataset = (
            H16StratifiedGTSequenceDataset(
                split_items=(
                    split_config[
                        subset_name
                    ]
                ),
                data_path=DATA_PATH,
                field_mean=(
                    field_mean_np
                ),
                field_std=(
                    field_std_np
                ),
                time_samples_per_trajectory=(
                    args.time_samples_per_trajectory
                ),
            )
        )

        datasets[
            subset_name
        ] = dataset

        print()
        print(
            f"========== {subset_name.upper()} DATA =========="
        )

        print(
            "samples:",
            len(dataset),
        )

        condition_counts = {}

        phase_counts = {}

        for meta in dataset.samples:

            key = (
                meta[
                    "ra"
                ],
                meta[
                    "pr"
                ],
            )

            condition_counts[
                key
            ] = (
                condition_counts.get(
                    key,
                    0,
                )
                + 1
            )

            phase = meta[
                "phase"
            ]

            phase_counts[
                phase
            ] = (
                phase_counts.get(
                    phase,
                    0,
                )
                + 1
            )

        for (
            ra,
            pr,
        ), count in sorted(
            condition_counts.items()
        ):
            print(
                f"  Ra={ra:.1e}, "
                f"Pr={pr:g}: "
                f"{count}"
            )

        print(
            "phases:",
            phase_counts,
        )

    all_rows = []

    # ========================================================
    # Checkpoints
    # ========================================================

    for seed in seeds:

        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            (
                "m10_2_bonly_rmscap_h4_"
                f"{args.split_label}_"
                f"seed{seed}_best.pth"
            ),
        )

        if not os.path.exists(
            checkpoint_path
        ):
            raise FileNotFoundError(
                checkpoint_path
            )

        print()
        print(
            "========================================"
        )

        print(
            f"CHECKPOINT: "
            f"{args.split_label} / seed {seed}"
        )

        print(
            "========================================"
        )

        (
            model,
            metadata,
            ckpt_mean_np,
            ckpt_std_np,
        ) = (
            build_model_from_checkpoint(
                checkpoint_path=(
                    checkpoint_path
                ),
                stats_path=args.stats,
                device=device,
            )
        )

        if (
            metadata[
                "seed"
            ]
            != seed
        ):
            raise RuntimeError(
                "Seed mismatch."
            )

        print(
            "best_epoch:",
            metadata[
                "best_epoch"
            ],
        )

        print(
            "alpha_max:",
            metadata[
                "alpha_max"
            ],
        )

        print(
            "old dx:",
            metadata[
                "dx"
            ],
        )

        print(
            "old dy:",
            metadata[
                "dy"
            ],
        )

        print(
            "RMSCap:",
            metadata[
                "path_b_rms_cap"
            ],
        )

        for subset_name in subset_names:

            rows = audit_rollout(
                model=model,
                dataset=(
                    datasets[
                        subset_name
                    ]
                ),
                split_label=(
                    args.split_label
                ),
                seed=seed,
                subset_name=(
                    subset_name
                ),
                field_mean_np=(
                    ckpt_mean_np
                ),
                field_std_np=(
                    ckpt_std_np
                ),
                device=device,
                batch_size=(
                    args.batch_size
                ),
            )

            all_rows.extend(
                rows
            )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_rows:
        raise RuntimeError(
            "No D1-1B rows."
        )

    rows_df = pd.DataFrame(
        all_rows
    )

    summary_df = make_summary(
        rows_df
    )

    rows_path = (
        args.output_prefix
        + "_rows.csv"
    )

    summary_path = (
        args.output_prefix
        + "_summary.csv"
    )

    for path in [
        rows_path,
        summary_path,
    ]:
        directory = os.path.dirname(
            path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

    rows_df.to_csv(
        rows_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print_compact(
        summary_df
    )

    print()
    print(
        "========================================"
    )

    print(
        "INTERPRETATION GUARDRAIL"
    )

    print(
        "========================================"
    )

    print(
        "canonical_path_b_rms is diagnostic only."
    )

    print(
        "It is recomputed with dx=1/64, "
        "dy=1/63, dt=0.25 on the SAME state."
    )

    print(
        "Existing M10 predictions remain completely unchanged."
    )

    print(
        "Old RMSCap thresholds must NOT be directly interpreted "
        "as canonical-interface thresholds."
    )

    print()
    print(
        "Rows:",
        rows_path,
    )

    print(
        "Summary:",
        summary_path,
    )

    print()
    print(
        "✅ D1-1B finished."
    )


if __name__ == "__main__":
    main()
