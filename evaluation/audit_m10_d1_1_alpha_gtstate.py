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
    EPS,
)

from models.operators.fno2d_m10_2_bonly_rmscap import (
    M10BOnlyRMSCapFNO2d,
)


# ============================================================
# Locked D1 interpretation
# ============================================================

CONTEXT_LENGTH = 4

# Exact training rollout support:
# context 4 frames + 4 future frames.
H4_ROLLOUT_STEPS = 4

# D1-0 source-grounded canonical frame interval.
DT_CANONICAL = 0.25


# ============================================================
# Helpers
# ============================================================

def parse_ra_pr(group_name):
    match = re.search(
        r"ra_([0-9.eE+-]+)_pr_([0-9.eE+-]+)",
        group_name,
    )

    if match is None:
        raise ValueError(
            f"Cannot parse Ra/Pr from group: {group_name}"
        )

    return (
        float(match.group(1)),
        float(match.group(2)),
    )


def evenly_spaced_indices(
    start,
    end,
    count,
):
    if end < start:
        return []

    available = end - start + 1

    count = min(
        int(count),
        available,
    )

    if count <= 0:
        return []

    if count == 1:
        return [
            int(
                round(
                    0.5 * (start + end)
                )
            )
        ]

    values = np.linspace(
        start,
        end,
        num=count,
    )

    return sorted(
        set(
            int(round(v))
            for v in values
        )
    )


def temporal_phase(
    t0,
    max_t0,
):
    if max_t0 <= 0:
        return "all"

    fraction = (
        float(t0)
        / float(max_t0)
    )

    if fraction < 1.0 / 3.0:
        return "early"

    if fraction < 2.0 / 3.0:
        return "middle"

    return "late"


def load_stats(
    stats_path,
):
    with open(
        stats_path,
        "r",
        encoding="utf-8",
    ) as f:
        stats = json.load(f)

    field_mean = np.asarray(
        [
            float(
                stats[field]["mean"]
            )
            for field in FIELD_ORDER
        ],
        dtype=np.float32,
    )

    field_std = np.asarray(
        [
            float(
                stats[field]["std"]
            )
            for field in FIELD_ORDER
        ],
        dtype=np.float32,
    )

    if np.any(field_std <= 0):
        raise RuntimeError(
            "All field std values must be positive."
        )

    return (
        field_mean,
        field_std,
    )


def load_checkpoint(
    path,
    device,
):
    try:
        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location=device,
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Expected dict checkpoint."
        )

    if "model_state_dict" not in payload:
        raise RuntimeError(
            "Checkpoint has no model_state_dict."
        )

    return payload


# ============================================================
# Exact H4-support stratified GT context dataset
# ============================================================

class H4GTContextDataset(Dataset):
    """
    D1-1A uses GT contexts only.

    For each selected training/validation trajectory:

        context:
            t0, t0+1, t0+2, t0+3

    t0 is restricted so the SAME context could have supported
    the formal H4 training protocol:

        future:
            t0+4 ... t0+7

    Therefore:

        max_t0 = T - 4 - 4

    No model-generated state enters this dataset.
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

        self.field_mean = (
            field_mean.astype(
                np.float32
            )
        )

        self.field_std = (
            field_std.astype(
                np.float32
            )
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
                        f"Missing HDF5 group: "
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

                # Exact H4 training support:
                #
                # context:
                #   t0 ... t0+3
                #
                # future:
                #   t0+4 ... t0+7
                #
                # Last valid future index <= T-1.
                max_t0 = (
                    total_t
                    - CONTEXT_LENGTH
                    - H4_ROLLOUT_STEPS
                )

                if max_t0 < 0:
                    continue

                t0_values = (
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

                    for t0 in t0_values:

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
                "No stratified D1-1 samples."
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

        fields = []

        for field in FIELD_ORDER:

            # [T=4, X, Y]
            values = np.asarray(
                group[field][
                    trajectory,
                    t0:
                    t0 + CONTEXT_LENGTH,
                ],
                dtype=np.float32,
            )

            fields.append(
                values
            )

        # fields list:
        #   4 × [T,X,Y]
        #
        # -> [T,C,X,Y]
        context_raw = np.stack(
            fields,
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

        # Exact RBCDataset / FieldWiseNormalizer convention.
        context_norm = (
            context_raw
            - mean
        ) / (
            std
            + float(EPS)
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
# Model construction from checkpoint contract
# ============================================================

def build_model_from_checkpoint(
    *,
    checkpoint_path,
    stats_path,
    device,
):
    payload = load_checkpoint(
        checkpoint_path,
        device,
    )

    experiment = payload.get(
        "experiment"
    )

    stage = payload.get(
        "stage"
    )

    mode = payload.get(
        "factorial_mode",
        "stateparam",
    )

    if experiment != (
        "M10-2-StateParam-BOnly-RMSCap-H4"
    ):
        raise RuntimeError(
            "Unexpected experiment: "
            f"{experiment}"
        )

    if stage != (
        "formal_candidate_bonly_rebase"
    ):
        raise RuntimeError(
            "Unexpected stage: "
            f"{stage}"
        )

    if mode != "stateparam":
        raise RuntimeError(
            "D1-1A is locked to formal "
            f"StateParam candidate, got {mode}"
        )

    dx = float(
        payload["dx"]
    )

    dy = float(
        payload["dy"]
    )

    alpha_max = float(
        payload["alpha_max"]
    )

    conditioner_info = (
        payload.get(
            "conditioner",
            {}
        )
    )

    conditioner_hidden = int(
        conditioner_info.get(
            "hidden_dim",
            32,
        )
    )

    stabilization = payload.get(
        "stabilization",
        {}
    )

    if (
        "path_b_rms_cap"
        not in stabilization
    ):
        raise RuntimeError(
            "Checkpoint missing "
            "stabilization.path_b_rms_cap"
        )

    path_b_rms_cap = float(
        stabilization[
            "path_b_rms_cap"
        ]
    )

    path_b_rms_eps = float(
        stabilization.get(
            "path_b_rms_eps",
            1.0e-12,
        )
    )

    (
        field_mean_np,
        field_std_np,
    ) = load_stats(
        stats_path
    )

    field_mean = (
        field_mean_np.tolist()
    )

    field_std = (
        field_std_np.tolist()
    )

    model = M10BOnlyRMSCapFNO2d(
        field_mean=field_mean,
        field_std=field_std,
        mode=mode,
        dx=dx,
        dy=dy,
        path_b_rms_cap=(
            path_b_rms_cap
        ),
        path_b_rms_eps=(
            path_b_rms_eps
        ),
        alpha_max=alpha_max,
        conditioner_hidden=(
            conditioner_hidden
        ),
        freeze_m6=True,
    ).to(
        device
    )

    model.load_state_dict(
        payload[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    # --------------------------------------------------------
    # Stats provenance check
    # --------------------------------------------------------

    expected_mean = torch.tensor(
        field_mean_np,
        dtype=model.field_mean.dtype,
        device=device,
    ).view(-1)

    expected_std = torch.tensor(
        field_std_np,
        dtype=model.field_std.dtype,
        device=device,
    ).view(-1)

    loaded_mean = (
        model.field_mean
        .view(-1)
    )

    loaded_std = (
        model.field_std
        .view(-1)
    )

    mean_err = float(
        torch.max(
            torch.abs(
                loaded_mean
                - expected_mean
            )
        ).item()
    )

    std_err = float(
        torch.max(
            torch.abs(
                loaded_std
                - expected_std
            )
        ).item()
    )

    if mean_err > 1.0e-7:
        raise RuntimeError(
            "Checkpoint field_mean "
            "does not match stats JSON. "
            f"max error={mean_err}"
        )

    if std_err > 1.0e-7:
        raise RuntimeError(
            "Checkpoint field_std "
            "does not match stats JSON. "
            f"max error={std_err}"
        )

    metadata = {
        "experiment":
            experiment,

        "stage":
            stage,

        "mode":
            mode,

        "seed":
            int(
                payload.get(
                    "seed",
                    -1,
                )
            ),

        "epoch":
            int(
                payload.get(
                    "epoch",
                    -1,
                )
            ),

        "best_epoch":
            int(
                payload.get(
                    "best_epoch",
                    -1,
                )
            ),

        "best_val_loss":
            float(
                payload.get(
                    "best_val_loss",
                    float("nan"),
                )
            ),

        "dx":
            dx,

        "dy":
            dy,

        "alpha_max":
            alpha_max,

        "conditioner_hidden":
            conditioner_hidden,

        "path_b_rms_cap":
            path_b_rms_cap,

        "path_b_rms_eps":
            path_b_rms_eps,

        "mean_buffer_max_error":
            mean_err,

        "std_buffer_max_error":
            std_err,
    }

    return (
        model,
        metadata,
        field_mean_np,
        field_std_np,
    )


# ============================================================
# One checkpoint / one subset
# ============================================================

@torch.no_grad()
def audit_checkpoint_subset(
    *,
    model,
    metadata,
    dataset,
    split_label,
    seed,
    subset_name,
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

    rows = []

    phase_name = {
        0: "early",
        1: "middle",
        2: "late",
        3: "all",
    }

    alpha_max = float(
        metadata[
            "alpha_max"
        ]
    )

    # --------------------------------------------------------
    # Capacity threshold:
    #
    # If
    #     P_can ~ dt * P_old
    #
    # and the canonical gate keeps the SAME alpha_max,
    # then old |alpha| above:
    #
    #     dt * alpha_max
    #
    # cannot be reproduced by simply scaling alpha up
    # within the same bound.
    #
    # This is ONLY a D1 capacity diagnostic.
    # --------------------------------------------------------

    capacity_threshold_old_alpha = (
        DT_CANONICAL
        * alpha_max
    )

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        (
            context_norm,
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
        )

        param = (
            param
            .to(
                device
            )
            .float()
        )

        (
            batch_actual,
            context_len,
            channels,
            nx,
            ny,
        ) = context_norm.shape

        if context_len != 4:
            raise RuntimeError(
                "Expected context length 4."
            )

        if channels != 4:
            raise RuntimeError(
                "Expected 4 fields."
            )

        model_input = (
            context_norm.reshape(
                batch_actual,
                context_len
                * channels,
                nx,
                ny,
            )
        )

        (
            _,
            comp,
        ) = model(
            model_input,
            params=param,
            return_components=True,
        )

        alpha_b = (
            comp[
                "alpha_b"
            ]
            .detach()
            .cpu()
            .double()
            .numpy()
        )

        base_raw_b = (
            comp[
                "base_raw"
            ][
                :,
                1,
            ]
            .detach()
            .cpu()
            .double()
            .numpy()
        )

        dynamic_raw_b = (
            comp[
                "dynamic_raw"
            ][
                :,
                1,
            ]
            .detach()
            .cpu()
            .double()
            .numpy()
        )

        combined_raw_b = (
            comp[
                "combined_raw"
            ][
                :,
                1,
            ]
            .detach()
            .cpu()
            .double()
            .numpy()
        )

        path_b_rms_raw = (
            comp[
                "path_b_rms_raw"
            ]
            .detach()
            .cpu()
            .double()
            .numpy()
        )

        path_b_rms_safe = (
            comp[
                "path_b_rms_safe"
            ]
            .detach()
            .cpu()
            .double()
            .numpy()
        )

        path_b_scale = (
            comp[
                "path_b_scale"
            ]
            .detach()
            .cpu()
            .double()
            .numpy()
        )

        cap_active = (
            comp[
                "path_b_cap_active"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )

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

        for i in range(
            batch_actual
        ):
            alpha = float(
                alpha_b[i]
            )

            abs_alpha = abs(
                alpha
            )

            # Exact old residual spatial RMS because alpha_B
            # is one scalar per sample:
            #
            # RMS(alpha * P_safe)
            #     = |alpha| * RMS(P_safe)
            injection_rms = (
                abs_alpha
                * float(
                    path_b_rms_safe[i]
                )
            )

            # Approximate coefficient required if the only
            # change were:
            #
            #     P_new = 0.25 * P_old
            #
            # This is NOT proposed as a final D2 coefficient.
            alpha_canonical_equiv = (
                alpha
                / DT_CANONICAL
            )

            rows.append(
                {
                    "split_label":
                        split_label,

                    "seed":
                        int(seed),

                    "subset":
                        subset_name,

                    "ra":
                        float(
                            ra_np[i]
                        ),

                    "pr":
                        float(
                            pr_np[i]
                        ),

                    "phase":
                        phase_name[
                            int(
                                phase_np[i]
                            )
                        ],

                    "trajectory":
                        int(
                            trajectory_np[i]
                        ),

                    "t0":
                        int(
                            t0_np[i]
                        ),

                    "alpha_b":
                        alpha,

                    "abs_alpha_b":
                        abs_alpha,

                    "alpha_max":
                        alpha_max,

                    "capacity_threshold_old_alpha":
                        capacity_threshold_old_alpha,

                    "abs_alpha_gt_capacity_threshold":
                        bool(
                            abs_alpha
                            >
                            capacity_threshold_old_alpha
                        ),

                    "alpha_near_saturation":
                        bool(
                            abs_alpha
                            >=
                            0.95
                            * alpha_max
                        ),

                    "alpha_positive":
                        bool(
                            alpha > 1.0e-12
                        ),

                    "alpha_negative":
                        bool(
                            alpha < -1.0e-12
                        ),

                    "alpha_canonical_equiv_approx":
                        alpha_canonical_equiv,

                    "abs_alpha_canonical_equiv_approx":
                        abs(
                            alpha_canonical_equiv
                        ),

                    "canonical_equiv_exceeds_alpha_max":
                        bool(
                            abs(
                                alpha_canonical_equiv
                            )
                            >
                            alpha_max
                        ),

                    "base_raw_b":
                        float(
                            base_raw_b[i]
                        ),

                    "dynamic_raw_b":
                        float(
                            dynamic_raw_b[i]
                        ),

                    "combined_raw_b":
                        float(
                            combined_raw_b[i]
                        ),

                    "path_b_rms_raw":
                        float(
                            path_b_rms_raw[i]
                        ),

                    "path_b_rms_safe":
                        float(
                            path_b_rms_safe[i]
                        ),

                    "path_b_scale":
                        float(
                            path_b_scale[i]
                        ),

                    "path_b_cap_active":
                        bool(
                            cap_active[i]
                        ),

                    "old_injection_rms":
                        injection_rms,

                    "checkpoint_dx":
                        float(
                            metadata["dx"]
                        ),

                    "checkpoint_dy":
                        float(
                            metadata["dy"]
                        ),

                    "checkpoint_path_b_rms_cap":
                        float(
                            metadata[
                                "path_b_rms_cap"
                            ]
                        ),

                    "checkpoint_best_epoch":
                        int(
                            metadata[
                                "best_epoch"
                            ]
                        ),

                    "checkpoint_best_val_loss":
                        float(
                            metadata[
                                "best_val_loss"
                            ]
                        ),
                }
            )

        if (
            batch_index == 1
            or batch_index % 25 == 0
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

def summarize_one_group(
    group,
):
    abs_alpha = (
        group[
            "abs_alpha_b"
        ].to_numpy(
            dtype=float
        )
    )

    alpha = (
        group[
            "alpha_b"
        ].to_numpy(
            dtype=float
        )
    )

    canonical_equiv_abs = (
        group[
            "abs_alpha_canonical_equiv_approx"
        ].to_numpy(
            dtype=float
        )
    )

    path_raw = (
        group[
            "path_b_rms_raw"
        ].to_numpy(
            dtype=float
        )
    )

    path_safe = (
        group[
            "path_b_rms_safe"
        ].to_numpy(
            dtype=float
        )
    )

    injection = (
        group[
            "old_injection_rms"
        ].to_numpy(
            dtype=float
        )
    )

    path_scale = (
        group[
            "path_b_scale"
        ].to_numpy(
            dtype=float
        )
    )

    return pd.Series(
        {
            "samples":
                len(group),

            "alpha_mean":
                float(
                    np.mean(alpha)
                ),

            "alpha_std":
                float(
                    np.std(
                        alpha,
                        ddof=0,
                    )
                ),

            "abs_alpha_mean":
                float(
                    np.mean(
                        abs_alpha
                    )
                ),

            "abs_alpha_median":
                float(
                    np.median(
                        abs_alpha
                    )
                ),

            "abs_alpha_p90":
                float(
                    np.quantile(
                        abs_alpha,
                        0.90,
                    )
                ),

            "abs_alpha_p95":
                float(
                    np.quantile(
                        abs_alpha,
                        0.95,
                    )
                ),

            "abs_alpha_max":
                float(
                    np.max(
                        abs_alpha
                    )
                ),

            "frac_abs_alpha_gt_0p0625":
                float(
                    np.mean(
                        group[
                            "abs_alpha_gt_capacity_threshold"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                ),

            "frac_canonical_equiv_gt_alpha_max":
                float(
                    np.mean(
                        group[
                            "canonical_equiv_exceeds_alpha_max"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                ),

            "frac_near_alpha_saturation":
                float(
                    np.mean(
                        group[
                            "alpha_near_saturation"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                ),

            "alpha_positive_fraction":
                float(
                    np.mean(
                        group[
                            "alpha_positive"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                ),

            "alpha_negative_fraction":
                float(
                    np.mean(
                        group[
                            "alpha_negative"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                ),

            "canonical_equiv_abs_mean":
                float(
                    np.mean(
                        canonical_equiv_abs
                    )
                ),

            "canonical_equiv_abs_p95":
                float(
                    np.quantile(
                        canonical_equiv_abs,
                        0.95,
                    )
                ),

            "base_raw_b_mean":
                float(
                    group[
                        "base_raw_b"
                    ].mean()
                ),

            "dynamic_raw_b_mean":
                float(
                    group[
                        "dynamic_raw_b"
                    ].mean()
                ),

            "dynamic_raw_b_abs_mean":
                float(
                    group[
                        "dynamic_raw_b"
                    ].abs().mean()
                ),

            "combined_raw_b_mean":
                float(
                    group[
                        "combined_raw_b"
                    ].mean()
                ),

            "path_b_rms_raw_mean":
                float(
                    np.mean(
                        path_raw
                    )
                ),

            "path_b_rms_raw_p95":
                float(
                    np.quantile(
                        path_raw,
                        0.95,
                    )
                ),

            "path_b_rms_raw_max":
                float(
                    np.max(
                        path_raw
                    )
                ),

            "path_b_rms_safe_mean":
                float(
                    np.mean(
                        path_safe
                    )
                ),

            "path_b_cap_active_fraction":
                float(
                    group[
                        "path_b_cap_active"
                    ].mean()
                ),

            "path_b_scale_mean":
                float(
                    np.mean(
                        path_scale
                    )
                ),

            "path_b_scale_min":
                float(
                    np.min(
                        path_scale
                    )
                ),

            "old_injection_rms_mean":
                float(
                    np.mean(
                        injection
                    )
                ),

            "old_injection_rms_p95":
                float(
                    np.quantile(
                        injection,
                        0.95,
                    )
                ),

            "old_injection_rms_max":
                float(
                    np.max(
                        injection
                    )
                ),
        }
    )


def make_summary(
    rows_df,
):
    blocks = []

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall = (
        rows_df.groupby(
            [
                "split_label",
                "seed",
                "subset",
            ],
            sort=True,
            dropna=False,
        )
        .apply(
            summarize_one_group
        )
        .reset_index()
    )

    overall.insert(
        3,
        "scope",
        "overall",
    )

    overall["ra"] = np.nan
    overall["pr"] = np.nan
    overall["phase"] = None

    blocks.append(
        overall
    )

    # --------------------------------------------------------
    # Condition
    # --------------------------------------------------------

    condition = (
        rows_df.groupby(
            [
                "split_label",
                "seed",
                "subset",
                "ra",
                "pr",
            ],
            sort=True,
            dropna=False,
        )
        .apply(
            summarize_one_group
        )
        .reset_index()
    )

    condition.insert(
        3,
        "scope",
        "condition",
    )

    condition["phase"] = None

    blocks.append(
        condition
    )

    # --------------------------------------------------------
    # Phase
    # --------------------------------------------------------

    phase = (
        rows_df.groupby(
            [
                "split_label",
                "seed",
                "subset",
                "phase",
            ],
            sort=True,
            dropna=False,
        )
        .apply(
            summarize_one_group
        )
        .reset_index()
    )

    phase.insert(
        3,
        "scope",
        "phase",
    )

    phase["ra"] = np.nan
    phase["pr"] = np.nan

    blocks.append(
        phase
    )

    # --------------------------------------------------------
    # Condition × phase
    # --------------------------------------------------------

    condition_phase = (
        rows_df.groupby(
            [
                "split_label",
                "seed",
                "subset",
                "ra",
                "pr",
                "phase",
            ],
            sort=True,
            dropna=False,
        )
        .apply(
            summarize_one_group
        )
        .reset_index()
    )

    condition_phase.insert(
        3,
        "scope",
        "condition_phase",
    )

    blocks.append(
        condition_phase
    )

    summary = pd.concat(
        blocks,
        ignore_index=True,
        sort=False,
    )

    preferred = [
        "split_label",
        "seed",
        "subset",
        "scope",
        "ra",
        "pr",
        "phase",
    ]

    remaining = [
        c
        for c in summary.columns
        if c not in preferred
    ]

    return summary[
        preferred
        + remaining
    ]


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "D1-1A Existing-M10 GT-state alpha audit. "
            "No training. TRAIN/VAL only. TEST forbidden."
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--stats",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--split_label",
        type=str,
        required=True,
        choices=[
            "unseen_pr",
            "unseen_ra",
        ],
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,2026",
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
    )

    parser.add_argument(
        "--time_samples_per_trajectory",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=[
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/m10_2",
    )

    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
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
        "D1-1A EXISTING-M10 GT-STATE ALPHA AUDIT"
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
        "GT contexts only: YES"
    )

    print(
        "Rollout-contaminated states: NO"
    )

    print(
        "TEST access: FORBIDDEN"
    )

    print(
        "Canonical dt used only for "
        "capacity interpretation:",
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

    if args.time_samples_per_trajectory <= 0:
        raise ValueError(
            "time_samples_per_trajectory "
            "must be positive."
        )

    seeds = [
        int(x.strip())
        for x in args.seeds.split(",")
        if x.strip()
    ]

    if not seeds:
        raise ValueError(
            "No seeds supplied."
        )

    device = torch.device(
        args.device
    )

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as f:
        split_config = json.load(f)

    if args.subset == "both":
        subset_names = [
            "train",
            "val",
        ]
    else:
        subset_names = [
            args.subset
        ]

    (
        field_mean_np,
        field_std_np,
    ) = load_stats(
        args.stats
    )

    # --------------------------------------------------------
    # Build deterministic datasets ONCE.
    # All seeds see exactly the same GT states.
    # --------------------------------------------------------

    datasets = {}

    for subset_name in subset_names:

        if subset_name not in split_config:
            raise KeyError(
                f"Split has no key: "
                f"{subset_name}"
            )

        dataset = H4GTContextDataset(
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

            condition = (
                meta["ra"],
                meta["pr"],
            )

            condition_counts[
                condition
            ] = (
                condition_counts.get(
                    condition,
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

        print(
            "conditions:"
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
    # Seed loop
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

        print(
            checkpoint_path
        )

        (
            model,
            metadata,
            _,
            _,
        ) = build_model_from_checkpoint(
            checkpoint_path=(
                checkpoint_path
            ),
            stats_path=args.stats,
            device=device,
        )

        if metadata[
            "seed"
        ] != seed:
            raise RuntimeError(
                "Checkpoint seed mismatch: "
                f"metadata={metadata['seed']}, "
                f"expected={seed}"
            )

        print(
            "experiment:",
            metadata[
                "experiment"
            ],
        )

        print(
            "stage:",
            metadata[
                "stage"
            ],
        )

        print(
            "mode:",
            metadata[
                "mode"
            ],
        )

        print(
            "best_epoch:",
            metadata[
                "best_epoch"
            ],
        )

        print(
            "best_val_loss:",
            metadata[
                "best_val_loss"
            ],
        )

        print(
            "dx:",
            metadata[
                "dx"
            ],
        )

        print(
            "dy:",
            metadata[
                "dy"
            ],
        )

        print(
            "alpha_max:",
            metadata[
                "alpha_max"
            ],
        )

        print(
            "Path-B RMS cap:",
            metadata[
                "path_b_rms_cap"
            ],
        )

        threshold = (
            DT_CANONICAL
            * metadata[
                "alpha_max"
            ]
        )

        print(
            "old-alpha capacity threshold "
            "under 0.25 canonical signal:",
            threshold,
        )

        for subset_name in subset_names:

            rows = audit_checkpoint_subset(
                model=model,
                metadata=metadata,
                dataset=(
                    datasets[
                        subset_name
                    ]
                ),
                split_label=(
                    args.split_label
                ),
                seed=seed,
                subset_name=subset_name,
                device=device,
                batch_size=args.batch_size,
            )

            all_rows.extend(
                rows
            )

        # Release one full FNO checkpoint
        # before loading next seed.
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_rows:
        raise RuntimeError(
            "D1-1A produced no rows."
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

    # ========================================================
    # Compact terminal output
    # ========================================================

    print()
    print(
        "========================================"
    )
    print(
        "D1-1A OVERALL ALPHA SUMMARY"
    )
    print(
        "========================================"
    )

    overall = summary_df[
        summary_df[
            "scope"
        ]
        == "overall"
    ][
        [
            "split_label",
            "seed",
            "subset",
            "samples",
            "alpha_mean",
            "abs_alpha_mean",
            "abs_alpha_median",
            "abs_alpha_p90",
            "abs_alpha_p95",
            "abs_alpha_max",
            "frac_abs_alpha_gt_0p0625",
            "frac_near_alpha_saturation",
            "alpha_positive_fraction",
            "alpha_negative_fraction",
            "path_b_cap_active_fraction",
            "path_b_rms_raw_mean",
            "path_b_rms_safe_mean",
            "old_injection_rms_mean",
        ]
    ].copy()

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
        "D1-1A TEMPORAL PHASE SUMMARY"
    )
    print(
        "========================================"
    )

    phase = summary_df[
        summary_df[
            "scope"
        ]
        == "phase"
    ][
        [
            "split_label",
            "seed",
            "subset",
            "phase",
            "samples",
            "abs_alpha_mean",
            "abs_alpha_p95",
            "frac_abs_alpha_gt_0p0625",
            "alpha_positive_fraction",
            "alpha_negative_fraction",
            "path_b_cap_active_fraction",
            "path_b_rms_raw_mean",
            "old_injection_rms_mean",
        ]
    ].copy()

    phase_order = {
        "early": 0,
        "middle": 1,
        "late": 2,
    }

    phase[
        "_phase_order"
    ] = (
        phase[
            "phase"
        ].map(
            phase_order
        )
    )

    phase = (
        phase.sort_values(
            [
                "seed",
                "subset",
                "_phase_order",
            ]
        )
        .drop(
            columns=[
                "_phase_order"
            ]
        )
    )

    print(
        phase.to_string(
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
        "INTERPRETATION GUARDRAIL"
    )
    print(
        "========================================"
    )

    print(
        "0.0625 = 0.25(dt) * 0.25(old alpha_max)."
    )

    print(
        "frac_abs_alpha_gt_0p0625 measures "
        "whether keeping alpha_max=0.25 after a pure "
        "0.25 signal shrink would reduce old residual capacity."
    )

    print(
        "This is a capacity diagnostic ONLY."
    )

    print(
        "It is NOT the final D2 alpha design because "
        "canonical dy and RMSCap representation must also "
        "be redesigned consistently."
    )

    print()
    print(
        "Detailed rows:"
    )
    print(
        rows_path
    )

    print(
        "Summary:"
    )
    print(
        summary_path
    )

    print()
    print(
        "✅ D1-1A finished."
    )


if __name__ == "__main__":
    main()
