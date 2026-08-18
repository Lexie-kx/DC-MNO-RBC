import os
import sys
import re
import json
import math
import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Project root
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

from evaluation.audit_m10_dimensional_interface_d1 import (
    DX_OLD,
    DY_OLD,
    DX_CANONICAL,
    DY_CANONICAL,
    DT_CANONICAL,
    VARIANT_ORDER,
    load_field_stats,
    compute_path_b_norm,
    PairAccumulator,
    variant_metadata,
)


CONTEXT_LENGTH = 4
NUM_FIELDS = 4


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
    """
    Inclusive deterministic temporal sampling.
    """

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

    indices = sorted(
        set(
            int(round(v))
            for v in values
        )
    )

    return indices


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


# ============================================================
# Explicit stratified transition dataset
# ============================================================

class StratifiedTransitionDataset(Dataset):
    """
    Directly build GT transitions using:

        group
        trajectory
        t0

    One sample uses:

        context frames:
            t0 ... t0+3

        current state:
            t0+3

        next state:
            t0+4

    Only current/next raw states are returned because
    D1-0 does not use a model.
    """

    def __init__(
        self,
        split_items,
        data_path,
        time_samples_per_trajectory,
    ):
        self.data_path = str(data_path)
        self.samples = []
        self._h5 = None

        with h5py.File(
            self.data_path,
            "r",
        ) as f:

            for item in split_items:

                group_name = item["group"]
                trajectories = item[
                    "trajectories"
                ]

                if group_name not in f:
                    raise KeyError(
                        f"Missing HDF5 group: "
                        f"{group_name}"
                    )

                group = f[group_name]

                total_t = int(
                    group[
                        FIELD_ORDER[0]
                    ].shape[1]
                )

                # Need:
                # current = t0 + 3
                # next    = t0 + 4
                max_t0 = (
                    total_t
                    - CONTEXT_LENGTH
                    - 1
                )

                if max_t0 < 0:
                    continue

                chosen_t0 = (
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

                    for t0 in chosen_t0:

                        self.samples.append(
                            {
                                "group": group_name,
                                "trajectory": int(
                                    trajectory
                                ),
                                "t0": int(t0),
                                "current_t": int(
                                    t0
                                    + CONTEXT_LENGTH
                                    - 1
                                ),
                                "next_t": int(
                                    t0
                                    + CONTEXT_LENGTH
                                ),
                                "max_t0": int(
                                    max_t0
                                ),
                                "phase": (
                                    temporal_phase(
                                        t0,
                                        max_t0,
                                    )
                                ),
                                "ra": float(ra),
                                "pr": float(pr),
                            }
                        )

        if not self.samples:
            raise RuntimeError(
                "No D1 stratified samples built."
            )

    def __len__(self):
        return len(self.samples)

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

        current_t = meta[
            "current_t"
        ]

        next_t = meta[
            "next_t"
        ]

        current_fields = []

        next_fields = []

        for field in FIELD_ORDER:

            current_fields.append(
                np.asarray(
                    group[field][
                        trajectory,
                        current_t,
                    ],
                    dtype=np.float32,
                )
            )

            next_fields.append(
                np.asarray(
                    group[field][
                        trajectory,
                        next_t,
                    ],
                    dtype=np.float32,
                )
            )

        current_raw = torch.from_numpy(
            np.stack(
                current_fields,
                axis=0,
            )
        )

        next_raw = torch.from_numpy(
            np.stack(
                next_fields,
                axis=0,
            )
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
            current_raw,
            next_raw,
            torch.tensor(
                meta["ra"],
                dtype=torch.float64,
            ),
            torch.tensor(
                meta["pr"],
                dtype=torch.float64,
            ),
            torch.tensor(
                meta["trajectory"],
                dtype=torch.long,
            ),
            torch.tensor(
                meta["t0"],
                dtype=torch.long,
            ),
            torch.tensor(
                phase_id,
                dtype=torch.long,
            ),
        )


# ============================================================
# State diagnostic accumulator
# ============================================================

class StateAccumulator:

    def __init__(self):
        self.points = 0
        self.sum_velocity_mag2 = 0.0

    def update(
        self,
        current_native,
    ):
        ux = (
            current_native[
                :,
                UX_IDX,
            ]
            .detach()
            .double()
        )

        uy = (
            current_native[
                :,
                UY_IDX,
            ]
            .detach()
            .double()
        )

        velocity_mag2 = (
            ux * ux
            +
            uy * uy
        )

        self.points += int(
            velocity_mag2.numel()
        )

        self.sum_velocity_mag2 += float(
            velocity_mag2.sum().item()
        )

    def finalize(self):

        if self.points <= 0:
            return {
                "velocity_mag_rms": (
                    float("nan")
                )
            }

        value = math.sqrt(
            self.sum_velocity_mag2
            / float(self.points)
        )

        return {
            "velocity_mag_rms": value
        }


# ============================================================
# Audit
# ============================================================

@torch.no_grad()
def run_subset(
    subset_name,
    dataset,
    field_mean,
    field_std,
    batch_size,
    device,
):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print()
    print(
        "========================================"
    )
    print(
        f"STRATIFIED SUBSET: "
        f"{subset_name.upper()}"
    )
    print(
        "========================================"
    )

    print(
        "Samples:",
        len(dataset),
    )

    # Show sampling distribution.
    condition_counts = defaultdict(
        int
    )

    phase_counts = defaultdict(
        int
    )

    trajectory_counts = defaultdict(
        set
    )

    for meta in dataset.samples:

        condition = (
            meta["ra"],
            meta["pr"],
        )

        condition_counts[
            condition
        ] += 1

        phase_counts[
            meta["phase"]
        ] += 1

        trajectory_counts[
            condition
        ].add(
            meta["trajectory"]
        )

    print()
    print(
        "Per-condition sampling:"
    )

    for condition in sorted(
        condition_counts
    ):
        ra, pr = condition

        print(
            f"  Ra={ra:.1e}, "
            f"Pr={pr:g} | "
            f"samples="
            f"{condition_counts[condition]} | "
            f"trajectories="
            f"{len(trajectory_counts[condition])}"
        )

    print(
        "Temporal phases:",
        dict(phase_counts),
    )

    mean_dev = field_mean.to(
        device
    )

    std_dev = field_std.to(
        device
    )

    norm_scale = (
        std_dev
        + float(EPS)
    ).view(
        1,
        NUM_FIELDS,
        1,
        1,
    )

    mean_view = (
        mean_dev.view(
            1,
            NUM_FIELDS,
            1,
            1,
        )
    )

    std_view = (
        std_dev.view(
            1,
            NUM_FIELDS,
            1,
            1,
        )
    )

    pair_acc = defaultdict(
        PairAccumulator
    )

    state_acc = defaultdict(
        StateAccumulator
    )

    max_reconstruction_error = 0.0

    phase_name = {
        0: "early",
        1: "middle",
        2: "late",
        3: "all",
    }

    def update_bucket(
        scope,
        ra_value,
        pr_value,
        phase_value,
        index_tensor,
        current_native,
        gt_delta,
        rb_map,
    ):
        bucket_key = (
            scope,
            ra_value,
            pr_value,
            phase_value,
        )

        current_part = (
            current_native.index_select(
                0,
                index_tensor,
            )
        )

        gt_part = (
            gt_delta.index_select(
                0,
                index_tensor,
            )
        )

        state_acc[
            bucket_key
        ].update(
            current_part
        )

        for variant in VARIANT_ORDER:

            rb_part = (
                rb_map[
                    variant
                ].index_select(
                    0,
                    index_tensor,
                )
            )

            pair_acc[
                bucket_key
                + (variant,)
            ].update(
                rb_part,
                gt_part,
            )

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        (
            current_raw,
            next_raw,
            ra,
            pr,
            trajectory,
            t0,
            phase_id,
        ) = batch

        current_raw = current_raw.to(
            device
        )

        next_raw = next_raw.to(
            device
        )

        # ----------------------------------------------------
        # Exact RBCDataset Z-score convention:
        #
        #   x_norm = (x - mean) / (std + eps)
        # ----------------------------------------------------

        current_norm = (
            current_raw
            - mean_view
        ) / norm_scale

        next_norm = (
            next_raw
            - mean_view
        ) / norm_scale

        # ----------------------------------------------------
        # Exact M10 reconstruction convention:
        #
        #   current_native_recon
        #       = current_norm * raw_std + mean
        #
        # Note:
        # normalizer uses std+eps while M10 inverse uses raw std.
        # Difference should be numerically negligible.
        # ----------------------------------------------------

        current_native = (
            current_norm
            * std_view
            + mean_view
        )

        reconstruction_error = (
            current_native
            - current_raw
        ).abs().max().item()

        max_reconstruction_error = max(
            max_reconstruction_error,
            reconstruction_error,
        )

        # ----------------------------------------------------
        # GT normalized one-frame buoyancy increment
        # ----------------------------------------------------

        gt_delta_b_norm = (
            next_norm[
                :,
                B_IDX,
            ]
            -
            current_norm[
                :,
                B_IDX,
            ]
        )

        # ----------------------------------------------------
        # D1 variants
        # ----------------------------------------------------

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

        batch_size_actual = (
            current_raw.shape[0]
        )

        all_indices = torch.arange(
            batch_size_actual,
            device=device,
        )

        # Overall.
        update_bucket(
            scope="overall",
            ra_value=None,
            pr_value=None,
            phase_value=None,
            index_tensor=all_indices,
            current_native=current_native,
            gt_delta=gt_delta_b_norm,
            rb_map=rb_map,
        )

        # Condition / phase / condition-phase.
        groups = defaultdict(
            list
        )

        for i in range(
            batch_size_actual
        ):
            key = (
                round(
                    float(ra[i].item()),
                    6,
                ),
                round(
                    float(pr[i].item()),
                    6,
                ),
                phase_name[
                    int(
                        phase_id[i].item()
                    )
                ],
            )

            groups[key].append(i)

        for (
            ra_value,
            pr_value,
            phase_value,
        ), indices in groups.items():

            idx = torch.tensor(
                indices,
                dtype=torch.long,
                device=device,
            )

            update_bucket(
                scope="condition",
                ra_value=ra_value,
                pr_value=pr_value,
                phase_value=None,
                index_tensor=idx,
                current_native=current_native,
                gt_delta=gt_delta_b_norm,
                rb_map=rb_map,
            )

            update_bucket(
                scope="phase",
                ra_value=None,
                pr_value=None,
                phase_value=phase_value,
                index_tensor=idx,
                current_native=current_native,
                gt_delta=gt_delta_b_norm,
                rb_map=rb_map,
            )

            update_bucket(
                scope="condition_phase",
                ra_value=ra_value,
                pr_value=pr_value,
                phase_value=phase_value,
                index_tensor=idx,
                current_native=current_native,
                gt_delta=gt_delta_b_norm,
                rb_map=rb_map,
            )

        if (
            batch_index == 1
            or batch_index % 25 == 0
        ):
            print(
                "Processed batches:",
                batch_index,
            )

    # ========================================================
    # Finalize
    # ========================================================

    rows = []

    for key, accumulator in pair_acc.items():

        (
            scope,
            ra_value,
            pr_value,
            phase_value,
            variant,
        ) = key

        bucket_key = (
            scope,
            ra_value,
            pr_value,
            phase_value,
        )

        metrics = accumulator.finalize()

        state_metrics = (
            state_acc[
                bucket_key
            ].finalize()
        )

        meta = variant_metadata(
            variant
        )

        row = {
            "subset": subset_name,
            "scope": scope,
            "ra": ra_value,
            "pr": pr_value,
            "phase": phase_value,
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

        row.update(
            state_metrics
        )

        rows.append(row)

    print(
        "Max M10 reconstruction "
        "abs error:",
        f"{max_reconstruction_error:.8e}",
    )

    return rows


# ============================================================
# Printing
# ============================================================

def print_summary(
    df,
):

    print()
    print(
        "========================================"
    )
    print(
        "D1-0B OVERALL"
    )
    print(
        "========================================"
    )

    overall = df[
        df["scope"]
        == "overall"
    ][
        [
            "subset",
            "variant",
            "samples",
            "velocity_mag_rms",
            "gt_delta_rms",
            "rb_rms",
            "scale_ratio",
            "cosine",
            "pearson",
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
        "PER-CONDITION SCALE RATIO"
    )
    print(
        "========================================"
    )

    condition = df[
        df["scope"]
        == "condition"
    ].copy()

    pivot = condition.pivot_table(
        index=[
            "subset",
            "ra",
            "pr",
        ],
        columns="variant",
        values="scale_ratio",
        aggfunc="first",
    ).reset_index()

    print(
        pivot.to_string(
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
        "TEMPORAL PHASE"
    )
    print(
        "========================================"
    )

    phase = df[
        df["scope"]
        == "phase"
    ][
        [
            "subset",
            "phase",
            "variant",
            "samples",
            "velocity_mag_rms",
            "gt_delta_rms",
            "rb_rms",
            "scale_ratio",
            "cosine",
        ]
    ].copy()

    phase_order = {
        "early": 0,
        "middle": 1,
        "late": 2,
    }

    phase["_order"] = (
        phase["phase"]
        .map(phase_order)
    )

    phase = phase.sort_values(
        [
            "subset",
            "_order",
            "variant",
        ]
    ).drop(
        columns="_order"
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
        "SANITY"
    )
    print(
        "========================================"
    )

    for subset_name in sorted(
        overall["subset"].unique()
    ):

        sub = overall[
            overall["subset"]
            == subset_name
        ]

        grid = sub[
            sub["variant"]
            == "grid"
        ]

        canonical = sub[
            sub["variant"]
            == "canonical"
        ]

        if (
            len(grid) == 1
            and len(canonical) == 1
        ):
            ratio = (
                float(
                    canonical[
                        "rb_rms"
                    ].iloc[0]
                )
                /
                float(
                    grid[
                        "rb_rms"
                    ].iloc[0]
                )
            )

            cos_diff = (
                float(
                    canonical[
                        "cosine"
                    ].iloc[0]
                )
                -
                float(
                    grid[
                        "cosine"
                    ].iloc[0]
                )
            )

            print(
                f"{subset_name}: "
                f"canonical/grid RMS "
                f"= {ratio:.8f} "
                f"(expected 0.25000000)"
            )

            print(
                f"{subset_name}: "
                f"cosine difference "
                f"= {cos_diff:.8e} "
                f"(expected ~0)"
            )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "D1-0B stratified GT-state "
            "dimensional-interface audit."
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
        "--subset",
        choices=[
            "train",
            "val",
            "both",
        ],
        default="both",
    )

    parser.add_argument(
        "--time_samples_per_trajectory",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "cuda",
        ],
        default="cpu",
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
        "D1-0B STRATIFIED DIMENSIONAL AUDIT"
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
        "Model checkpoint: NOT LOADED"
    )
    print(
        "Gate / alpha / RMSCap: OFF"
    )
    print(
        "TEST: FORBIDDEN"
    )

    print()
    print(
        "Canonical analysis convention:"
    )
    print(
        f"  dx = {DX_CANONICAL}"
    )
    print(
        f"  dy = {DY_CANONICAL}"
    )
    print(
        f"  dt = {DT_CANONICAL}"
    )

    print(
        "Time samples / trajectory:",
        args.time_samples_per_trajectory,
    )

    if args.time_samples_per_trajectory <= 0:
        raise ValueError(
            "time_samples_per_trajectory "
            "must be positive."
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

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA unavailable."
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

    rows = []

    for subset_name in subset_names:

        if subset_name not in split_config:
            raise KeyError(
                f"No split key: "
                f"{subset_name}"
            )

        dataset = (
            StratifiedTransitionDataset(
                split_items=(
                    split_config[
                        subset_name
                    ]
                ),
                data_path=DATA_PATH,
                time_samples_per_trajectory=(
                    args.time_samples_per_trajectory
                ),
            )
        )

        subset_rows = run_subset(
            subset_name=subset_name,
            dataset=dataset,
            field_mean=field_mean,
            field_std=field_std,
            batch_size=args.batch_size,
            device=device,
        )

        rows.extend(
            subset_rows
        )

    df = pd.DataFrame(
        rows
    )

    if args.output is None:

        split_stem = Path(
            args.split
        ).stem

        output = (
            "outputs/tables/"
            "d1_stratified_"
            f"{split_stem}.csv"
        )

    else:
        output = args.output

    os.makedirs(
        os.path.dirname(output),
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    print_summary(
        df
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
        "✅ D1-0B finished."
    )


if __name__ == "__main__":
    main()
