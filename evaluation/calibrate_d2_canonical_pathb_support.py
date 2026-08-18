import os
import sys
import re
import json
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import DATA_PATH, FIELD_ORDER


# ============================================================
# Locked D2 canonical interface
# ============================================================

DX_OLD = 1.0 / 64.0
DY_OLD = 1.0 / 64.0

DX_CANONICAL = 1.0 / 64.0
DY_CANONICAL = 1.0 / 63.0
DT_CANONICAL = 0.25

CONTEXT_LENGTH = 4
H4_ROLLOUT_STEPS = 4

B_IDX = FIELD_ORDER.index("buoyancy")
UX_IDX = FIELD_ORDER.index("u_x")
UY_IDX = FIELD_ORDER.index("u_y")


def parse_ra_pr(group_name):
    match = re.search(
        r"ra_([0-9.eE+-]+)_pr_([0-9.eE+-]+)",
        group_name,
    )

    if match is None:
        raise ValueError(
            f"Cannot parse Ra/Pr from {group_name}"
        )

    return (
        float(match.group(1)),
        float(match.group(2)),
    )


def load_b_std(stats_path):
    with open(
        stats_path,
        "r",
        encoding="utf-8",
    ) as f:
        stats = json.load(f)

    return float(
        stats["buoyancy"]["std"]
    )


def grad_x_periodic(f, dx):
    return (
        np.roll(f, -1, axis=-2)
        -
        np.roll(f, 1, axis=-2)
    ) / (2.0 * dx)


def grad_y_nonperiodic(f, dy):
    grad = np.empty_like(
        f,
        dtype=np.float64,
    )

    grad[..., 1:-1] = (
        f[..., 2:]
        -
        f[..., :-2]
    ) / (2.0 * dy)

    grad[..., 0] = (
        -3.0 * f[..., 0]
        + 4.0 * f[..., 1]
        - f[..., 2]
    ) / (2.0 * dy)

    grad[..., -1] = (
        3.0 * f[..., -1]
        - 4.0 * f[..., -2]
        + f[..., -3]
    ) / (2.0 * dy)

    return grad


def path_b_norm(
    b,
    ux,
    uy,
    *,
    dx,
    dy,
    b_std,
    dt_factor,
):
    db_dx = grad_x_periodic(
        b,
        dx,
    )

    db_dy = grad_y_nonperiodic(
        b,
        dy,
    )

    rate = -(
        ux * db_dx
        +
        uy * db_dy
    )

    return (
        dt_factor
        * rate
        / b_std
    )


def per_state_rms(x):
    return np.sqrt(
        np.mean(
            np.asarray(
                x,
                dtype=np.float64,
            ) ** 2,
            axis=(-2, -1),
        )
    )


def summarize(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "samples":
            int(values.size),

        "p50":
            float(
                np.quantile(
                    values,
                    0.50,
                )
            ),

        "p95":
            float(
                np.quantile(
                    values,
                    0.95,
                )
            ),

        "p99":
            float(
                np.quantile(
                    values,
                    0.99,
                )
            ),

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "D2-0 canonical Path-B TRAIN-GT support calibration."
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
        "--output",
        required=True,
        type=str,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(
        "========================================"
    )
    print(
        "D2-0 CANONICAL PATH-B SUPPORT CALIBRATION"
    )
    print(
        "========================================"
    )

    print(
        "Stage: PRE-TRAINING CONTROL / CALIBRATION"
    )
    print(
        "Training: OFF"
    )
    print(
        "Checkpoint: NOT LOADED"
    )
    print(
        "VAL: NOT USED"
    )
    print(
        "TEST: FORBIDDEN"
    )

    print()
    print(
        "Canonical interface:"
    )
    print(
        "dx =", DX_CANONICAL
    )
    print(
        "dy =", DY_CANONICAL
    )
    print(
        "dt =", DT_CANONICAL
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

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as f:
        split_config = json.load(f)

    if "train" not in split_config:
        raise KeyError(
            "split JSON has no train key."
        )

    b_std = load_b_std(
        args.stats
    )

    print()
    print(
        "buoyancy raw std =",
        b_std,
    )

    rows = []

    old_all = []
    canonical_all = []

    with h5py.File(
        DATA_PATH,
        "r",
    ) as f:

        for item in split_config[
            "train"
        ]:
            group_name = item[
                "group"
            ]

            trajectories = item[
                "trajectories"
            ]

            if group_name not in f:
                raise KeyError(
                    f"Missing group: {group_name}"
                )

            group = f[
                group_name
            ]

            ra, pr = parse_ra_pr(
                group_name
            )

            total_t = int(
                group[
                    FIELD_ORDER[B_IDX]
                ].shape[1]
            )

            max_t0 = (
                total_t
                - CONTEXT_LENGTH
                - H4_ROLLOUT_STEPS
            )

            if max_t0 < 0:
                raise RuntimeError(
                    f"Trajectory too short: T={total_t}"
                )

            # Union of all GT current states that can appear
            # across the four H4 training rollout steps:
            #
            # current index =
            #     t0 + (CONTEXT_LENGTH - 1) + step
            #
            # step = 0..3
            #
            # Therefore unique support:
            #     3 .. 98 for T=100.
            current_start = (
                CONTEXT_LENGTH
                - 1
            )

            current_end = (
                max_t0
                + CONTEXT_LENGTH
                - 1
                + H4_ROLLOUT_STEPS
                - 1
            )

            condition_old = []
            condition_can = []

            for trajectory in trajectories:

                b = np.asarray(
                    group[
                        FIELD_ORDER[B_IDX]
                    ][
                        trajectory,
                        current_start:
                        current_end + 1,
                    ],
                    dtype=np.float64,
                )

                ux = np.asarray(
                    group[
                        FIELD_ORDER[UX_IDX]
                    ][
                        trajectory,
                        current_start:
                        current_end + 1,
                    ],
                    dtype=np.float64,
                )

                uy = np.asarray(
                    group[
                        FIELD_ORDER[UY_IDX]
                    ][
                        trajectory,
                        current_start:
                        current_end + 1,
                    ],
                    dtype=np.float64,
                )

                old = path_b_norm(
                    b,
                    ux,
                    uy,
                    dx=DX_OLD,
                    dy=DY_OLD,
                    b_std=b_std,
                    dt_factor=1.0,
                )

                canonical = path_b_norm(
                    b,
                    ux,
                    uy,
                    dx=DX_CANONICAL,
                    dy=DY_CANONICAL,
                    b_std=b_std,
                    dt_factor=DT_CANONICAL,
                )

                old_rms = per_state_rms(
                    old
                )

                can_rms = per_state_rms(
                    canonical
                )

                condition_old.extend(
                    old_rms.tolist()
                )

                condition_can.extend(
                    can_rms.tolist()
                )

                old_all.extend(
                    old_rms.tolist()
                )

                canonical_all.extend(
                    can_rms.tolist()
                )

            old_stats = summarize(
                condition_old
            )

            can_stats = summarize(
                condition_can
            )

            rows.append(
                {
                    "split_label":
                        args.split_label,

                    "scope":
                        "condition",

                    "ra":
                        ra,

                    "pr":
                        pr,

                    "trajectories":
                        len(
                            trajectories
                        ),

                    "current_start":
                        current_start,

                    "current_end":
                        current_end,

                    "samples":
                        can_stats[
                            "samples"
                        ],

                    "old_p50":
                        old_stats["p50"],

                    "old_p95":
                        old_stats["p95"],

                    "old_p99":
                        old_stats["p99"],

                    "old_max":
                        old_stats["max"],

                    "canonical_p50":
                        can_stats["p50"],

                    "canonical_p95":
                        can_stats["p95"],

                    "canonical_p99":
                        can_stats["p99"],

                    "canonical_mean":
                        can_stats["mean"],

                    "canonical_max":
                        can_stats["max"],

                    "canonical_max_over_old_max":
                        (
                            can_stats["max"]
                            /
                            (
                                old_stats["max"]
                                + 1.0e-30
                            )
                        ),
                }
            )

    old_overall = summarize(
        old_all
    )

    can_overall = summarize(
        canonical_all
    )

    rows.append(
        {
            "split_label":
                args.split_label,

            "scope":
                "overall",

            "ra":
                np.nan,

            "pr":
                np.nan,

            "trajectories":
                np.nan,

            "current_start":
                np.nan,

            "current_end":
                np.nan,

            "samples":
                can_overall[
                    "samples"
                ],

            "old_p50":
                old_overall["p50"],

            "old_p95":
                old_overall["p95"],

            "old_p99":
                old_overall["p99"],

            "old_max":
                old_overall["max"],

            "canonical_p50":
                can_overall["p50"],

            "canonical_p95":
                can_overall["p95"],

            "canonical_p99":
                can_overall["p99"],

            "canonical_mean":
                can_overall["mean"],

            "canonical_max":
                can_overall["max"],

            "canonical_max_over_old_max":
                (
                    can_overall["max"]
                    /
                    (
                        old_overall["max"]
                        + 1.0e-30
                    )
                ),
        }
    )

    df = pd.DataFrame(
        rows
    )

    output_dir = os.path.dirname(
        args.output
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    df.to_csv(
        args.output,
        index=False,
    )

    print()
    print(
        "========================================"
    )
    print(
        "CONDITION SUMMARY"
    )
    print(
        "========================================"
    )

    condition_df = df[
        df["scope"]
        == "condition"
    ][
        [
            "ra",
            "pr",
            "samples",
            "old_p95",
            "old_p99",
            "old_max",
            "canonical_p95",
            "canonical_p99",
            "canonical_max",
            "canonical_max_over_old_max",
        ]
    ]

    print(
        condition_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.8e}"
            ),
        )
    )

    overall = df[
        df["scope"]
        == "overall"
    ].iloc[0]

    print()
    print(
        "========================================"
    )
    print(
        "OVERALL TRAIN-GT SUPPORT"
    )
    print(
        "========================================"
    )

    print(
        "samples              =",
        int(
            overall["samples"]
        ),
    )

    print(
        "old P50              =",
        f"{overall['old_p50']:.10e}",
    )

    print(
        "old P95              =",
        f"{overall['old_p95']:.10e}",
    )

    print(
        "old P99              =",
        f"{overall['old_p99']:.10e}",
    )

    print(
        "old MAX              =",
        f"{overall['old_max']:.10e}",
    )

    print()

    print(
        "canonical P50        =",
        f"{overall['canonical_p50']:.10e}",
    )

    print(
        "canonical P95        =",
        f"{overall['canonical_p95']:.10e}",
    )

    print(
        "canonical P99        =",
        f"{overall['canonical_p99']:.10e}",
    )

    print(
        "canonical MAX        =",
        f"{overall['canonical_max']:.10e}",
    )

    print()

    print(
        "canonical/old MAX    =",
        f"{overall['canonical_max_over_old_max']:.10e}",
    )

    print()
    print(
        "LOCKED D2 RMSCap candidate:"
    )
    print(
        f"{overall['canonical_max']:.10e}"
    )

    print()
    print(
        "CSV saved to:"
    )
    print(
        args.output
    )

    print()
    print(
        "✅ D2-0 calibration finished."
    )


if __name__ == "__main__":
    main()
