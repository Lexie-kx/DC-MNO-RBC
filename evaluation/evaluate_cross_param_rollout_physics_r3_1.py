from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


from constants import (
    DATA_PATH,
    FIELD_ORDER,
)

from training.normalization import (
    FieldWiseNormalizer,
)

from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
)


# ============================================================
# Reuse FROZEN / AUDITED evaluators
# ============================================================

R3_ROLLOUT_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_cross_param_rollout_r3_1.py",
)

PHYSICS_AUDIT_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_m10_physics_audit_v2.py",
)


def load_python_module(
    name,
    path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or
        spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import module from {path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


r3_rollout = load_python_module(
    "r3_1_frozen_rollout",
    R3_ROLLOUT_PATH,
)

physics = load_python_module(
    "m10_physics_audit_v2_defs",
    PHYSICS_AUDIT_PATH,
)


# ============================================================
# R3-1 formal contract
# ============================================================

DISPLAY_M6 = (
    r3_rollout.DISPLAY_M6
)

DISPLAY_NAIVE = (
    r3_rollout.DISPLAY_NAIVE
)

DISPLAY_CANONICAL = (
    r3_rollout.DISPLAY_CANONICAL
)

MODEL_ORDER = (
    DISPLAY_M6,
    DISPLAY_NAIVE,
    DISPLAY_CANONICAL,
)

ACTIVE_TERMS = (
    "buoyancy_advection",
    "buoyancy_forcing",
)


# ============================================================
# Formal physics discretization
#
# IMPORTANT:
# These values define ONLY the offline FD comparative proxy.
# They DO NOT alter the frozen neural operators.
#
# X:
#   periodic 256-point proxy grid on length-4 domain
#
# Y:
#   64 points including y=0 and y=1
#
# Time:
#   nominal frame interval used identically for pred and GT
# ============================================================

PHYSICS_DX = 1.0 / 64.0
PHYSICS_DY = 1.0 / 63.0
PHYSICS_DT = 0.25

FORMAL_HORIZONS = (
    1,
    4,
    8,
    16,
)

FORMAL_TEST_WINDOWS_H16 = 2430

PREDICTION_REPRO_ABS_TOL = 1.0e-7


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "R3-1 matched formal TEST physics evaluation: "
            "M6 vs R3-1a Naive vs R3-1b Canonical. "
            "Uses the frozen R3-1 TEST rollout windows and "
            "the audited M10 Physics-Audit-v2 FD definitions. "
            "No training. FD metrics are comparative proxies, "
            "not solver-level PDE violation."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
    )

    parser.add_argument(
        "--stats",
        required=True,
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
        "--seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--m6_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--naive_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--canonical_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--rollout_summary",
        required=True,
        help=(
            "Frozen formal R3-1 prediction-rollout summary CSV. "
            "Used to verify that this physics evaluator exactly "
            "reproduces the already-frozen prediction trajectories."
        ),
    )

    parser.add_argument(
        "--contract",
        default=(
            "configs/r3/"
            "r3_0_architecture_contract.json"
        ),
    )

    parser.add_argument(
        "--horizons",
        default="1,4,8,16",
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
        help=(
            "DEBUG ONLY. "
            "Formal physics evaluation must omit this."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/tables/"
            "r3_1_physics"
        ),
    )

    return parser.parse_args()


# ============================================================
# General utilities
# ============================================================

def resolve_path(
    path,
):

    if os.path.isabs(
        path
    ):
        return path

    return os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            path,
        )
    )


def sha256_file(
    path,
):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def set_seed(
    seed,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = (
        False
    )

    torch.backends.cudnn.deterministic = (
        True
    )


# ============================================================
# R3-1 free-autoregressive physics rollout
#
# Exactly the same state update as the frozen prediction
# evaluator:
#
#   delta_norm = model(context)
#   next_norm  = current_norm + delta_norm
#   next prediction is fed back into context
#
# GT is used ONLY for offline evaluation.
# ============================================================

@torch.no_grad()
def run_one_model(
    *,
    model_name,
    model,
    x0_phys,
    future_phys,
    param,
    normalizer,
    max_horizon,
    stats,
):

    x_norm = (
        normalizer
        .normalize_x(
            x0_phys
        )
    )

    # Last observed physical frame.
    pred_prev_phys = (
        x0_phys[
            :,
            -len(FIELD_ORDER):,
            :,
            :,
        ]
    )

    # At step 1 the GT previous state is exactly
    # the last observed frame.
    gt_prev_phys = (
        pred_prev_phys
    )

    for step in range(
        1,
        max_horizon + 1,
    ):

        current_norm = (
            x_norm[
                :,
                -len(FIELD_ORDER):,
                :,
                :,
            ]
        )

        if (
            model_name
            ==
            DISPLAY_M6
        ):

            pred_delta_norm = (
                model(
                    x_norm
                )
            )

        else:

            pred_delta_norm = (
                model(
                    x_norm,
                    params=param,
                )
            )

        pred_next_norm = (
            current_norm
            +
            pred_delta_norm
        )

        pred_next_phys = (
            normalizer
            .denormalize_y(
                pred_next_norm
            )
        )

        gt_next_phys = (
            future_phys[
                :,
                step - 1,
                :,
                :,
                :,
            ]
        )

        key = (
            model_name,
            step,
        )

        if key not in stats:
            stats[
                key
            ] = physics.new_bucket()

        physics.update_physics_metrics(
            stats[
                key
            ],
            pred_prev=(
                pred_prev_phys
            ),
            pred_next=(
                pred_next_phys
            ),
            gt_prev=(
                gt_prev_phys
            ),
            gt_next=(
                gt_next_phys
            ),
            param=(
                param
            ),
            dx=(
                PHYSICS_DX
            ),
            dy=(
                PHYSICS_DY
            ),
            dt=(
                PHYSICS_DT
            ),
        )

        # Free autoregressive:
        # own prediction enters next context.
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

        pred_prev_phys = (
            pred_next_phys
        )

        gt_prev_phys = (
            gt_next_phys
        )


# ============================================================
# Physics difference table
# ============================================================

def make_difference_table(
    summary_df,
):

    pairs = [
        (
            DISPLAY_NAIVE,
            DISPLAY_M6,
        ),
        (
            DISPLAY_CANONICAL,
            DISPLAY_M6,
        ),
        (
            DISPLAY_CANONICAL,
            DISPLAY_NAIVE,
        ),
    ]

    rows = []

    for (
        model_a,
        model_b,
    ) in pairs:

        a = summary_df[
            summary_df[
                "model"
            ]
            ==
            model_a
        ]

        b = summary_df[
            summary_df[
                "model"
            ]
            ==
            model_b
        ]

        merged = a.merge(
            b,
            on=[
                "split",
                "seed",
                "horizon",
            ],
            suffixes=(
                "_a",
                "_b",
            ),
        )

        for _, row in (
            merged.iterrows()
        ):

            out = {
                "split":
                    row[
                        "split"
                    ],

                "seed":
                    int(
                        row[
                            "seed"
                        ]
                    ),

                "comparison":
                    (
                        f"{model_a} "
                        f"- {model_b}"
                    ),

                "horizon":
                    int(
                        row[
                            "horizon"
                        ]
                    ),
            }

            for metric in (
                physics.LOWER_IS_BETTER
            ):

                out[
                    f"{metric}_diff"
                ] = (
                    row[
                        f"{metric}_a"
                    ]
                    -
                    row[
                        f"{metric}_b"
                    ]
                )

            rows.append(
                out
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Frozen prediction-rollout reproduction audit
#
# The formal physics evaluator must recover the SAME field/global
# Rel-L2 values as:
#
#   evaluation/evaluate_cross_param_rollout_r3_1.py
#
# because:
#   * same TEST windows
#   * same models
#   * same checkpoints
#   * same free-autoregressive trajectory
#
# Any mismatch means the physics evaluator is NOT evaluating the
# frozen formal prediction trajectory.
# ============================================================

def audit_prediction_reproduction(
    physics_summary,
    rollout_summary_path,
    *,
    split_label,
    seed,
):

    frozen = pd.read_csv(
        rollout_summary_path
    )

    required_columns = {
        "split",
        "seed",
        "model",
        "horizon",
        "field",
        "rel_l2_percent",
    }

    missing = (
        required_columns
        -
        set(
            frozen.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Frozen rollout summary is missing "
            f"columns: {sorted(missing)}"
        )

    frozen = frozen[
        (
            frozen[
                "split"
            ]
            ==
            split_label
        )
        &
        (
            frozen[
                "seed"
            ].astype(int)
            ==
            int(seed)
        )
    ].copy()

    if frozen.empty:
        raise RuntimeError(
            "No matching rows in frozen rollout "
            f"summary for split={split_label}, "
            f"seed={seed}"
        )

    metric_map = {
        "global":
            "global_rel_l2_percent",

        "buoyancy":
            "buoyancy_rel_l2_percent",

        "u_x":
            "u_x_rel_l2_percent",

        "u_y":
            "u_y_rel_l2_percent",

        "pressure":
            "pressure_rel_l2_percent",
    }

    audit_rows = []
    max_abs_diff = 0.0

    for model_name in (
        MODEL_ORDER
    ):

        for horizon in (
            FORMAL_HORIZONS
        ):

            phys_row = (
                physics_summary[
                    (
                        physics_summary[
                            "model"
                        ]
                        ==
                        model_name
                    )
                    &
                    (
                        physics_summary[
                            "horizon"
                        ]
                        ==
                        horizon
                    )
                ]
            )

            if len(
                phys_row
            ) != 1:
                raise RuntimeError(
                    "Expected exactly one physics row "
                    f"for {model_name}, h={horizon}; "
                    f"got {len(phys_row)}"
                )

            phys_row = (
                phys_row.iloc[
                    0
                ]
            )

            for (
                field,
                physics_column,
            ) in metric_map.items():

                frozen_row = frozen[
                    (
                        frozen[
                            "model"
                        ]
                        ==
                        model_name
                    )
                    &
                    (
                        frozen[
                            "horizon"
                        ]
                        ==
                        horizon
                    )
                    &
                    (
                        frozen[
                            "field"
                        ]
                        ==
                        field
                    )
                ]

                if len(
                    frozen_row
                ) != 1:
                    raise RuntimeError(
                        "Expected exactly one frozen "
                        "rollout row for "
                        f"{model_name}, h={horizon}, "
                        f"field={field}; "
                        f"got {len(frozen_row)}"
                    )

                frozen_value = float(
                    frozen_row.iloc[
                        0
                    ][
                        "rel_l2_percent"
                    ]
                )

                physics_value = float(
                    phys_row[
                        physics_column
                    ]
                )

                abs_diff = abs(
                    physics_value
                    -
                    frozen_value
                )

                max_abs_diff = max(
                    max_abs_diff,
                    abs_diff,
                )

                audit_rows.append(
                    {
                        "split":
                            split_label,

                        "seed":
                            int(seed),

                        "model":
                            model_name,

                        "horizon":
                            horizon,

                        "field":
                            field,

                        "frozen_rollout_rel_l2_percent":
                            frozen_value,

                        "physics_eval_rel_l2_percent":
                            physics_value,

                        "abs_diff":
                            abs_diff,
                    }
                )

    audit_df = pd.DataFrame(
        audit_rows
    )

    if (
        max_abs_diff
        >
        PREDICTION_REPRO_ABS_TOL
    ):
        raise RuntimeError(
            "R3-1 physics evaluator does NOT "
            "reproduce the frozen prediction rollout. "
            "max_abs_diff="
            f"{max_abs_diff:.12e}, "
            "tolerance="
            f"{PREDICTION_REPRO_ABS_TOL:.12e}"
        )

    print()
    print(
        "========== FROZEN PREDICTION REPRODUCTION =========="
    )

    print(
        "max_abs Rel-L2 difference = "
        f"{max_abs_diff:.12e}"
    )

    print(
        "✅ Physics evaluator exactly reproduces "
        "the frozen R3-1 prediction trajectory"
    )

    return (
        audit_df,
        max_abs_diff,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    requested_horizons = sorted(
        {
            int(x)
            for x
            in args.horizons.split(
                ","
            )
            if x.strip()
        }
    )

    if (
        requested_horizons
        !=
        list(
            FORMAL_HORIZONS
        )
    ):
        raise ValueError(
            "R3-1 physics protocol is locked to "
            "horizons 1,4,8,16."
        )

    max_horizon = max(
        requested_horizons
    )

    if max_horizon != 16:
        raise RuntimeError(
            "R3-1 physics protocol is locked to H16."
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

    naive_path = resolve_path(
        args.naive_checkpoint
    )

    canonical_path = resolve_path(
        args.canonical_checkpoint
    )

    rollout_summary_path = resolve_path(
        args.rollout_summary
    )

    contract_path = resolve_path(
        args.contract
    )

    data_path = resolve_path(
        DATA_PATH
    )

    required_paths = [
        split_path,
        stats_path,
        m6_path,
        naive_path,
        canonical_path,
        rollout_summary_path,
        contract_path,
        data_path,
        R3_ROLLOUT_PATH,
        PHYSICS_AUDIT_PATH,
    ]

    for path in (
        required_paths
    ):

        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                path
            )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    (
        _,
        alpha_map,
    ) = (
        r3_rollout
        .load_locked_contract(
            contract_path
        )
    )

    contract_sha = (
        sha256_file(
            contract_path
        )
    )

    print(
        "=" * 115
    )

    print(
        "R3-1 MATCHED FORMAL TEST PHYSICS EVALUATION"
    )

    print(
        "=" * 115
    )

    print(
        "Stage: R3-1 physics closeout"
    )

    print(
        "Type: formal evaluation; NO TRAINING"
    )

    print(
        "Models:",
        MODEL_ORDER,
    )

    print(
        "Split:",
        args.split_label,
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Horizons:",
        requested_horizons,
    )

    print(
        "Full curve: h=1..16"
    )

    print(
        "TEST access: YES "
        "(formal post-training evaluation)"
    )

    print(
        "Physics semantics: "
        "FD-based comparative proxy; "
        "NOT solver-level PDE violation"
    )

    print(
        "Formal alpha max by term:",
        alpha_map,
    )

    print()
    print(
        "========== PHYSICS DISCRETIZATION =========="
    )

    print(
        "physics_dx =",
        PHYSICS_DX,
    )

    print(
        "physics_dy =",
        PHYSICS_DY,
    )

    print(
        "physics_dt =",
        PHYSICS_DT,
    )

    print(
        "X derivative: periodic central FD"
    )

    print(
        "Y derivative: interior central FD only"
    )

    print(
        "These settings do NOT alter frozen models."
    )

    if (
        args.max_samples
        is not None
    ):

        print()
        print(
            "⚠️ DEBUG ONLY: max_samples =",
            args.max_samples,
        )

        print(
            "⚠️ Debug physics results are NOT formal evidence."
        )

    print()
    print(
        "========== PROVENANCE =========="
    )

    provenance = {
        "DATA_SHA256":
            sha256_file(
                data_path
            ),

        "CONTRACT_SHA256":
            contract_sha,

        "SPLIT_SHA256":
            sha256_file(
                split_path
            ),

        "STATS_SHA256":
            sha256_file(
                stats_path
            ),

        "M6_SHA256":
            sha256_file(
                m6_path
            ),

        "NAIVE_SHA256":
            sha256_file(
                naive_path
            ),

        "CANONICAL_SHA256":
            sha256_file(
                canonical_path
            ),

        "FROZEN_ROLLOUT_SUMMARY_SHA256":
            sha256_file(
                rollout_summary_path
            ),

        "FROZEN_R3_EVALUATOR_SHA256":
            sha256_file(
                R3_ROLLOUT_PATH
            ),

        "PHYSICS_AUDIT_V2_SHA256":
            sha256_file(
                PHYSICS_AUDIT_PATH
            ),
    }

    for key, value in (
        provenance.items()
    ):
        print(
            f"{key}:",
            value,
        )

    # ========================================================
    # TEST dataset
    #
    # IMPORTANT:
    # Reuse the frozen R3-1 RolloutDataset directly.
    # No physics-specific stride/subsampling.
    # ========================================================

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as f:

        split_config = json.load(
            f
        )

    dataset = (
        r3_rollout
        .RolloutDataset(
            split_config=(
                split_config[
                    "test"
                ]
            ),
            max_horizon=(
                max_horizon
            ),
            max_samples=(
                args.max_samples
            ),
        )
    )

    if (
        args.max_samples
        is None
        and
        len(dataset)
        !=
        FORMAL_TEST_WINDOWS_H16
    ):
        raise RuntimeError(
            "Formal R3-1 physics evaluation must "
            "use exactly the same 2430 H16 TEST windows "
            "as the frozen prediction evaluator; "
            f"got {len(dataset)}."
        )

    print()
    print(
        "========== DATA CONTRACT =========="
    )

    print(
        "TEST_WINDOWS:",
        len(dataset),
    )

    print(
        "Temporal-start stride: 1 "
        "(all legal frozen R3 TEST windows)"
    )

    if (
        args.max_samples
        is None
    ):
        print(
            "✅ Formal TEST window count = 2430"
        )

    loader = DataLoader(
        dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # Models
    # ========================================================

    normalizer = (
        FieldWiseNormalizer(
            stats_path
        )
        .to(device)
    )

    metadata = (
        build_rbc_canonical_metadata(
            stats_path
        )
    )

    (
        m6,
        _,
    ) = (
        r3_rollout
        .build_m6(
            m6_path,
            device,
        )
    )

    (
        naive,
        naive_payload,
    ) = (
        r3_rollout
        .build_r3(
            representation_mode=(
                "naive"
            ),
            checkpoint_path=(
                naive_path
            ),
            metadata=(
                metadata
            ),
            alpha_map=(
                alpha_map
            ),
            contract_sha=(
                contract_sha
            ),
            args=(
                args
            ),
            device=(
                device
            ),
        )
    )

    (
        canonical,
        canonical_payload,
    ) = (
        r3_rollout
        .build_r3(
            representation_mode=(
                "canonical"
            ),
            checkpoint_path=(
                canonical_path
            ),
            metadata=(
                metadata
            ),
            alpha_map=(
                alpha_map
            ),
            contract_sha=(
                contract_sha
            ),
            args=(
                args
            ),
            device=(
                device
            ),
        )
    )

    r3_rollout.assert_same_m6(
        m6,
        naive,
        DISPLAY_NAIVE,
    )

    r3_rollout.assert_same_m6(
        m6,
        canonical,
        DISPLAY_CANONICAL,
    )

    models = {
        DISPLAY_M6:
            m6,

        DISPLAY_NAIVE:
            naive,

        DISPLAY_CANONICAL:
            canonical,
    }

    print()
    print(
        "========== CHECKPOINT MODEL SELECTION =========="
    )

    for (
        model_name,
        payload,
    ) in (
        (
            DISPLAY_NAIVE,
            naive_payload,
        ),
        (
            DISPLAY_CANONICAL,
            canonical_payload,
        ),
    ):

        if isinstance(
            payload,
            dict,
        ):

            print(
                f"{model_name}: "
                "best_val="
                f"{payload.get('best_val_loss', payload.get('val_loss', 'NA'))}, "
                "best_epoch="
                f"{payload.get('best_epoch', payload.get('epoch', 'NA'))}"
            )

    # ========================================================
    # Full free-autoregressive TEST physics rollout
    # ========================================================

    stats = {}

    print()
    print(
        "🔥 Starting R3-1 free-autoregressive "
        "TEST Physics Audit v2..."
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

            x0_phys = (
                x0_phys.to(
                    device,
                    non_blocking=True,
                )
            )

            future_phys = (
                future_phys.to(
                    device,
                    non_blocking=True,
                )
            )

            param = (
                param.to(
                    device,
                    non_blocking=True,
                )
            )

            for model_name in (
                MODEL_ORDER
            ):

                run_one_model(
                    model_name=(
                        model_name
                    ),
                    model=(
                        models[
                            model_name
                        ]
                    ),
                    x0_phys=(
                        x0_phys
                    ),
                    future_phys=(
                        future_phys
                    ),
                    param=(
                        param
                    ),
                    normalizer=(
                        normalizer
                    ),
                    max_horizon=(
                        max_horizon
                    ),
                    stats=(
                        stats
                    ),
                )

            if (
                batch_idx % 10 == 0
                or
                batch_idx == len(
                    loader
                )
            ):

                print(
                    "  processed batch "
                    f"{batch_idx}/"
                    f"{len(loader)}"
                )

    # ========================================================
    # Finalize full H1-H16 physics curve
    # ========================================================

    rows = []

    for model_name in (
        MODEL_ORDER
    ):

        for horizon in range(
            1,
            max_horizon + 1,
        ):

            key = (
                model_name,
                horizon,
            )

            if key not in stats:
                raise RuntimeError(
                    "Missing physics bucket: "
                    f"{key}"
                )

            rows.append(
                physics.finalize_bucket(
                    split_label=(
                        args.split_label
                    ),
                    seed=(
                        args.seed
                    ),
                    model_name=(
                        model_name
                    ),
                    horizon=(
                        horizon
                    ),
                    bucket=(
                        stats[
                            key
                        ]
                    ),
                )
            )

    curve_df = pd.DataFrame(
        rows
    )

    summary_df = curve_df[
        curve_df[
            "horizon"
        ]
        .isin(
            requested_horizons
        )
    ].copy()

    diff_df = (
        make_difference_table(
            summary_df
        )
    )

    # ========================================================
    # Frozen prediction reproduction audit
    #
    # Only meaningful for the formal full-window run.
    # ========================================================

    if (
        args.max_samples
        is None
    ):

        (
            reproduction_df,
            reproduction_max_abs_diff,
        ) = (
            audit_prediction_reproduction(
                summary_df,
                rollout_summary_path,
                split_label=(
                    args.split_label
                ),
                seed=(
                    args.seed
                ),
            )
        )

    else:

        reproduction_df = (
            pd.DataFrame()
        )

        reproduction_max_abs_diff = (
            float("nan")
        )

        print()
        print(
            "⚠️ Frozen prediction reproduction "
            "audit skipped for DEBUG subset."
        )

    # ========================================================
    # Save
    # ========================================================

    output_dir = resolve_path(
        args.output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    prefix = (
        "r3_1_physics_"
        f"{args.split_label}_"
        f"seed{args.seed}"
    )

    paths = {
        "curve":
            os.path.join(
                output_dir,
                (
                    f"{prefix}_"
                    "curve_h1_h16.csv"
                ),
            ),

        "summary":
            os.path.join(
                output_dir,
                (
                    f"{prefix}_"
                    "summary_h1_h4_h8_h16.csv"
                ),
            ),

        "differences":
            os.path.join(
                output_dir,
                (
                    f"{prefix}_"
                    "differences.csv"
                ),
            ),

        "prediction_reproduction":
            os.path.join(
                output_dir,
                (
                    f"{prefix}_"
                    "prediction_reproduction.csv"
                ),
            ),

        "metadata":
            os.path.join(
                output_dir,
                (
                    f"{prefix}_"
                    "metadata.json"
                ),
            ),
    }

    curve_df.to_csv(
        paths[
            "curve"
        ],
        index=False,
    )

    summary_df.to_csv(
        paths[
            "summary"
        ],
        index=False,
    )

    diff_df.to_csv(
        paths[
            "differences"
        ],
        index=False,
    )

    if not reproduction_df.empty:

        reproduction_df.to_csv(
            paths[
                "prediction_reproduction"
            ],
            index=False,
        )

    metadata_out = {
        "experiment":
            "R3-1-matched-formal-test-physics",

        "stage":
            "R3-1-physics-closeout",

        "formal_run":
            (
                args.max_samples
                is None
            ),

        "neural_operator_training":
            False,

        "test_split_accessed":
            True,

        "split_label":
            args.split_label,

        "seed":
            args.seed,

        "models":
            list(
                MODEL_ORDER
            ),

        "formal_test_windows_h16":
            (
                len(dataset)
                if args.max_samples
                is None
                else None
            ),

        "sampling_protocol":
            (
                "same frozen R3-1 TEST rollout windows; "
                "all legal t0 starts; no physics-specific stride"
            ),

        "rollout_protocol":
            (
                "free-autoregressive; model prediction "
                "is fed back into next context"
            ),

        "horizons_reported":
            requested_horizons,

        "full_curve":
            "h=1..16",

        "physics_semantics":
            (
                "FD-based comparative proxy; "
                "not solver-level PDE violation"
            ),

        "physics_dx":
            PHYSICS_DX,

        "physics_dy":
            PHYSICS_DY,

        "physics_dt":
            PHYSICS_DT,

        "x_derivative":
            "periodic second-order central finite difference",

        "y_derivative":
            (
                "non-periodic second-order central finite "
                "difference evaluated only on interior y points"
            ),

        "primary_pde_metric":
            (
                "R_pred^FD - R_GT^FD normalized by "
                "same-equation same-unit GT PDE-term energy"
            ),

        "formal_alpha_max_by_term":
            alpha_map,

        "prediction_reproduction_abs_tol":
            PREDICTION_REPRO_ABS_TOL,

        "prediction_reproduction_max_abs_diff":
            (
                reproduction_max_abs_diff
                if args.max_samples
                is None
                else None
            ),

        "provenance":
            provenance,
    }

    with open(
        paths[
            "metadata"
        ],
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata_out,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Compact terminal output
    # ========================================================

    compact_cols = [
        "model",
        "horizon",

        "global_rel_l2_percent",
        "buoyancy_rel_l2_percent",
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
        "=" * 150
    )

    print(
        "R3-1 FORMAL PHYSICS SUMMARY"
    )

    print(
        "=" * 150
    )

    print(
        summary_df[
            compact_cols
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Main causal comparison:
    # Canonical - Naive
    #
    # Negative = Canonical better.
    # --------------------------------------------------------

    key_diff = diff_df[
        diff_df[
            "comparison"
        ]
        ==
        (
            f"{DISPLAY_CANONICAL} "
            f"- {DISPLAY_NAIVE}"
        )
    ].copy()

    key_diff_cols = [
        "comparison",
        "horizon",

        "global_rel_l2_percent_diff",
        "buoyancy_rel_l2_percent_diff",
        "u_y_rel_l2_percent_diff",

        "adv_b_rel_l2_percent_diff",

        "div_error_mae_diff",
        "vorticity_rel_l2_percent_diff",
        "grad_p_rel_l2_percent_diff",

        "r_b_normalized_mismatch_diff",
        "r_uy_normalized_mismatch_diff",
        "r_u_normalized_mismatch_diff",
    ]

    print()
    print(
        "=" * 150
    )

    print(
        "R3-1b CANONICAL - R3-1a NAIVE "
        "| KEY PHYSICS DIFFERENCES"
    )

    print(
        "Negative = Canonical better "
        "for every displayed metric."
    )

    print(
        "=" * 150
    )

    print(
        key_diff[
            key_diff_cols
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # GT FD-proxy calibration
    # --------------------------------------------------------

    gt_calibration_cols = [
        "horizon",

        "r_b_gt_rms",
        "r_uy_gt_rms",
        "r_u_gt_rms",

        "r_b_gt_residual_to_term_scale",
        "r_uy_gt_residual_to_term_scale",
        "r_u_gt_residual_to_term_scale",
    ]

    gt_calibration = summary_df[
        summary_df[
            "model"
        ]
        ==
        DISPLAY_M6
    ][
        gt_calibration_cols
    ].copy()

    print()
    print(
        "========== GT FD-PROXY CALIBRATION =========="
    )

    print(
        "Non-zero GT FD residual is calibration only; "
        "it is NOT a model error."
    )

    print(
        gt_calibration.to_string(
            index=False
        )
    )

    print()
    print(
        "✅ Saved:"
    )

    for key, path in (
        paths.items()
    ):

        if (
            key
            ==
            "prediction_reproduction"
            and
            reproduction_df.empty
        ):
            continue

        print(
            f"  {key}: {path}"
        )

    print()

    if (
        args.max_samples
        is None
    ):

        print(
            "✅ FORMAL R3-1 TEST PHYSICS EVALUATION COMPLETE"
        )

    else:

        print(
            "✅ DEBUG R3-1 TEST PHYSICS EVALUATION COMPLETE "
            "(NOT FORMAL)"
        )


if __name__ == "__main__":
    main()
