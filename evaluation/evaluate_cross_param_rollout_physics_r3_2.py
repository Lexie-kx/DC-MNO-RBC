from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys

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


from training.normalization import (
    FieldWiseNormalizer,
)

from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
)


# ============================================================
# Reuse FROZEN / AUDITED R3 evaluators
# ============================================================

R3_2_ROLLOUT_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_cross_param_rollout_r3_2.py",
)

R3_1_PHYSICS_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_cross_param_rollout_physics_r3_1.py",
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
    "r3_2_frozen_rollout",
    R3_2_ROLLOUT_PATH,
)

r3_1_physics = load_python_module(
    "r3_1_frozen_physics",
    R3_1_PHYSICS_PATH,
)

# M10 Physics-Audit-v2 definitions already loaded
# and frozen through the R3-1 physics evaluator.
physics = r3_1_physics.physics


# ============================================================
# R3-2 matched model identities
# ============================================================

DISPLAY_CANONICAL = (
    r3_rollout.DISPLAY_CANONICAL
)

DISPLAY_PARAMONLY = (
    r3_rollout.DISPLAY_PARAMONLY
)

DISPLAY_STATEPARAM = (
    r3_rollout.DISPLAY_STATEPARAM
)

MODEL_ORDER = (
    DISPLAY_CANONICAL,
    DISPLAY_PARAMONLY,
    DISPLAY_STATEPARAM,
)


# ============================================================
# Formal physics protocol inherited EXACTLY from R3-1
# ============================================================

PHYSICS_DX = (
    r3_1_physics.PHYSICS_DX
)

PHYSICS_DY = (
    r3_1_physics.PHYSICS_DY
)

PHYSICS_DT = (
    r3_1_physics.PHYSICS_DT
)

FORMAL_HORIZONS = (
    r3_1_physics.FORMAL_HORIZONS
)

FORMAL_TEST_WINDOWS_H16 = (
    r3_1_physics.FORMAL_TEST_WINDOWS_H16
)

PREDICTION_REPRO_ABS_TOL = (
    r3_1_physics.PREDICTION_REPRO_ABS_TOL
)


# ============================================================
# Frozen source provenance
#
# These hashes are fixed BEFORE R3-2 physics TEST.
# ============================================================

EXPECTED_R3_2_ROLLOUT_SHA256 = (
    "888f1205550af0caf582ce50eac2ff96"
    "d0f891f4862d6b083b64e47f9d764949"
)

EXPECTED_R3_1_PHYSICS_SHA256 = (
    "70233daa37a64c1207e57896a7670f46"
    "9d0a04d5d668c637d05f62d57d2de0b0"
)

EXPECTED_R3_2_REGISTRY_SHA256 = (
    "8a67403db5ab10d8035adafddd9d471b"
    "0afa14400e78f6767cbb564fe6fb1efe"
)


# ============================================================
# Predeclared causal interpretation
# ============================================================

PRIMARY_COMPARISON = (
    f"{DISPLAY_STATEPARAM} "
    f"- {DISPLAY_PARAMONLY}"
)

SECONDARY_COMPARISONS = (
    (
        f"{DISPLAY_PARAMONLY} "
        f"- {DISPLAY_CANONICAL}"
    ),
    (
        f"{DISPLAY_STATEPARAM} "
        f"- {DISPLAY_CANONICAL}"
    ),
)

# Declared before any R3-2 physics result is seen.
KEY_PHYSICS_METRICS = (
    "adv_b_rel_l2_percent",
    "r_b_normalized_mismatch",
    "div_error_mae",
    "vorticity_rel_l2_percent",
)


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "R3-2 matched formal TEST physics evaluation: "
            "R3-1b Canonical vs "
            "R3-2a ParamOnly vs "
            "R3-2b StateParam. "
            "Reuses frozen R3-2 prediction rollout windows "
            "and frozen R3-1 / M10 Physics-Audit-v2 FD "
            "definitions. No training. Physics metrics are "
            "FD-based comparative proxies, not solver-level "
            "PDE violation."
        )
    )

    parser.add_argument(
        "--split_label",
        required=True,
        choices=(
            "unseen_pr",
            "unseen_ra",
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
            "DEBUG TEST only. "
            "Formal evaluation must omit this."
        ),
    )

    parser.add_argument(
        "--audit_only",
        action="store_true",
        help=(
            "Audit frozen source files, registry, "
            "rollout result provenance, checkpoints "
            "and model interfaces only. "
            "Does NOT construct the physics TEST dataset."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/tables/"
            "r3_2_physics"
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


def load_json(
    path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def audit_sha(
    path,
    expected,
    label,
):

    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            path
        )

    actual = sha256_file(
        path
    )

    if actual != expected:

        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected={expected}\n"
            f"Actual={actual}\n"
            f"Path={path}"
        )

    return actual


# ============================================================
# Frozen R3-2 prediction-rollout output
# ============================================================

def frozen_rollout_paths(
    split_label,
    seed,
):

    base = resolve_path(
        os.path.join(
            "outputs",
            "tables",
            "r3_2_rollout",
        )
    )

    prefix = (
        f"r3_2_rollout_"
        f"{split_label}_"
        f"seed{seed}"
    )

    return {
        "summary":
            os.path.join(
                base,
                f"{prefix}_summary.csv",
            ),

        "metadata":
            os.path.join(
                base,
                f"{prefix}_metadata.json",
            ),
    }


def audit_frozen_rollout_output(
    *,
    split_label,
    seed,
    registry_sha,
    entries,
):

    paths = frozen_rollout_paths(
        split_label,
        seed,
    )

    for path in (
        paths.values()
    ):

        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                path
            )

    metadata = load_json(
        paths[
            "metadata"
        ]
    )

    checks = {
        "experiment":
            "R3-2-matched-formal-test-rollout",

        "formal_run":
            True,

        "split_label":
            split_label,

        "seed":
            seed,

        "checkpoint_set_frozen_before_test":
            True,

        "test_accessed":
            True,

        "r3_2_registry_sha256":
            registry_sha,

        "r3_2a_checkpoint_sha256":
            entries[
                "R3-2a"
            ][
                "sha256"
            ],

        "r3_2b_checkpoint_sha256":
            entries[
                "R3-2b"
            ][
                "sha256"
            ],
    }

    for key, expected in (
        checks.items()
    ):

        actual = metadata.get(
            key
        )

        if actual != expected:

            raise RuntimeError(
                "Frozen R3-2 rollout metadata "
                f"mismatch.\n"
                f"key={key}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            )

    if (
        metadata.get(
            "formal_test_windows"
        )
        !=
        FORMAL_TEST_WINDOWS_H16
    ):
        raise RuntimeError(
            "Frozen rollout formal TEST "
            "window count is not 2430."
        )

    if (
        metadata.get(
            "primary_comparison"
        )
        !=
        PRIMARY_COMPARISON
    ):
        raise RuntimeError(
            "Frozen rollout primary "
            "comparison changed."
        )

    return (
        paths,
        metadata,
    )


# ============================================================
# Physics difference table
# ============================================================

def make_difference_table(
    summary_df,
):

    pairs = [
        # Secondary.
        (
            DISPLAY_PARAMONLY,
            DISPLAY_CANONICAL,
        ),

        # Secondary.
        (
            DISPLAY_STATEPARAM,
            DISPLAY_CANONICAL,
        ),

        # PRIMARY.
        (
            DISPLAY_STATEPARAM,
            DISPLAY_PARAMONLY,
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
# Same concept as R3-1:
# physics evaluator must recover the SAME prediction Rel-L2
# values as the already-completed frozen R3-2 rollout.
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
            "Frozen R3-2 rollout summary "
            "is missing columns: "
            f"{sorted(missing)}"
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
            "No matching rows in frozen "
            "R3-2 rollout summary for "
            f"split={split_label}, seed={seed}"
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
                    "Expected exactly one "
                    "physics row for "
                    f"{model_name}, h={horizon}; "
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
                        "Expected exactly one "
                        "frozen R3-2 rollout row "
                        f"for {model_name}, "
                        f"h={horizon}, "
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
            "R3-2 physics evaluator does NOT "
            "reproduce the frozen R3-2 "
            "prediction rollout. "
            "max_abs_diff="
            f"{max_abs_diff:.12e}, "
            "tolerance="
            f"{PREDICTION_REPRO_ABS_TOL:.12e}"
        )

    print()
    print(
        "========== FROZEN R3-2 "
        "PREDICTION REPRODUCTION =========="
    )

    print(
        "max_abs Rel-L2 difference = "
        f"{max_abs_diff:.12e}"
    )

    print(
        "✅ Physics evaluator exactly "
        "reproduces the frozen R3-2 "
        "prediction trajectory"
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

    # --------------------------------------------------------
    # Formal protocol locks
    # --------------------------------------------------------

    if args.seed != 42:

        raise ValueError(
            "Formal R3-2 physics "
            "evaluation is locked "
            "to seed=42."
        )

    if args.batch_size != 4:

        raise ValueError(
            "Formal R3-2 physics "
            "evaluation is locked "
            "to batch_size=4."
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
            "R3-2 physics protocol "
            "is locked to horizons "
            "1,4,8,16."
        )

    max_horizon = max(
        requested_horizons
    )

    if max_horizon != 16:

        raise RuntimeError(
            "R3-2 physics protocol "
            "is locked to H16."
        )

    # --------------------------------------------------------
    # Frozen source files
    # --------------------------------------------------------

    r3_2_rollout_sha = (
        audit_sha(
            R3_2_ROLLOUT_PATH,
            EXPECTED_R3_2_ROLLOUT_SHA256,
            "Frozen R3-2 rollout evaluator",
        )
    )

    r3_1_physics_sha = (
        audit_sha(
            R3_1_PHYSICS_PATH,
            EXPECTED_R3_1_PHYSICS_SHA256,
            "Frozen R3-1 physics evaluator",
        )
    )

    # --------------------------------------------------------
    # Frozen split resources
    # --------------------------------------------------------

    resource = (
        r3_rollout
        .LOCKED_RESOURCES[
            args.split_label
        ]
    )

    split_path = resolve_path(
        resource[
            "split"
        ]
    )

    stats_path = resolve_path(
        resource[
            "stats"
        ]
    )

    m6_path = resolve_path(
        resource[
            "m6"
        ]
    )

    r3_1b_path = resolve_path(
        resource[
            "r3_1b"
        ]
    )

    r3_rollout.audit_locked_file(
        split_path,
        resource[
            "split_sha256"
        ],
        (
            f"{args.split_label} split"
        ),
    )

    r3_rollout.audit_locked_file(
        stats_path,
        resource[
            "stats_sha256"
        ],
        (
            f"{args.split_label} stats"
        ),
    )

    r3_rollout.audit_locked_file(
        m6_path,
        resource[
            "m6_sha256"
        ],
        (
            f"{args.split_label} M6"
        ),
    )

    # --------------------------------------------------------
    # Frozen R3-2 registry/checkpoints
    # --------------------------------------------------------

    (
        registry,
        entries,
        registry_path,
        r3_2_contract_path,
        r3_2_contract_sha,
    ) = (
        r3_rollout
        .load_and_audit_registry(
            args.split_label
        )
    )

    registry_sha = (
        audit_sha(
            registry_path,
            EXPECTED_R3_2_REGISTRY_SHA256,
            "Frozen R3-2 pre-TEST registry",
        )
    )

    parent_sha = (
        entries[
            "R3-2a"
        ][
            "parent_r3_1b_sha256"
        ]
    )

    if (
        entries[
            "R3-2b"
        ][
            "parent_r3_1b_sha256"
        ]
        !=
        parent_sha
    ):

        raise RuntimeError(
            "R3-2a and R3-2b do not "
            "share the same frozen "
            "R3-1b parent."
        )

    r3_rollout.audit_locked_file(
        r3_1b_path,
        parent_sha,
        (
            f"{args.split_label} "
            "R3-1b parent"
        ),
    )

    paramonly_path = resolve_path(
        entries[
            "R3-2a"
        ][
            "checkpoint"
        ]
    )

    stateparam_path = resolve_path(
        entries[
            "R3-2b"
        ][
            "checkpoint"
        ]
    )

    paramonly_sha = sha256_file(
        paramonly_path
    )

    stateparam_sha = sha256_file(
        stateparam_path
    )

    # --------------------------------------------------------
    # Frozen completed R3-2 rollout result
    # --------------------------------------------------------

    (
        rollout_paths,
        rollout_metadata,
    ) = (
        audit_frozen_rollout_output(
            split_label=(
                args.split_label
            ),
            seed=(
                args.seed
            ),
            registry_sha=(
                registry_sha
            ),
            entries=(
                entries
            ),
        )
    )

    rollout_summary_path = (
        rollout_paths[
            "summary"
        ]
    )

    rollout_summary_sha = (
        sha256_file(
            rollout_summary_path
        )
    )

    rollout_metadata_sha = (
        sha256_file(
            rollout_paths[
                "metadata"
            ]
        )
    )

    # --------------------------------------------------------
    # Formal alpha capacities
    # --------------------------------------------------------

    r3_0_contract_path = resolve_path(
        r3_rollout.R3_0_CONTRACT
    )

    (
        _,
        alpha_map,
    ) = (
        r3_rollout
        .load_locked_contract(
            r3_0_contract_path
        )
    )

    r3_0_contract_sha = (
        sha256_file(
            r3_0_contract_path
        )
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        "=" * 128
    )

    print(
        "R3-2 MATCHED FORMAL TEST "
        "PHYSICS EVALUATION"
    )

    print(
        "=" * 128
    )

    print(
        "Stage: R3-2 physics closeout"
    )

    print(
        "Type: formal evaluation; "
        "NO TRAINING"
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
        "Primary contrast:",
        PRIMARY_COMPARISON,
    )

    print(
        "Negative primary difference = "
        "StateParam better"
    )

    print(
        "Predeclared key physics metrics:",
        KEY_PHYSICS_METRICS,
    )

    print(
        "Horizons:",
        requested_horizons,
    )

    print(
        "Full physics curve: h=1..16"
    )

    print(
        "Physics semantics: "
        "FD-based comparative proxy; "
        "NOT solver-level PDE violation"
    )

    print(
        "Audit only:",
        args.audit_only,
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
        "These settings do NOT alter "
        "the frozen models."
    )

    print()
    print(
        "========== FROZEN PROVENANCE =========="
    )

    provenance = {
        "R3_0_CONTRACT_SHA256":
            r3_0_contract_sha,

        "R3_2_CONTRACT_SHA256":
            r3_2_contract_sha,

        "R3_2_REGISTRY_SHA256":
            registry_sha,

        "R3_2_ROLLOUT_EVALUATOR_SHA256":
            r3_2_rollout_sha,

        "R3_1_PHYSICS_EVALUATOR_SHA256":
            r3_1_physics_sha,

        "SPLIT_SHA256":
            resource[
                "split_sha256"
            ],

        "STATS_SHA256":
            resource[
                "stats_sha256"
            ],

        "M6_SHA256":
            resource[
                "m6_sha256"
            ],

        "R3_1B_SHA256":
            parent_sha,

        "R3_2A_SHA256":
            paramonly_sha,

        "R3_2B_SHA256":
            stateparam_sha,

        "FROZEN_R3_2_ROLLOUT_SUMMARY_SHA256":
            rollout_summary_sha,

        "FROZEN_R3_2_ROLLOUT_METADATA_SHA256":
            rollout_metadata_sha,

        "PHYSICS_AUDIT_V2_SHA256":
            sha256_file(
                r3_1_physics
                .PHYSICS_AUDIT_PATH
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
    # Build frozen models BEFORE physics TEST dataset
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

    class R3Args:
        pass

    r3_args = R3Args()

    r3_args.split_label = (
        args.split_label
    )

    r3_args.seed = (
        args.seed
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
                r3_1b_path
            ),
            metadata=(
                metadata
            ),
            alpha_map=(
                alpha_map
            ),
            contract_sha=(
                r3_0_contract_sha
            ),
            args=(
                r3_args
            ),
            device=(
                device
            ),
        )
    )

    (
        paramonly,
        paramonly_payload,
    ) = (
        r3_rollout
        .build_r3_2(
            adaptation_mode=(
                "paramonly"
            ),
            checkpoint_path=(
                paramonly_path
            ),
            expected_entry=(
                entries[
                    "R3-2a"
                ]
            ),
            metadata=(
                metadata
            ),
            alpha_map=(
                alpha_map
            ),
            r3_2_contract_sha=(
                r3_2_contract_sha
            ),
            split_label=(
                args.split_label
            ),
            device=(
                device
            ),
        )
    )

    (
        stateparam,
        stateparam_payload,
    ) = (
        r3_rollout
        .build_r3_2(
            adaptation_mode=(
                "stateparam"
            ),
            checkpoint_path=(
                stateparam_path
            ),
            expected_entry=(
                entries[
                    "R3-2b"
                ]
            ),
            metadata=(
                metadata
            ),
            alpha_map=(
                alpha_map
            ),
            r3_2_contract_sha=(
                r3_2_contract_sha
            ),
            split_label=(
                args.split_label
            ),
            device=(
                device
            ),
        )
    )

    # --------------------------------------------------------
    # Exact embedded M6 audits
    # --------------------------------------------------------

    r3_rollout.assert_same_m6(
        m6,
        canonical,
        DISPLAY_CANONICAL,
    )

    r3_rollout.assert_same_m6(
        m6,
        paramonly,
        DISPLAY_PARAMONLY,
    )

    r3_rollout.assert_same_m6(
        m6,
        stateparam,
        DISPLAY_STATEPARAM,
    )

    # --------------------------------------------------------
    # Exact frozen parent raw-alpha audits
    # --------------------------------------------------------

    r3_rollout.assert_same_parent_raw_alpha(
        canonical,
        paramonly,
        DISPLAY_PARAMONLY,
    )

    r3_rollout.assert_same_parent_raw_alpha(
        canonical,
        stateparam,
        DISPLAY_STATEPARAM,
    )

    # --------------------------------------------------------
    # Same R3-2 model interface audit
    # --------------------------------------------------------

    r3_rollout.synthetic_interface_audit(
        paramonly=(
            paramonly
        ),
        stateparam=(
            stateparam
        ),
        alpha_map=(
            alpha_map
        ),
        device=(
            device
        ),
    )

    print()
    print(
        "========== FROZEN CHECKPOINT SELECTION =========="
    )

    for (
        model_name,
        payload,
    ) in (
        (
            DISPLAY_CANONICAL,
            canonical_payload,
        ),
        (
            DISPLAY_PARAMONLY,
            paramonly_payload,
        ),
        (
            DISPLAY_STATEPARAM,
            stateparam_payload,
        ),
    ):

        print(
            f"{model_name}: "
            "best_val="
            f"{payload.get('best_val_loss', payload.get('val_loss'))} "
            "best_epoch="
            f"{payload.get('best_epoch', payload.get('epoch'))}"
        )

    # --------------------------------------------------------
    # AUDIT-ONLY STOP.
    #
    # No physics TEST dataset constructed.
    # --------------------------------------------------------

    if args.audit_only:

        print()
        print(
            "✅ R3-2 PHYSICS "
            "EVALUATOR AUDIT-ONLY PASS"
        )

        print(
            "✅ Frozen source files / "
            "rollout outputs / registry / "
            "checkpoints / interfaces verified"
        )

        print(
            "✅ Physics TEST dataset "
            "was NOT constructed"
        )

        return

    # ========================================================
    # Formal physics TEST access starts here
    # ========================================================

    print()
    print(
        "TEST access: YES "
        "(formal post-training physics evaluation)"
    )

    if (
        args.max_samples
        is not None
    ):

        print(
            "⚠️ DEBUG ONLY: max_samples =",
            args.max_samples,
        )

        print(
            "⚠️ Debug physics results "
            "are NOT formal evidence."
        )

    split_config = load_json(
        split_path
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
            "Formal R3-2 physics evaluation "
            "must use exactly the same "
            "2430 H16 TEST windows as "
            "the frozen R3-2 prediction rollout; "
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
        "(all legal frozen R3-2 TEST windows)"
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
    # Full free-autoregressive physics rollout
    # ========================================================

    models = {
        DISPLAY_CANONICAL:
            canonical,

        DISPLAY_PARAMONLY:
            paramonly,

        DISPLAY_STATEPARAM:
            stateparam,
    }

    stats = {}

    print()
    print(
        "🔥 Starting R3-2 "
        "free-autoregressive TEST "
        "Physics Audit v2..."
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

                # Reuse the frozen R3-1 physics
                # free-autoregressive trajectory
                # and FD metric accumulation exactly.
                r3_1_physics.run_one_model(
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
                batch_idx
                ==
                len(loader)
            ):

                print(
                    "  processed batch "
                    f"{batch_idx}/"
                    f"{len(loader)}"
                )

    # ========================================================
    # Finalize full H1-H16 curve
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
            "⚠️ Frozen prediction "
            "reproduction audit skipped "
            "for DEBUG subset."
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
        "r3_2_physics_"
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
            "R3-2-matched-formal-test-physics",

        "stage":
            "R3-2-physics-closeout",

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

        "primary_comparison":
            PRIMARY_COMPARISON,

        "primary_interpretation":
            (
                "incremental physics behavior "
                "of current-state information "
                "beyond the matched "
                "parameter-only conditioner"
            ),

        "negative_primary_difference_means":
            "StateParam better",

        "secondary_comparisons":
            list(
                SECONDARY_COMPARISONS
            ),

        "key_physics_metrics_predeclared_before_test":
            list(
                KEY_PHYSICS_METRICS
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
                "same frozen R3-2 TEST "
                "rollout windows; "
                "all legal t0 starts; "
                "no physics-specific stride"
            ),

        "rollout_protocol":
            (
                "free-autoregressive; "
                "model prediction is fed "
                "back into next context"
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
            (
                "periodic second-order "
                "central finite difference"
            ),

        "y_derivative":
            (
                "non-periodic second-order "
                "central finite difference "
                "evaluated only on "
                "interior y points"
            ),

        "primary_pde_metric":
            (
                "R_pred^FD - R_GT^FD "
                "normalized by same-equation "
                "same-unit GT PDE-term energy"
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

        "frozen_r3_2_rollout_summary":
            rollout_summary_path,

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
        "=" * 160
    )

    print(
        "R3-2 FORMAL PHYSICS SUMMARY"
    )

    print(
        "=" * 160
    )

    print(
        summary_df[
            compact_cols
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # PRIMARY:
    # StateParam - ParamOnly
    #
    # Negative = StateParam better.
    # --------------------------------------------------------

    key_diff = diff_df[
        diff_df[
            "comparison"
        ]
        ==
        PRIMARY_COMPARISON
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
        "=" * 160
    )

    print(
        "R3-2b STATEPARAM "
        "- R3-2a PARAMONLY "
        "| PRIMARY PHYSICS DIFFERENCES"
    )

    print(
        "Negative = StateParam better "
        "for every displayed metric."
    )

    print(
        "=" * 160
    )

    print(
        key_diff[
            key_diff_cols
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # GT FD proxy calibration
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
        DISPLAY_CANONICAL
    ][
        gt_calibration_cols
    ].copy()

    print()
    print(
        "========== GT FD-PROXY CALIBRATION =========="
    )

    print(
        "Non-zero GT FD residual is "
        "calibration only; "
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
            "✅ FORMAL R3-2 TEST "
            "PHYSICS EVALUATION COMPLETE"
        )

    else:

        print(
            "✅ DEBUG R3-2 TEST "
            "PHYSICS EVALUATION COMPLETE "
            "(NOT FORMAL)"
        )


if __name__ == "__main__":
    main()
