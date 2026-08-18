import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
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


# ============================================================
# Reuse CLOSED D2-2c implementation.
#
# D2-2d changes only the rollout horizon:
#
#   H4  -> H16
#
# It does NOT change:
#   - D2-1 checkpoint
#   - canonical dx/dy/dt
#   - RMSCap
#   - utility rows
#   - Legacy10 features
#   - TemporalAlpha20 features
#   - classifier training protocol
#   - classifier epochs
#   - step1 trust
#   - soft Gate semantics
#   - thresholding policy
# ============================================================

D2_2C_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_d2_2c_canonical_utility_gate_val_h4.py",
)

if not os.path.exists(
    D2_2C_PATH
):
    raise FileNotFoundError(
        D2_2C_PATH
    )


spec = importlib.util.spec_from_file_location(
    "closed_d2_2c_h4",
    D2_2C_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Cannot import {D2_2C_PATH}"
    )

d2c = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    d2c
)


HORIZONS = [
    1,
    4,
    8,
    16,
]

EXPECTED_FORMAL_H16_WINDOWS = 486


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "D2-2d canonical Utility-Gate "
            "long-rollout validation, H16."
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
        "--m6_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--d2_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--train_utility_csv",
        required=True,
    )

    parser.add_argument(
        "--val_utility_csv",
        required=True,
    )

    parser.add_argument(
        "--d2_2a_metadata",
        required=True,
    )

    parser.add_argument(
        "--d2_2b_metadata",
        required=True,
    )

    parser.add_argument(
        "--d2_2b_classification_csv",
        required=True,
    )

    parser.add_argument(
        "--d2_2c_metadata",
        required=True,
    )

    parser.add_argument(
        "--d2_2c_summary_csv",
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
        type=int,
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--classifier_epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help=(
            "Smoke only. Formal run leaves "
            "this unset."
        ),
    )

    parser.add_argument(
        "--run_name",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/tables/"
            "d2_2d_canonical_gate_h16"
        ),
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def resolve_output_dir(path):

    if os.path.isabs(path):
        return path

    return os.path.join(
        PROJECT_ROOT,
        path,
    )


def captured_call(
    function,
    *args,
    **kwargs,
):

    buffer = io.StringIO()

    try:

        with contextlib.redirect_stdout(
            buffer
        ):

            result = function(
                *args,
                **kwargs,
            )

    except Exception:

        print(
            buffer.getvalue()
        )

        raise

    return (
        result,
        buffer.getvalue(),
    )


def rel_value(
    summary,
    model,
    horizon,
    field,
):

    row = summary[
        (
            summary["model"]
            ==
            model
        )
        &
        (
            summary["horizon"]
            ==
            horizon
        )
        &
        (
            summary["field"]
            ==
            field
        )
    ]

    if len(row) != 1:

        raise RuntimeError(
            "Cannot uniquely locate metric: "
            f"model={model}, "
            f"h={horizon}, "
            f"field={field}"
        )

    return float(
        row.iloc[0][
            "rel_l2_percent"
        ]
    )


# ============================================================
# D2-2c closure lock
# ============================================================

def audit_d2_2c_closure(
    args,
    provenance,
):

    for path in [
        args.d2_2c_metadata,
        args.d2_2c_summary_csv,
    ]:

        if not os.path.exists(path):
            raise FileNotFoundError(path)

    with open(
        args.d2_2c_metadata,
        "r",
        encoding="utf-8",
    ) as handle:

        metadata = json.load(
            handle
        )

    if (
        metadata.get("experiment")
        !=
        "D2-2c Canonical Utility-Gate Closed-Loop VAL H4"
    ):
        raise RuntimeError(
            "Unexpected D2-2c metadata."
        )

    if (
        metadata.get("stage")
        !=
        "formal_gate_control_h4"
    ):
        raise RuntimeError(
            "D2-2c is not a formal H4 run."
        )

    if (
        metadata.get("split_label")
        !=
        args.split_label
    ):
        raise RuntimeError(
            "D2-2c split mismatch."
        )

    if (
        int(
            metadata.get(
                "seed",
                -1,
            )
        )
        !=
        args.seed
    ):
        raise RuntimeError(
            "D2-2c seed mismatch."
        )

    if (
        metadata.get(
            "test_split_accessed"
        )
        is not False
    ):
        raise RuntimeError(
            "D2-2c TEST contract failed."
        )

    if (
        metadata.get(
            "classifier_primary"
        )
        !=
        "Legacy10"
    ):
        raise RuntimeError(
            "Legacy10 is not D2-2c primary."
        )

    if (
        int(
            metadata.get(
                "max_horizon",
                -1,
            )
        )
        !=
        4
    ):
        raise RuntimeError(
            "D2-2c max_horizon mismatch."
        )

    if (
        metadata.get(
            "max_val_samples"
        )
        is not None
    ):
        raise RuntimeError(
            "D2-2c metadata refers to "
            "a smoke run, not formal H4."
        )

    if (
        int(
            metadata.get(
                "formal_h4_windows",
                -1,
            )
        )
        !=
        558
    ):
        raise RuntimeError(
            "D2-2c formal H4 window "
            "count mismatch."
        )

    provenance_pairs = [
        (
            "split_sha256",
            provenance["split_sha"],
        ),
        (
            "stats_sha256",
            provenance["stats_sha"],
        ),
        (
            "m6_sha256",
            provenance["m6_sha"],
        ),
        (
            "d2_checkpoint_sha256",
            provenance["d2_sha"],
        ),
        (
            "train_utility_csv_sha256",
            provenance["train_sha"],
        ),
        (
            "val_utility_csv_sha256",
            provenance["val_sha"],
        ),
        (
            "d2_2a_metadata_sha256",
            provenance["d2a_meta_sha"],
        ),
        (
            "d2_2b_metadata_sha256",
            provenance["d2b_meta_sha"],
        ),
        (
            "d2_2b_classification_sha256",
            provenance[
                "classification_sha"
            ],
        ),
    ]

    for key, expected in provenance_pairs:

        if (
            metadata.get(key)
            !=
            expected
        ):

            raise RuntimeError(
                "D2-2c provenance mismatch: "
                f"{key}"
            )

    summary_sha = sha256_file(
        args.d2_2c_summary_csv
    )

    summary_name = os.path.basename(
        args.d2_2c_summary_csv
    )

    if (
        metadata.get(
            "outputs",
            {},
        ).get(
            summary_name
        )
        !=
        summary_sha
    ):

        raise RuntimeError(
            "D2-2c summary SHA mismatch."
        )

    summary = pd.read_csv(
        args.d2_2c_summary_csv
    )

    # --------------------------------------------------------
    # D2-2c was declared STRONG PASS:
    #
    # 1. Legacy10 better than B-only at h3/h4
    #    on buoyancy + global.
    #
    # 2. Legacy10 better than or equal to M6
    #    at h4 on buoyancy + global.
    # --------------------------------------------------------

    for horizon in [
        3,
        4,
    ]:

        for field in [
            "buoyancy",
            "global",
        ]:

            legacy = rel_value(
                summary,
                "UtilityGate-Legacy10",
                horizon,
                field,
            )

            bonly = rel_value(
                summary,
                "B-only",
                horizon,
                field,
            )

            if not (
                legacy < bonly
            ):

                raise RuntimeError(
                    "D2-2c STRONG-PASS lock "
                    "failed: "
                    f"h={horizon}, "
                    f"field={field}, "
                    f"Legacy={legacy}, "
                    f"B-only={bonly}"
                )

    for field in [
        "buoyancy",
        "global",
    ]:

        legacy = rel_value(
            summary,
            "UtilityGate-Legacy10",
            4,
            field,
        )

        m6 = rel_value(
            summary,
            "M6",
            4,
            field,
        )

        if not (
            legacy <= m6
        ):

            raise RuntimeError(
                "D2-2c strong-pass M6 "
                "lock failed: "
                f"field={field}, "
                f"Legacy={legacy}, "
                f"M6={m6}"
            )

    return {
        "metadata":
            metadata,

        "metadata_sha":
            sha256_file(
                args.d2_2c_metadata
            ),

        "summary_sha":
            summary_sha,
    }


# ============================================================
# Compact tables
# ============================================================

def make_compact_metrics(
    summary,
):

    rows = []

    for field in [
        "buoyancy",
        "global",
    ]:

        for horizon in HORIZONS:

            m6 = rel_value(
                summary,
                "M6",
                horizon,
                field,
            )

            bonly = rel_value(
                summary,
                "B-only",
                horizon,
                field,
            )

            legacy = rel_value(
                summary,
                "UtilityGate-Legacy10",
                horizon,
                field,
            )

            temporal = rel_value(
                summary,
                "UtilityGate-TemporalAlpha20",
                horizon,
                field,
            )

            rows.append(
                {
                    "field":
                        field,

                    "horizon":
                        horizon,

                    "M6":
                        m6,

                    "B_only":
                        bonly,

                    "Legacy10":
                        legacy,

                    "TemporalAlpha20":
                        temporal,

                    "Legacy_minus_Bonly_pp":
                        legacy
                        -
                        bonly,

                    "Legacy_minus_M6_pp":
                        legacy
                        -
                        m6,
                }
            )

    return pd.DataFrame(
        rows
    )


def make_compact_gate(
    gate_df,
):

    rows = []

    for horizon in HORIZONS:

        row = {
            "horizon":
                horizon,
        }

        for model, label in [
            (
                "UtilityGate-Legacy10",
                "Legacy10_q_mean",
            ),
            (
                "UtilityGate-TemporalAlpha20",
                "Temporal20_q_mean",
            ),
        ]:

            sub = gate_df[
                (
                    gate_df["model"]
                    ==
                    model
                )
                &
                (
                    gate_df["horizon"]
                    ==
                    horizon
                )
            ]

            if len(sub) != 1:
                raise RuntimeError(
                    "Gate-stat lookup failed: "
                    f"{model}, h={horizon}"
                )

            row[label] = float(
                sub.iloc[0][
                    "q_mean"
                ]
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if (
        args.classifier_epochs
        !=
        100
    ):
        raise ValueError(
            "D2-2d classifier protocol "
            "is locked to 100 epochs."
        )

    if (
        args.max_val_samples
        is not None
        and
        args.max_val_samples <= 0
    ):
        raise ValueError(
            "max_val_samples must be positive."
        )

    d2c.set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    lock = d2c.LOCKED[
        args.split_label
    ]

    output_dir = resolve_output_dir(
        args.output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    prefix = os.path.join(
        output_dir,
        args.run_name,
    )

    audit_trace = []

    print()
    print(
        "=" * 90
    )

    print(
        "D2-2d CANONICAL UTILITY-GATE "
        "VAL H16"
    )

    print(
        "=" * 90
    )

    print(
        f"split={args.split_label} | "
        f"seed={args.seed} | "
        f"device={device}"
    )

    print(
        "D2-1 frozen | "
        "Legacy10 primary | "
        "Temporal20 ablation"
    )

    print(
        "step1 q=1 | "
        "step2-16 frozen soft q | "
        "NO H16 refit | TEST forbidden"
    )

    # --------------------------------------------------------
    # Full D2-2a / 2b provenance audit.
    # Suppressed from terminal, saved in audit trace.
    # --------------------------------------------------------

    (
        provenance,
        trace,
    ) = captured_call(
        d2c.audit_provenance,
        args,
        lock,
    )

    audit_trace.append(
        "========== D2-2a/2b PROVENANCE ==========\n"
        +
        trace
    )

    print(
        "✅ D2-2a / D2-2b provenance PASS"
    )

    # --------------------------------------------------------
    # Formal D2-2c H4 closure lock.
    # --------------------------------------------------------

    h4_lock = audit_d2_2c_closure(
        args,
        provenance,
    )

    print(
        "✅ D2-2c formal H4 STRONG-PASS lock PASS"
    )

    # --------------------------------------------------------
    # Rebuild exactly the same TRAIN-only classifiers.
    # --------------------------------------------------------

    (
        classifier_result,
        trace,
    ) = captured_call(
        d2c.fit_and_reproduce_classifiers,
        args,
        provenance,
    )

    (
        legacy_predictor,
        temporal_predictor,
        predictor_sanity,
    ) = classifier_result

    audit_trace.append(
        "\n========== CLASSIFIER REPRODUCTION ==========\n"
        +
        trace
    )

    legacy_auc = float(
        predictor_sanity[
            predictor_sanity[
                "predictor"
            ]
            ==
            "Legacy10"
        ].iloc[0][
            "val_auc"
        ]
    )

    temporal_auc = float(
        predictor_sanity[
            predictor_sanity[
                "predictor"
            ]
            ==
            "TemporalAlpha20"
        ].iloc[0][
            "val_auc"
        ]
    )

    print(
        "✅ Classifier reproduction PASS | "
        f"Legacy10 AUC={legacy_auc:.6f} | "
        f"Temporal20 AUC={temporal_auc:.6f}"
    )

    # --------------------------------------------------------
    # H16 VAL dataset.
    # --------------------------------------------------------

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as handle:

        split_config = json.load(
            handle
        )

    dataset = (
        d2c.closed.audit.base.RolloutDataset(
            split_config=(
                split_config["val"]
            ),
            max_horizon=16,
            max_samples=(
                args.max_val_samples
            ),
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if (
        args.max_val_samples
        is None
    ):

        if (
            len(dataset)
            !=
            EXPECTED_FORMAL_H16_WINDOWS
        ):

            raise RuntimeError(
                "Formal H16 expected "
                f"{EXPECTED_FORMAL_H16_WINDOWS} "
                f"VAL windows, got "
                f"{len(dataset)}"
            )

    else:

        if (
            len(dataset)
            !=
            args.max_val_samples
        ):

            raise RuntimeError(
                "Smoke H16 sample-count mismatch."
            )

    print(
        f"✅ VAL H16 windows: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Models.
    # --------------------------------------------------------

    normalizer = (
        d2c.closed.audit.base.FieldWiseNormalizer(
            args.stats
        ).to(device)
    )

    (
        field_mean,
        field_std,
    ) = (
        d2c.closed.audit.base.build_field_stats(
            args.stats
        )
    )

    m6, _ = (
        d2c.closed.audit.base.build_m6(
            args.m6_checkpoint,
            device,
        )
    )

    (
        d2_result,
        trace,
    ) = captured_call(
        d2c.build_d2,
        args.d2_checkpoint,
        field_mean,
        field_std,
        lock,
        device,
    )

    d2, d2_payload = d2_result

    audit_trace.append(
        "\n========== D2-1 MODEL CONTRACT ==========\n"
        +
        trace
    )

    (
        _,
        trace,
    ) = captured_call(
        d2c.closed.audit.base.assert_same_m6,
        m6,
        d2,
        "D2-1-DimConsistent-BOnly",
    )

    audit_trace.append(
        "\n========== EMBEDDED M6 STATIC CHECK ==========\n"
        +
        trace
    )

    print(
        "✅ D2-1 canonical model / embedded M6 PASS"
    )

    # --------------------------------------------------------
    # H16 closed loop.
    # This function is generic in max_horizon.
    # --------------------------------------------------------

    print(
        "🔥 H16 rollout..."
    )

    (
        error_stats,
        gate_stats,
        embedded_m6_max_abs,
    ) = (
        d2c.closed.evaluate_closed_loop(
            loader=loader,
            normalizer=normalizer,
            m6=m6,
            m10=d2,
            legacy_predictor=(
                legacy_predictor
            ),
            temporal_predictor=(
                temporal_predictor
            ),
            device=device,
            max_horizon=16,
        )
    )

    if (
        embedded_m6_max_abs
        >
        1.0e-6
    ):

        raise RuntimeError(
            "Embedded M6 mismatch during H16: "
            f"{embedded_m6_max_abs}"
        )

    print(
        "✅ Embedded M6 H16 max_abs = "
        f"{embedded_m6_max_abs:.3e}"
    )

    # --------------------------------------------------------
    # Full results.
    # --------------------------------------------------------

    summary = pd.DataFrame(
        d2c.closed.audit.base.error_rows(
            error_stats,
            d2c.closed.MODEL_ORDER,
            16,
        )
    )

    differences = (
        d2c.closed.make_differences(
            summary
        )
    )

    gate_df = (
        d2c.closed.finalize_gate_stats(
            gate_stats
        )
    )

    compact = make_compact_metrics(
        summary
    )

    compact_gate = make_compact_gate(
        gate_df
    )

    # --------------------------------------------------------
    # Compact terminal output only.
    # --------------------------------------------------------

    print()
    print(
        "========== BUOYANCY =========="
    )

    print(
        compact[
            compact["field"]
            ==
            "buoyancy"
        ][
            [
                "horizon",
                "M6",
                "B_only",
                "Legacy10",
                "TemporalAlpha20",
                "Legacy_minus_Bonly_pp",
                "Legacy_minus_M6_pp",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print(
        "========== GLOBAL =========="
    )

    print(
        compact[
            compact["field"]
            ==
            "global"
        ][
            [
                "horizon",
                "M6",
                "B_only",
                "Legacy10",
                "TemporalAlpha20",
                "Legacy_minus_Bonly_pp",
                "Legacy_minus_M6_pp",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print(
        "========== GATE q_mean =========="
    )

    print(
        compact_gate.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Predeclared decision.
    # No new tuned threshold.
    # --------------------------------------------------------

    core_conditions = []

    for horizon in [
        8,
        16,
    ]:

        for field in [
            "buoyancy",
            "global",
        ]:

            legacy = rel_value(
                summary,
                "UtilityGate-Legacy10",
                horizon,
                field,
            )

            bonly = rel_value(
                summary,
                "B-only",
                horizon,
                field,
            )

            core_conditions.append(
                legacy < bonly
            )

    core_pass = all(
        core_conditions
    )

    strong_conditions = []

    for field in [
        "buoyancy",
        "global",
    ]:

        legacy = rel_value(
            summary,
            "UtilityGate-Legacy10",
            16,
            field,
        )

        m6_value = rel_value(
            summary,
            "M6",
            16,
            field,
        )

        strong_conditions.append(
            legacy <= m6_value
        )

    strong_pass = (
        core_pass
        and
        all(
            strong_conditions
        )
    )

    legacy_uy_h16 = rel_value(
        summary,
        "UtilityGate-Legacy10",
        16,
        "u_y",
    )

    bonly_uy_h16 = rel_value(
        summary,
        "B-only",
        16,
        "u_y",
    )

    m6_uy_h16 = rel_value(
        summary,
        "M6",
        16,
        "u_y",
    )

    if strong_pass:

        decision = (
            "STRONG_PASS"
        )

    elif core_pass:

        decision = (
            "PASS_CORE_ONLY"
        )

    else:

        decision = (
            "FAIL_CORE"
        )

    print()
    print(
        "========== H16 DECISION =========="
    )

    print(
        "Legacy10 primary:",
        decision,
    )

    print(
        "h16 u_y: "
        f"Legacy-Bonly="
        f"{legacy_uy_h16 - bonly_uy_h16:+.6f} pp | "
        f"Legacy-M6="
        f"{legacy_uy_h16 - m6_uy_h16:+.6f} pp"
    )

    # --------------------------------------------------------
    # Save EVERYTHING, despite compact terminal.
    # --------------------------------------------------------

    summary_path = (
        prefix
        +
        "_summary.csv"
    )

    differences_path = (
        prefix
        +
        "_differences.csv"
    )

    gate_path = (
        prefix
        +
        "_gate_stats.csv"
    )

    predictor_path = (
        prefix
        +
        "_predictor_sanity.csv"
    )

    compact_path = (
        prefix
        +
        "_compact.csv"
    )

    compact_gate_path = (
        prefix
        +
        "_compact_gate.csv"
    )

    audit_path = (
        prefix
        +
        "_audit_trace.txt"
    )

    metadata_path = (
        prefix
        +
        "_metadata.json"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    differences.to_csv(
        differences_path,
        index=False,
    )

    gate_df.to_csv(
        gate_path,
        index=False,
    )

    predictor_sanity.to_csv(
        predictor_path,
        index=False,
    )

    compact.to_csv(
        compact_path,
        index=False,
    )

    compact_gate.to_csv(
        compact_gate_path,
        index=False,
    )

    with open(
        audit_path,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "\n".join(
                audit_trace
            )
        )

    metadata = {
        "experiment":
            (
                "D2-2d Canonical Utility-Gate "
                "Long-Rollout VAL H16"
            ),

        "stage":
            (
                "formal_long_rollout_validation"
            ),

        "neural_operator_training":
            False,

        "d2_checkpoint_frozen":
            True,

        "utility_classifier_training":
            "TRAIN-only H4 rows",

        "h16_refit":
            False,

        "classifier_epochs":
            args.classifier_epochs,

        "closed_loop_evaluation":
            "VAL-only",

        "test_split_accessed":
            False,

        "path_a_enabled":
            False,

        "step1_trust":
            1.0,

        "step2_to_16_trust":
            (
                "frozen soft P(Path B helps | "
                "inference-visible state)"
            ),

        "thresholding":
            False,

        "classifier_primary":
            "Legacy10",

        "classifier_ablation":
            "TemporalAlpha20",

        "max_horizon":
            16,

        "formal_h16_windows":
            (
                EXPECTED_FORMAL_H16_WINDOWS
                if args.max_val_samples
                is None
                else None
            ),

        "max_val_samples":
            args.max_val_samples,

        "split_label":
            args.split_label,

        "seed":
            args.seed,

        "d2_2c_source_sha256":
            sha256_file(
                D2_2C_PATH
            ),

        "d2_2c_metadata_sha256":
            h4_lock[
                "metadata_sha"
            ],

        "d2_2c_summary_sha256":
            h4_lock[
                "summary_sha"
            ],

        "split_sha256":
            provenance[
                "split_sha"
            ],

        "stats_sha256":
            provenance[
                "stats_sha"
            ],

        "m6_sha256":
            provenance[
                "m6_sha"
            ],

        "d2_checkpoint_sha256":
            provenance[
                "d2_sha"
            ],

        "train_utility_csv_sha256":
            provenance[
                "train_sha"
            ],

        "val_utility_csv_sha256":
            provenance[
                "val_sha"
            ],

        "d2_2a_metadata_sha256":
            provenance[
                "d2a_meta_sha"
            ],

        "d2_2b_metadata_sha256":
            provenance[
                "d2b_meta_sha"
            ],

        "d2_2b_classification_sha256":
            provenance[
                "classification_sha"
            ],

        "embedded_m6_max_abs":
            embedded_m6_max_abs,

        "decision":
            decision,

        "decision_definition": {
            "core_pass":
                (
                    "Legacy10 better than B-only "
                    "at h8 and h16 on both "
                    "buoyancy and global"
                ),

            "strong_pass":
                (
                    "core_pass plus Legacy10 "
                    "better than or equal to M6 "
                    "at h16 on buoyancy/global"
                ),

            "u_y":
                (
                    "reported separately; "
                    "no post-hoc materiality "
                    "threshold introduced"
                ),
        },

        "h16_u_y_diffs_pp": {
            "Legacy_minus_Bonly":
                (
                    legacy_uy_h16
                    -
                    bonly_uy_h16
                ),

            "Legacy_minus_M6":
                (
                    legacy_uy_h16
                    -
                    m6_uy_h16
                ),
        },

        "outputs": {},
    }

    output_paths = [
        summary_path,
        differences_path,
        gate_path,
        predictor_path,
        compact_path,
        compact_gate_path,
        audit_path,
    ]

    metadata[
        "outputs"
    ] = {
        os.path.basename(path):
            sha256_file(path)
        for path in output_paths
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "✅ Full CSV/audit/metadata saved"
    )

    print(
        "✅ D2-2d finished"
    )


if __name__ == "__main__":
    main()
