import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)


# ============================================================
# Reuse CLOSED M10-2b implementation EXACTLY.
#
# We do NOT copy or modify:
#   - feature definitions
#   - TRAIN normalization
#   - classifier architecture
#   - regressor architecture
#   - losses
#   - optimizer
#   - epochs
#   - batch size
#   - AUC / accuracy equations
#   - predicted-bin protocol
# ============================================================

ORIGINAL_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "audit_m10_2b_pathb_utility_predictability.py",
)

spec = importlib.util.spec_from_file_location(
    "closed_m10_2b",
    ORIGINAL_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Cannot load closed M10-2b source: "
        f"{ORIGINAL_PATH}"
    )

original = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    original
)


EXPECTED_TRAIN_ROWS = 17856
EXPECTED_VAL_ROWS = 2232

EXPECTED_TRAIN_ROLLOUT_ROWS = 13392
EXPECTED_VAL_ROLLOUT_ROWS = 1674


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "D2-2b canonical Path-B utility "
            "predictability audit. "
            "Reuses closed M10-2b algorithm unchanged."
        )
    )

    parser.add_argument(
        "--train_csv",
        required=True,
    )

    parser.add_argument(
        "--val_csv",
        required=True,
    )

    parser.add_argument(
        "--d2_2a_metadata",
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
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--run_name",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/tables/"
            "d2_2b_canonical_utility"
        ),
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def sha256_file(
    path,
):

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


def resolve_output_dir(
    value,
):

    if os.path.isabs(
        value
    ):
        return value

    return os.path.join(
        PROJECT_ROOT,
        value,
    )


# ============================================================
# Input contract
# ============================================================

def audit_inputs(
    args,
):

    for path in [
        args.train_csv,
        args.val_csv,
        args.d2_2a_metadata,
        ORIGINAL_PATH,
    ]:

        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                path
            )

    with open(
        args.d2_2a_metadata,
        "r",
        encoding="utf-8",
    ) as handle:

        metadata = json.load(
            handle
        )

    if (
        metadata.get(
            "experiment"
        )
        !=
        "D2-2a Canonical Path-B Utility-Row Regeneration"
    ):
        raise RuntimeError(
            "Unexpected D2-2a metadata experiment: "
            f"{metadata.get('experiment')}"
        )

    if (
        metadata.get(
            "split_label"
        )
        !=
        args.split_label
    ):
        raise RuntimeError(
            "split_label mismatch: "
            f"metadata={metadata.get('split_label')}, "
            f"cli={args.split_label}"
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
            "seed mismatch: "
            f"metadata={metadata.get('seed')}, "
            f"cli={args.seed}"
        )

    locked_false = [
        "main_operator_training",
        "utility_classifier_training",
        "utility_gate_enabled",
        "test_split_accessed",
    ]

    for key in locked_false:

        if (
            metadata.get(
                key
            )
            is not False
        ):

            raise RuntimeError(
                f"D2-2a metadata contract "
                f"failed: {key}="
                f"{metadata.get(key)}"
            )

    if (
        metadata.get(
            "closed_m10_2a_generate_rows_reused_unchanged"
        )
        is not True
    ):

        raise RuntimeError(
            "D2-2a did not declare "
            "unchanged closed M10-2a "
            "generate_rows."
        )

    if (
        int(
            metadata.get(
                "train_rows",
                -1,
            )
        )
        !=
        EXPECTED_TRAIN_ROWS
    ):

        raise RuntimeError(
            "Unexpected D2-2a TRAIN row count "
            f"in metadata: "
            f"{metadata.get('train_rows')}"
        )

    if (
        int(
            metadata.get(
                "val_rows",
                -1,
            )
        )
        !=
        EXPECTED_VAL_ROWS
    ):

        raise RuntimeError(
            "Unexpected D2-2a VAL row count "
            f"in metadata: "
            f"{metadata.get('val_rows')}"
        )

    train = pd.read_csv(
        args.train_csv
    )

    val = pd.read_csv(
        args.val_csv
    )

    if len(train) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            "Unexpected TRAIN CSV rows: "
            f"{len(train)}"
        )

    if len(val) != EXPECTED_VAL_ROWS:
        raise RuntimeError(
            "Unexpected VAL CSV rows: "
            f"{len(val)}"
        )

    required_columns = {
        "split",
        "step",
        "utility_gain",
        "path_b_helps",
        "m6_b_error",
        "b_only_b_error",
        "q_target",
        "alpha_b",
    }

    for name in original.LEGACY:
        required_columns.add(
            f"legacy__{name}"
        )

    for name in original.CAP:
        required_columns.add(
            f"cap__{name}"
        )

    for name in original.TEMPORAL:
        required_columns.add(
            f"temporal__{name}"
        )

    missing = (
        required_columns.difference(
            train.columns
        )
        |
        required_columns.difference(
            val.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Missing D2-2b required columns: "
            f"{sorted(missing)}"
        )

    # ========================================================
    # Exact H4 coverage.
    # ========================================================

    for frame_name, frame, expected in [
        (
            "TRAIN",
            train,
            4464,
        ),
        (
            "VAL",
            val,
            558,
        ),
    ]:

        counts = (
            frame.groupby(
                "step"
            )
            .size()
            .to_dict()
        )

        expected_counts = {
            1: expected,
            2: expected,
            3: expected,
            4: expected,
        }

        if counts != expected_counts:

            raise RuntimeError(
                f"{frame_name} H4 coverage mismatch: "
                f"{counts}"
            )

    # ========================================================
    # Critical utility-label reproduction lock.
    #
    # M10-2b defines:
    #
    # relative_gain =
    #     utility_gain / (m6_b_error + 1e-12)
    #
    # Since denominator > 0, its binary sign MUST equal
    # D2-2a path_b_helps exactly.
    # ========================================================

    for frame_name, frame in [
        (
            "TRAIN",
            train,
        ),
        (
            "VAL",
            val,
        ),
    ]:

        rollout = frame[
            frame[
                "step"
            ] >= 2
        ].copy()

        relative_gain = (
            rollout[
                "utility_gain"
            ].to_numpy(
                dtype=np.float64
            )
            /
            (
                rollout[
                    "m6_b_error"
                ].to_numpy(
                    dtype=np.float64
                )
                +
                1.0e-12
            )
        )

        reproduced_help = (
            relative_gain > 0.0
        ).astype(
            np.int64
        )

        saved_help = (
            rollout[
                "path_b_helps"
            ].to_numpy(
                dtype=np.int64
            )
        )

        mismatch = int(
            np.sum(
                reproduced_help
                !=
                saved_help
            )
        )

        if mismatch != 0:

            raise RuntimeError(
                f"{frame_name} utility-label "
                f"reproduction failed: "
                f"mismatch={mismatch}"
            )

        expected_rollout_rows = (
            EXPECTED_TRAIN_ROLLOUT_ROWS
            if frame_name == "TRAIN"
            else
            EXPECTED_VAL_ROLLOUT_ROWS
        )

        if len(rollout) != expected_rollout_rows:

            raise RuntimeError(
                f"{frame_name} rollout rows "
                f"mismatch: {len(rollout)}"
            )

    return (
        metadata,
        train,
        val,
    )


# ============================================================
# Run CLOSED M10-2b main() unchanged.
# ============================================================

def run_closed_m10_2b(
    args,
):

    sys.argv[:] = [
        sys.argv[0],

        "--train_csv",
        args.train_csv,

        "--val_csv",
        args.val_csv,

        "--seed",
        str(
            args.seed
        ),

        "--epochs",
        str(
            args.epochs
        ),

        "--run_name",
        args.run_name,

        "--output_dir",
        args.output_dir,
    ]

    original.main()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    print(
        "=" * 110
    )

    print(
        "D2-2b CANONICAL PATH-B "
        "UTILITY PREDICTABILITY"
    )

    print(
        "=" * 110
    )

    print(
        "📌 Stage: lightweight classifier "
        "control experiment"
    )

    print(
        "📌 Main neural operator training: OFF"
    )

    print(
        "📌 Neural operator rollout: OFF"
    )

    print(
        "📌 Utility Gate: OFF"
    )

    print(
        "📌 Input: frozen D2-2a "
        "TRAIN/VAL utility rows"
    )

    print(
        "📌 Step1 excluded by closed "
        "M10-2b implementation"
    )

    print(
        "📌 Classifier/regressor algorithm: "
        "UNCHANGED from closed M10-2b"
    )

    print(
        "📌 TRAIN normalization only"
    )

    print(
        "📌 TEST split: NOT ACCESSED"
    )

    print(
        "📌 Legacy10: PRIMARY candidate "
        "for later closed-loop test"
    )

    print(
        "📌 TemporalAlpha20: PREDECLARED "
        "ablation"
    )

    print(
        "📌 Offline AUC will NOT be used "
        "to post-hoc choose the final gate"
    )

    print(
        "📌 Split:",
        args.split_label,
    )

    print(
        "📌 Seed:",
        args.seed,
    )

    print(
        "📌 Epochs:",
        args.epochs,
    )

    print()
    print(
        "========== D2-2a INPUT AUDIT =========="
    )

    (
        d2_metadata,
        train,
        val,
    ) = audit_inputs(
        args
    )

    train_sha = sha256_file(
        args.train_csv
    )

    val_sha = sha256_file(
        args.val_csv
    )

    metadata_sha = sha256_file(
        args.d2_2a_metadata
    )

    source_sha = sha256_file(
        ORIGINAL_PATH
    )

    print(
        "TRAIN_CSV_SHA256:",
        train_sha,
    )

    print(
        "VAL_CSV_SHA256:",
        val_sha,
    )

    print(
        "D2_2A_METADATA_SHA256:",
        metadata_sha,
    )

    print(
        "CLOSED_M10_2B_SOURCE_SHA256:",
        source_sha,
    )

    print(
        "TRAIN_ROWS_FULL:",
        len(
            train
        ),
    )

    print(
        "VAL_ROWS_FULL:",
        len(
            val
        ),
    )

    print(
        "TRAIN_ROWS_STEP2_4:",
        len(
            train[
                train["step"] >= 2
            ]
        ),
    )

    print(
        "VAL_ROWS_STEP2_4:",
        len(
            val[
                val["step"] >= 2
            ]
        ),
    )

    print(
        "UTILITY_LABEL_REPRODUCTION:",
        "EXACT PASS",
    )

    print(
        "✅ D2-2a input/provenance "
        "contract PASS"
    )

    print()
    print(
        "=" * 110
    )

    print(
        "RUNNING CLOSED M10-2b "
        "ALGORITHM UNCHANGED"
    )

    print(
        "The inner banner still says "
        "'M10-2b' by design."
    )

    print(
        "=" * 110
    )

    run_closed_m10_2b(
        args
    )

    # ========================================================
    # Output provenance metadata.
    # ========================================================

    output_dir = resolve_output_dir(
        args.output_dir
    )

    prefix = os.path.join(
        output_dir,
        args.run_name,
    )

    expected_outputs = [
        prefix
        + "_utility_summary.csv",

        prefix
        + "_classification.csv",

        prefix
        + "_regression.csv",

        prefix
        + "_predicted_bins.csv",
    ]

    for path in expected_outputs:

        if not os.path.exists(
            path
        ):
            raise RuntimeError(
                "Expected D2-2b output "
                f"not found: {path}"
            )

    classification = pd.read_csv(
        prefix
        + "_classification.csv"
    )

    expected_features = {
        "Legacy10",
        "LegacyCap11",
        "Temporal19",
        "TemporalAlpha20",
    }

    actual_features = set(
        classification[
            "features"
        ].tolist()
    )

    if (
        actual_features
        !=
        expected_features
    ):

        raise RuntimeError(
            "Unexpected classifier feature "
            f"sets: {actual_features}"
        )

    metadata_out = {
        "experiment":
            (
                "D2-2b Canonical Path-B "
                "Utility Predictability"
            ),

        "stage":
            "lightweight_classifier_control",

        "main_operator_training":
            False,

        "neural_operator_rollout":
            False,

        "utility_gate_enabled":
            False,

        "test_split_accessed":
            False,

        "closed_m10_2b_algorithm_reused_unchanged":
            True,

        "closed_m10_2b_source":
            ORIGINAL_PATH,

        "closed_m10_2b_source_sha256":
            source_sha,

        "split_label":
            args.split_label,

        "seed":
            args.seed,

        "epochs":
            args.epochs,

        "train_csv":
            args.train_csv,

        "train_csv_sha256":
            train_sha,

        "val_csv":
            args.val_csv,

        "val_csv_sha256":
            val_sha,

        "d2_2a_metadata":
            args.d2_2a_metadata,

        "d2_2a_metadata_sha256":
            metadata_sha,

        "d2_checkpoint":
            d2_metadata.get(
                "d2_checkpoint"
            ),

        "d2_checkpoint_sha256":
            d2_metadata.get(
                "d2_checkpoint_sha256"
            ),

        "dimensional_interface":
            d2_metadata.get(
                "dimensional_interface"
            ),

        "path_b_rms_cap":
            d2_metadata.get(
                "path_b_rms_cap"
            ),

        "step1_excluded":
            True,

        "train_rows_step2_4":
            EXPECTED_TRAIN_ROLLOUT_ROWS,

        "val_rows_step2_4":
            EXPECTED_VAL_ROLLOUT_ROWS,

        "utility_label_reproduction":
            "exact_pass",

        "classifier_primary":
            "Legacy10",

        "classifier_ablation":
            "TemporalAlpha20",

        "classifier_selection_rule":
            (
                "predeclared; offline AUC "
                "does not select final gate"
            ),

        "feature_sets":
            [
                "Legacy10",
                "LegacyCap11",
                "Temporal19",
                "TemporalAlpha20",
            ],

        "outputs": {
            os.path.basename(path):
                sha256_file(path)
            for path in expected_outputs
        },
    }

    metadata_path = (
        prefix
        +
        "_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata_out,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "✅ D2-2b provenance metadata saved:"
    )

    print(
        " ",
        metadata_path,
    )

    print()
    print(
        "✅ D2-2b finished."
    )


if __name__ == "__main__":
    main()
