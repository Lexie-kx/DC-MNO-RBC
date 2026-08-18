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


# ============================================================
# Closed M10-2c H4 implementation.
#
# Reused unchanged:
#   - UtilityClassifier
#   - fit_classifier
#   - runtime Legacy10 features
#   - runtime TemporalAlpha20 features
#   - predict_probability
#   - make_path_b_delta
#   - gate statistics
#   - independent closed-loop branch rollout
#   - error/difference equations
# ============================================================

CLOSED_M10_2C_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "diagnose_m10_2c_closedloop_utility_gate.py",
)

EXPECTED_CLOSED_M10_2C_SHA = (
    "a4c19c6b9e7290fc9ebf11c5924f486b"
    "ed854ad59d59eb5f85a4dd21aaa6628b"
)


def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


if not os.path.exists(
    CLOSED_M10_2C_PATH
):
    raise FileNotFoundError(
        CLOSED_M10_2C_PATH
    )

actual_closed_sha = sha256_file(
    CLOSED_M10_2C_PATH
)

if (
    actual_closed_sha
    !=
    EXPECTED_CLOSED_M10_2C_SHA
):
    raise RuntimeError(
        "Closed M10-2c source SHA mismatch\n"
        f"expected={EXPECTED_CLOSED_M10_2C_SHA}\n"
        f"actual={actual_closed_sha}"
    )


spec = importlib.util.spec_from_file_location(
    "closed_m10_2c_h4",
    CLOSED_M10_2C_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Cannot load {CLOSED_M10_2C_PATH}"
    )

closed = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    closed
)


from models.operators.fno2d_d2_1_dimconsistent_bonly import (
    D21DimConsistentBOnlyFNO2d,
)


# ============================================================
# Locked D2 interface
# ============================================================

DX = 1.0 / 64.0
DY = 1.0 / 63.0
DT = 0.25

ALPHA_MAX = 0.25
CONDITIONER_HIDDEN = 32
PATH_B_RMS_EPS = 1.0e-12

EXPECTED_TRAIN_ROWS_FULL = 17856
EXPECTED_VAL_ROWS_FULL = 2232

EXPECTED_TRAIN_ROWS_STEP2_4 = 13392
EXPECTED_VAL_ROWS_STEP2_4 = 1674

EXPECTED_FORMAL_H4_WINDOWS = 558


LOCKED = {
    "unseen_pr": {
        "split_sha":
            (
                "ca4f1707c913c880b33398bdf17906ae"
                "70ea4dba77662a76d12de5e319cf86be"
            ),

        "stats_sha":
            (
                "299828fb4c986f54493a552fdea8871e"
                "114fc6dd0b756c45dde0117340b54233"
            ),

        "m6_sha":
            (
                "f7a91e4661703127d90c5a7e7ae6864"
                "56f96ef3525e720dd605f422047001cab"
            ),

        "cap":
            0.53869654465,
    },

    "unseen_ra": {
        "split_sha":
            (
                "475d3092bb9d0ad16f023088446419b"
                "5651b1186d369ccfe0019c841f1fd8e36"
            ),

        "stats_sha":
            (
                "a96b1a01cf25d7b9910e01abd4de567"
                "2078e6ec3f6a6dda8a96a19e1a022d5af"
            ),

        "m6_sha":
            (
                "3d0c1571dcc57e65b1cad45fbdcae72"
                "e2b737249033d379426dc616a6c414e53"
            ),

        "cap":
            0.39047183691,
    },
}


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "D2-2c canonical Utility-Gate "
            "closed-loop validation, H4 only."
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
            "Smoke only. "
            "Formal run leaves this unset."
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
            "d2_2c_canonical_gate_h4"
        ),
    )

    return parser.parse_args()


# ============================================================
# Reproducibility / scalar helpers
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def require_sha(
    path,
    expected,
    label,
):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    actual = sha256_file(path)

    print(
        f"{label}_SHA256:",
        actual,
    )

    if actual != expected:

        raise RuntimeError(
            f"{label} SHA mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}\n"
            f"path={path}"
        )

    return actual


def require_close(
    actual,
    expected,
    label,
    tol=1.0e-10,
):

    if (
        actual is None
        or not math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=tol,
        )
    ):

        raise RuntimeError(
            f"{label} mismatch: "
            f"actual={actual}, "
            f"expected={expected}"
        )


def resolve_output_dir(path):

    if os.path.isabs(path):
        return path

    return os.path.join(
        PROJECT_ROOT,
        path,
    )


# ============================================================
# Provenance audit
# ============================================================

def audit_provenance(
    args,
    lock,
):

    for path in [
        args.split,
        args.stats,
        args.m6_checkpoint,
        args.d2_checkpoint,
        args.train_utility_csv,
        args.val_utility_csv,
        args.d2_2a_metadata,
        args.d2_2b_metadata,
        args.d2_2b_classification_csv,
    ]:

        if not os.path.exists(path):
            raise FileNotFoundError(path)

    split_sha = require_sha(
        args.split,
        lock["split_sha"],
        "SPLIT",
    )

    stats_sha = require_sha(
        args.stats,
        lock["stats_sha"],
        "STATS",
    )

    m6_sha = require_sha(
        args.m6_checkpoint,
        lock["m6_sha"],
        "M6",
    )

    d2_sha = sha256_file(
        args.d2_checkpoint
    )

    train_sha = sha256_file(
        args.train_utility_csv
    )

    val_sha = sha256_file(
        args.val_utility_csv
    )

    d2a_meta_sha = sha256_file(
        args.d2_2a_metadata
    )

    d2b_meta_sha = sha256_file(
        args.d2_2b_metadata
    )

    classification_sha = sha256_file(
        args.d2_2b_classification_csv
    )

    print(
        "D2_CHECKPOINT_SHA256:",
        d2_sha,
    )

    print(
        "TRAIN_UTILITY_CSV_SHA256:",
        train_sha,
    )

    print(
        "VAL_UTILITY_CSV_SHA256:",
        val_sha,
    )

    print(
        "D2_2A_METADATA_SHA256:",
        d2a_meta_sha,
    )

    print(
        "D2_2B_METADATA_SHA256:",
        d2b_meta_sha,
    )

    print(
        "D2_2B_CLASSIFICATION_SHA256:",
        classification_sha,
    )

    print(
        "CLOSED_M10_2C_SOURCE_SHA256:",
        actual_closed_sha,
    )

    # --------------------------------------------------------
    # D2-2a metadata
    # --------------------------------------------------------

    with open(
        args.d2_2a_metadata,
        "r",
        encoding="utf-8",
    ) as handle:

        meta_a = json.load(handle)

    if (
        meta_a.get("experiment")
        !=
        "D2-2a Canonical Path-B Utility-Row Regeneration"
    ):
        raise RuntimeError(
            "Unexpected D2-2a experiment metadata."
        )

    if (
        meta_a.get("split_label")
        !=
        args.split_label
    ):
        raise RuntimeError(
            "D2-2a split_label mismatch."
        )

    if int(
        meta_a.get(
            "seed",
            -1,
        )
    ) != args.seed:

        raise RuntimeError(
            "D2-2a seed mismatch."
        )

    if (
        meta_a.get("split_sha256")
        !=
        split_sha
    ):
        raise RuntimeError(
            "D2-2a split provenance mismatch."
        )

    if (
        meta_a.get("stats_sha256")
        !=
        stats_sha
    ):
        raise RuntimeError(
            "D2-2a stats provenance mismatch."
        )

    if (
        meta_a.get(
            "m6_checkpoint_sha256"
        )
        !=
        m6_sha
    ):
        raise RuntimeError(
            "D2-2a M6 provenance mismatch."
        )

    if (
        meta_a.get(
            "d2_checkpoint_sha256"
        )
        !=
        d2_sha
    ):
        raise RuntimeError(
            "D2-2a D2 checkpoint mismatch."
        )

    if int(
        meta_a.get(
            "train_rows",
            -1,
        )
    ) != EXPECTED_TRAIN_ROWS_FULL:

        raise RuntimeError(
            "D2-2a TRAIN row-count mismatch."
        )

    if int(
        meta_a.get(
            "val_rows",
            -1,
        )
    ) != EXPECTED_VAL_ROWS_FULL:

        raise RuntimeError(
            "D2-2a VAL row-count mismatch."
        )

    interface = meta_a.get(
        "dimensional_interface",
        {},
    )

    require_close(
        interface.get("dx"),
        DX,
        "D2-2a dx",
    )

    require_close(
        interface.get("dy"),
        DY,
        "D2-2a dy",
    )

    require_close(
        interface.get("dt"),
        DT,
        "D2-2a dt",
    )

    if (
        interface.get(
            "rate_to_increment"
        )
        is not True
    ):
        raise RuntimeError(
            "D2-2a rate_to_increment "
            "must be True."
        )

    require_close(
        meta_a.get(
            "path_b_rms_cap"
        ),
        lock["cap"],
        "D2-2a canonical cap",
    )

    # --------------------------------------------------------
    # D2-2b metadata
    # --------------------------------------------------------

    with open(
        args.d2_2b_metadata,
        "r",
        encoding="utf-8",
    ) as handle:

        meta_b = json.load(handle)

    if (
        meta_b.get("experiment")
        !=
        "D2-2b Canonical Path-B Utility Predictability"
    ):
        raise RuntimeError(
            "Unexpected D2-2b experiment metadata."
        )

    if (
        meta_b.get("split_label")
        !=
        args.split_label
    ):
        raise RuntimeError(
            "D2-2b split_label mismatch."
        )

    if int(
        meta_b.get(
            "seed",
            -1,
        )
    ) != args.seed:

        raise RuntimeError(
            "D2-2b seed mismatch."
        )

    if (
        int(
            meta_b.get(
                "epochs",
                -1,
            )
        )
        !=
        args.classifier_epochs
    ):
        raise RuntimeError(
            "D2-2b classifier epoch mismatch."
        )

    if (
        meta_b.get(
            "closed_m10_2b_algorithm_reused_unchanged"
        )
        is not True
    ):
        raise RuntimeError(
            "D2-2b closed-algorithm "
            "contract failed."
        )

    if (
        meta_b.get(
            "classifier_primary"
        )
        !=
        "Legacy10"
    ):
        raise RuntimeError(
            "Legacy10 is not locked primary."
        )

    if (
        meta_b.get(
            "classifier_ablation"
        )
        !=
        "TemporalAlpha20"
    ):
        raise RuntimeError(
            "TemporalAlpha20 is not "
            "locked ablation."
        )

    if (
        meta_b.get(
            "train_csv_sha256"
        )
        !=
        train_sha
    ):
        raise RuntimeError(
            "D2-2b TRAIN utility CSV mismatch."
        )

    if (
        meta_b.get(
            "val_csv_sha256"
        )
        !=
        val_sha
    ):
        raise RuntimeError(
            "D2-2b VAL utility CSV mismatch."
        )

    if (
        meta_b.get(
            "d2_2a_metadata_sha256"
        )
        !=
        d2a_meta_sha
    ):
        raise RuntimeError(
            "D2-2b D2-2a metadata mismatch."
        )

    if (
        meta_b.get(
            "d2_checkpoint_sha256"
        )
        !=
        d2_sha
    ):
        raise RuntimeError(
            "D2-2b D2 checkpoint mismatch."
        )

    outputs = meta_b.get(
        "outputs",
        {},
    )

    classification_name = os.path.basename(
        args.d2_2b_classification_csv
    )

    if (
        outputs.get(
            classification_name
        )
        !=
        classification_sha
    ):
        raise RuntimeError(
            "D2-2b classification CSV "
            "provenance mismatch."
        )

    # --------------------------------------------------------
    # Utility CSV structural audit
    # --------------------------------------------------------

    train_df = pd.read_csv(
        args.train_utility_csv
    )

    val_df = pd.read_csv(
        args.val_utility_csv
    )

    if len(train_df) != EXPECTED_TRAIN_ROWS_FULL:
        raise RuntimeError(
            "Unexpected TRAIN utility rows."
        )

    if len(val_df) != EXPECTED_VAL_ROWS_FULL:
        raise RuntimeError(
            "Unexpected VAL utility rows."
        )

    train_probe = train_df[
        train_df["step"] >= 2
    ].copy()

    val_probe = val_df[
        val_df["step"] >= 2
    ].copy()

    if (
        len(train_probe)
        !=
        EXPECTED_TRAIN_ROWS_STEP2_4
    ):
        raise RuntimeError(
            "Unexpected TRAIN step2-4 rows."
        )

    if (
        len(val_probe)
        !=
        EXPECTED_VAL_ROWS_STEP2_4
    ):
        raise RuntimeError(
            "Unexpected VAL step2-4 rows."
        )

    print(
        "TRAIN_ROWS_STEP2_4:",
        len(train_probe),
    )

    print(
        "VAL_ROWS_STEP2_4:",
        len(val_probe),
    )

    return {
        "meta_a": meta_a,
        "meta_b": meta_b,
        "train_df": train_df,
        "val_df": val_df,
        "train_probe": train_probe,
        "val_probe": val_probe,
        "split_sha": split_sha,
        "stats_sha": stats_sha,
        "m6_sha": m6_sha,
        "d2_sha": d2_sha,
        "train_sha": train_sha,
        "val_sha": val_sha,
        "d2a_meta_sha": d2a_meta_sha,
        "d2b_meta_sha": d2b_meta_sha,
        "classification_sha":
            classification_sha,
    }


# ============================================================
# Build frozen D2-1
# ============================================================

def build_d2(
    checkpoint_path,
    field_mean,
    field_std,
    lock,
    device,
):

    payload = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        payload.get("experiment")
        !=
        "D2-1-DimConsistent-StateParam-BOnly-H4"
    ):
        raise RuntimeError(
            "Unexpected D2-1 experiment."
        )

    interface = payload.get(
        "dimensional_interface",
        {},
    )

    stabilization = payload.get(
        "stabilization",
        {},
    )

    conditioner = payload.get(
        "conditioner",
        {},
    )

    require_close(
        interface.get("dx"),
        DX,
        "D2 checkpoint dx",
    )

    require_close(
        interface.get("dy"),
        DY,
        "D2 checkpoint dy",
    )

    require_close(
        interface.get("dt"),
        DT,
        "D2 checkpoint dt",
    )

    if (
        interface.get(
            "rate_to_increment"
        )
        is not True
    ):
        raise RuntimeError(
            "D2 checkpoint must use "
            "rate_to_increment=True."
        )

    require_close(
        stabilization.get(
            "path_b_rms_cap"
        ),
        lock["cap"],
        "D2 checkpoint cap",
    )

    require_close(
        payload.get(
            "alpha_max"
        ),
        ALPHA_MAX,
        "alpha_max",
    )

    hidden = int(
        conditioner.get(
            "hidden_dim",
            CONDITIONER_HIDDEN,
        )
    )

    if hidden != CONDITIONER_HIDDEN:
        raise RuntimeError(
            "Unexpected conditioner hidden dim."
        )

    model = (
        D21DimConsistentBOnlyFNO2d(
            field_mean=field_mean,
            field_std=field_std,
            dx=DX,
            dy=DY,
            dt=DT,
            path_b_rms_cap=(
                lock["cap"]
            ),
            path_b_rms_eps=float(
                stabilization.get(
                    "path_b_rms_eps",
                    PATH_B_RMS_EPS,
                )
            ),
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32,
            alpha_max=ALPHA_MAX,
            conditioner_hidden=(
                CONDITIONER_HIDDEN
            ),
            freeze_m6=True,
        )
        .to(device)
    )

    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )

    model.eval()
    model.m6.eval()

    trainable_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    expected_trainable = {
        "base_raw_alpha_b",
        "conditioner.0.weight",
        "conditioner.0.bias",
        "conditioner.2.weight",
        "conditioner.2.bias",
    }

    if (
        set(trainable_names)
        !=
        expected_trainable
    ):
        raise RuntimeError(
            "D2-1 architecture mismatch: "
            f"{trainable_names}"
        )

    trainable_numel = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    if trainable_numel != 386:
        raise RuntimeError(
            "D2-1 trainable numel mismatch: "
            f"{trainable_numel}"
        )

    alpha_a = float(
        model.base_raw_alpha_a
        .detach()
        .cpu()
        .item()
    )

    if (
        model.base_raw_alpha_a.requires_grad
        or alpha_a != 0.0
    ):
        raise RuntimeError(
            "Path-A output gate mismatch."
        )

    print()
    print(
        "========== D2-1 MODEL CONTRACT =========="
    )

    print(
        "TRAINABLE_NUMEL:",
        trainable_numel,
    )

    print(
        "PATH_A_OUTPUT_GATE:",
        alpha_a,
    )

    print(
        "CANONICAL_DX:",
        DX,
    )

    print(
        "CANONICAL_DY:",
        DY,
    )

    print(
        "CANONICAL_DT:",
        DT,
    )

    print(
        "PATH_B_RMS_CAP:",
        lock["cap"],
    )

    print(
        "✅ D2-1 model contract PASS"
    )

    return model, payload


# ============================================================
# Classifier reproduction
# ============================================================

def fit_and_reproduce_classifiers(
    args,
    provenance,
):

    train_probe = provenance[
        "train_probe"
    ]

    val_probe = provenance[
        "val_probe"
    ]

    legacy = closed.fit_classifier(
        train_probe,
        val_probe,
        closed.LEGACY_COLUMNS,
        seed=args.seed,
        epochs=args.classifier_epochs,
    )

    temporal = closed.fit_classifier(
        train_probe,
        val_probe,
        closed.TEMPORAL_ALPHA_COLUMNS,
        seed=args.seed,
        epochs=args.classifier_epochs,
    )

    sanity = pd.DataFrame(
        [
            {
                "predictor":
                    "Legacy10",

                "num_features":
                    legacy["num_features"],

                "val_auc":
                    legacy["auc"],

                "val_accuracy":
                    legacy["accuracy"],
            },

            {
                "predictor":
                    "TemporalAlpha20",

                "num_features":
                    temporal["num_features"],

                "val_auc":
                    temporal["auc"],

                "val_accuracy":
                    temporal["accuracy"],
            },
        ]
    )

    d2b = pd.read_csv(
        args.d2_2b_classification_csv
    )

    expected = {}

    for name in [
        "Legacy10",
        "TemporalAlpha20",
    ]:

        sub = d2b[
            d2b["features"] == name
        ]

        if len(sub) != 1:
            raise RuntimeError(
                "Cannot uniquely locate "
                f"{name} in D2-2b results."
            )

        expected[name] = float(
            sub.iloc[0]["auc"]
        )

    print()
    print(
        "========== CLASSIFIER REPRODUCTION =========="
    )

    print(
        sanity.to_string(
            index=False
        )
    )

    for _, row in sanity.iterrows():

        name = row["predictor"]

        actual = float(
            row["val_auc"]
        )

        target = expected[name]

        diff = abs(
            actual - target
        )

        print(
            f"{name}_AUC_EXPECTED:",
            f"{target:.9f}",
        )

        print(
            f"{name}_AUC_REBUILT:",
            f"{actual:.9f}",
        )

        print(
            f"{name}_AUC_ABS_DIFF:",
            f"{diff:.9e}",
        )

        if diff > 2.0e-3:

            raise RuntimeError(
                "D2-2b classifier reproduction "
                f"failed for {name}: "
                f"actual={actual:.6f}, "
                f"expected={target:.6f}"
            )

    print(
        "✅ D2-2b classifier reproduction PASS"
    )

    return (
        legacy,
        temporal,
        sanity,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.classifier_epochs != 100:
        raise ValueError(
            "D2-2c classifier protocol is "
            "locked to 100 epochs."
        )

    if (
        args.max_val_samples is not None
        and args.max_val_samples <= 0
    ):
        raise ValueError(
            "max_val_samples must be positive."
        )

    set_seed(
        args.seed
    )

    lock = LOCKED[
        args.split_label
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 118
    )

    print(
        "D2-2c CANONICAL UTILITY-GATE "
        "CLOSED-LOOP VAL H4"
    )

    print(
        "=" * 118
    )

    print(
        "📌 Stage: FORMAL GATE CONTROL / "
        "H4 IN-SUPPORT CLOSED LOOP"
    )

    print(
        "📌 Neural-operator training: OFF"
    )

    print(
        "📌 D2-1 checkpoint: FROZEN"
    )

    print(
        "📌 Utility classifiers: "
        "TRAIN-only fit, then frozen"
    )

    print(
        "📌 Closed-loop split: VAL only"
    )

    print(
        "📌 TEST: FORBIDDEN"
    )

    print(
        "📌 Path A output injection: OFF"
    )

    print(
        "📌 Path B: canonical finite-step "
        "D2 residual"
    )

    print(
        "📌 dx=1/64, dy=1/63, dt=0.25"
    )

    print(
        "📌 step1 trust = 1"
    )

    print(
        "📌 step2-4 trust = frozen soft "
        "P(Path B helps)"
    )

    print(
        "📌 No utility threshold tuning"
    )

    print(
        "📌 Legacy10 = PRIMARY"
    )

    print(
        "📌 TemporalAlpha20 = "
        "PREDECLARED ABLATION"
    )

    print(
        "📌 H4 only; H16 belongs to D2-2d"
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
        "📌 Device:",
        device,
    )

    print()
    print(
        "========== PROVENANCE =========="
    )

    provenance = audit_provenance(
        args,
        lock,
    )

    (
        legacy_predictor,
        temporal_predictor,
        predictor_sanity,
    ) = fit_and_reproduce_classifiers(
        args,
        provenance,
    )

    # --------------------------------------------------------
    # VAL H4 rollout dataset
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
        closed.audit.base.RolloutDataset(
            split_config=(
                split_config["val"]
            ),
            max_horizon=4,
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

    print()
    print(
        "========== VAL H4 DATA =========="
    )

    print(
        "VAL_ROLLOUT_SAMPLES:",
        len(dataset),
    )

    if args.max_val_samples is None:

        if (
            len(dataset)
            !=
            EXPECTED_FORMAL_H4_WINDOWS
        ):
            raise RuntimeError(
                "Formal H4 run expected "
                f"{EXPECTED_FORMAL_H4_WINDOWS} "
                f"VAL windows, got {len(dataset)}"
            )

    else:

        if (
            len(dataset)
            !=
            args.max_val_samples
        ):
            raise RuntimeError(
                "Smoke VAL sample count mismatch."
            )

    # --------------------------------------------------------
    # Models / normalizer
    # --------------------------------------------------------

    normalizer = (
        closed.audit.base.FieldWiseNormalizer(
            args.stats
        ).to(device)
    )

    (
        field_mean,
        field_std,
    ) = (
        closed.audit.base.build_field_stats(
            args.stats
        )
    )

    m6, _ = (
        closed.audit.base.build_m6(
            args.m6_checkpoint,
            device,
        )
    )

    d2, d2_payload = build_d2(
        args.d2_checkpoint,
        field_mean,
        field_std,
        lock,
        device,
    )

    closed.audit.base.assert_same_m6(
        m6,
        d2,
        "D2-1-DimConsistent-BOnly",
    )

    print(
        "D2_1_BEST_VAL:",
        d2_payload.get(
            "best_val_loss",
            d2_payload.get(
                "val_loss",
                "NA",
            ),
        ),
    )

    # --------------------------------------------------------
    # Closed-loop H4
    # --------------------------------------------------------

    print()
    print(
        "🔥 Starting D2-2c closed-loop VAL H4..."
    )

    (
        error_stats,
        gate_stats,
        embedded_m6_max_abs,
    ) = closed.evaluate_closed_loop(
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
        max_horizon=4,
    )

    print()
    print(
        "========== EMBEDDED M6 CHECK =========="
    )

    print(
        "max_abs embedded vs independent M6 =",
        f"{embedded_m6_max_abs:.12e}",
    )

    if embedded_m6_max_abs > 1.0e-6:
        raise RuntimeError(
            "Embedded M6 mismatch."
        )

    print(
        "✅ Embedded M6 reconstruction PASS"
    )

    # --------------------------------------------------------
    # Final tables
    # --------------------------------------------------------

    summary = pd.DataFrame(
        closed.audit.base.error_rows(
            error_stats,
            closed.MODEL_ORDER,
            4,
        )
    )

    differences = (
        closed.make_differences(
            summary
        )
    )

    gate_df = (
        closed.finalize_gate_stats(
            gate_stats
        )
    )

    print()
    print(
        "=" * 118
    )

    print(
        "GLOBAL REL-L2 (%)"
    )

    print(
        "=" * 118
    )

    global_table = (
        summary[
            summary["field"]
            ==
            "global"
        ]
        .pivot(
            index="horizon",
            columns="model",
            values="rel_l2_percent",
        )
        .reindex(
            columns=closed.MODEL_ORDER
        )
    )

    print(
        global_table.to_string()
    )

    print()
    print(
        "=" * 118
    )

    print(
        "BUOYANCY REL-L2 (%)"
    )

    print(
        "=" * 118
    )

    buoy_table = (
        summary[
            summary["field"]
            ==
            "buoyancy"
        ]
        .pivot(
            index="horizon",
            columns="model",
            values="rel_l2_percent",
        )
        .reindex(
            columns=closed.MODEL_ORDER
        )
    )

    print(
        buoy_table.to_string()
    )

    print()
    print(
        "=" * 118
    )

    print(
        "UTILITY TRUST DURING CLOSED LOOP"
    )

    print(
        "=" * 118
    )

    print(
        gate_df.to_string(
            index=False
        )
    )

    decision = differences[
        (
            differences[
                "horizon"
            ].isin(
                [3, 4]
            )
        )
        &
        (
            differences[
                "field"
            ].isin(
                [
                    "global",
                    "buoyancy",
                    "u_y",
                ]
            )
        )
    ].copy()

    print()
    print(
        "=" * 118
    )

    print(
        "H3/H4 DECISION TABLE"
    )

    print(
        "Negative difference = "
        "first model better"
    )

    print(
        "=" * 118
    )

    print(
        decision.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    summary_path = (
        prefix
        +
        "_summary.csv"
    )

    diff_path = (
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
        diff_path,
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

    metadata = {
        "experiment":
            (
                "D2-2c Canonical Utility-Gate "
                "Closed-Loop VAL H4"
            ),

        "stage":
            "formal_gate_control_h4",

        "neural_operator_training":
            False,

        "d2_checkpoint_frozen":
            True,

        "utility_classifier_training":
            "TRAIN-only",

        "classifier_epochs":
            args.classifier_epochs,

        "closed_loop_evaluation":
            "VAL-only",

        "test_split_accessed":
            False,

        "path_a_enabled":
            False,

        "path_b":
            (
                "D2 canonical finite-step "
                "RMSCap-safe Path B"
            ),

        "dimensional_interface": {
            "dx": DX,
            "dy": DY,
            "dt": DT,
            "rate_to_increment": True,
        },

        "step1_trust":
            1.0,

        "step2_to_4_trust":
            (
                "soft P(Path B helps | "
                "inference-visible state)"
            ),

        "thresholding":
            False,

        "classifier_primary":
            "Legacy10",

        "classifier_ablation":
            "TemporalAlpha20",

        "max_horizon":
            4,

        "max_val_samples":
            args.max_val_samples,

        "formal_h4_windows":
            (
                EXPECTED_FORMAL_H4_WINDOWS
                if args.max_val_samples is None
                else None
            ),

        "models":
            closed.MODEL_ORDER,

        "split_label":
            args.split_label,

        "seed":
            args.seed,

        "split_sha256":
            provenance["split_sha"],

        "stats_sha256":
            provenance["stats_sha"],

        "m6_sha256":
            provenance["m6_sha"],

        "d2_checkpoint_sha256":
            provenance["d2_sha"],

        "train_utility_csv_sha256":
            provenance["train_sha"],

        "val_utility_csv_sha256":
            provenance["val_sha"],

        "d2_2a_metadata_sha256":
            provenance["d2a_meta_sha"],

        "d2_2b_metadata_sha256":
            provenance["d2b_meta_sha"],

        "d2_2b_classification_sha256":
            provenance[
                "classification_sha"
            ],

        "closed_m10_2c_source_sha256":
            actual_closed_sha,

        "closed_functions_reused":
            [
                "fit_classifier",
                "build_legacy_features",
                "build_temporal_alpha_features",
                "predict_probability",
                "make_path_b_delta",
                "evaluate_closed_loop",
                "make_differences",
                "finalize_gate_stats",
            ],

        "embedded_m6_max_abs":
            embedded_m6_max_abs,

        "outputs": {
            os.path.basename(path):
                sha256_file(path)
            for path in [
                summary_path,
                diff_path,
                gate_path,
                predictor_path,
            ]
        },
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

    print()
    print(
        "✅ Saved:"
    )

    for path in [
        summary_path,
        diff_path,
        gate_path,
        predictor_path,
        metadata_path,
    ]:

        print(
            " ",
            path,
        )

    print()
    print(
        "========== D2-2c DECISION KEY =========="
    )

    print(
        "PRIMARY = UtilityGate-Legacy10"
    )

    print(
        "PASS core:"
    )

    print(
        "  Legacy10 Gate improves over "
        "D2 B-only at h3/h4, especially "
        "buoyancy/global."
    )

    print(
        "STRONG PASS:"
    )

    print(
        "  Above condition + Gate improves "
        "or matches M6 at h4."
    )

    print(
        "PARTIAL:"
    )

    print(
        "  Gate improves over B-only but "
        "remains worse than M6."
    )

    print(
        "FAIL:"
    )

    print(
        "  Gate does not improve over B-only."
    )

    print()
    print(
        "✅ D2-2c finished."
    )


if __name__ == "__main__":
    main()
