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
# Reuse CLOSED M10-2a row-generation implementation EXACTLY.
# We do NOT edit:
#   - utility_gain equation
#   - path_b_helps definition
#   - q_target equation
#   - feature definitions
#   - generate_rows rollout logic
# ============================================================

ORIGINAL_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "audit_m10_2a_pathb_reliability.py",
)

spec = importlib.util.spec_from_file_location(
    "m10_2a_closed",
    ORIGINAL_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Cannot load {ORIGINAL_PATH}"
    )

original = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    original
)


from models.operators.fno2d_d2_1_dimconsistent_bonly import (
    D21DimConsistentBOnlyFNO2d,
)


# ============================================================
# Locked D2 interface / provenance
# ============================================================

LOCKED = {
    "unseen_pr": {
        "split_name":
            "unseen_pr_split.json",

        "stats_name":
            "rbc_field_stats_unseen_pr.json",

        "split_sha":
            "ca4f1707c913c880b33398bdf17906ae"
            "70ea4dba77662a76d12de5e319cf86be",

        "stats_sha":
            "299828fb4c986f54493a552fdea8871e"
            "114fc6dd0b756c45dde0117340b54233",

        "m6_sha":
            "f7a91e4661703127d90c5a7e7ae6864"
            "56f96ef3525e720dd605f422047001cab",

        "cap":
            0.53869654465,
    },

    "unseen_ra": {
        "split_name":
            "unseen_ra_split.json",

        "stats_name":
            "rbc_field_stats_unseen_ra.json",

        "split_sha":
            "475d3092bb9d0ad16f023088446419b"
            "5651b1186d369ccfe0019c841f1fd8e36",

        "stats_sha":
            "a96b1a01cf25d7b9910e01abd4de567"
            "2078e6ec3f6a6dda8a96a19e1a022d5af",

        "m6_sha":
            "3d0c1571dcc57e65b1cad45fbdcae72"
            "e2b737249033d379426dc616a6c414e53",

        "cap":
            0.39047183691,
    },
}


DX = 1.0 / 64.0
DY = 1.0 / 63.0
DT = 0.25

ALPHA_MAX = 0.25
HIDDEN = 32
RMS_EPS = 1.0e-12


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "D2-2a canonical Path-B utility-row "
            "regeneration."
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
        "--d2_checkpoint",
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
        default=42,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--run_name",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/tables/"
            "d2_2a_canonical_utility_rows"
        ),
    )

    return parser.parse_args()


# ============================================================
# Reproducibility / hashes
# ============================================================

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

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def require_sha(
    path,
    expected,
    label,
):

    actual = sha256_file(
        path
    )

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
    tol=1.0e-12,
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


# ============================================================
# D2-1 builder
# ============================================================

def build_d2(
    payload,
    field_mean,
    field_std,
    lock,
    device,
):

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

    if (
        payload.get(
            "experiment"
        )
        !=
        "D2-1-DimConsistent-StateParam-BOnly-H4"
    ):

        raise RuntimeError(
            "Unexpected experiment: "
            f"{payload.get('experiment')}"
        )

    require_close(
        interface.get(
            "dx"
        ),
        DX,
        "dx",
    )

    require_close(
        interface.get(
            "dy"
        ),
        DY,
        "dy",
    )

    require_close(
        interface.get(
            "dt"
        ),
        DT,
        "dt",
    )

    if (
        interface.get(
            "rate_to_increment"
        )
        is not True
    ):

        raise RuntimeError(
            "rate_to_increment must be True"
        )

    require_close(
        stabilization.get(
            "path_b_rms_cap"
        ),
        lock[
            "cap"
        ],
        "canonical RMSCap",
        tol=1.0e-10,
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
            HIDDEN,
        )
    )

    if hidden != HIDDEN:

        raise RuntimeError(
            "Unexpected conditioner "
            f"hidden_dim={hidden}"
        )

    model = (
        D21DimConsistentBOnlyFNO2d(
            field_mean=field_mean,
            field_std=field_std,
            dx=float(
                interface[
                    "dx"
                ]
            ),
            dy=float(
                interface[
                    "dy"
                ]
            ),
            dt=float(
                interface[
                    "dt"
                ]
            ),
            path_b_rms_cap=float(
                stabilization[
                    "path_b_rms_cap"
                ]
            ),
            path_b_rms_eps=float(
                stabilization.get(
                    "path_b_rms_eps",
                    RMS_EPS,
                )
            ),
            in_channels=16,
            out_channels=4,
            modes1=16,
            modes2=16,
            width=32,
            alpha_max=float(
                payload[
                    "alpha_max"
                ]
            ),
            conditioner_hidden=(
                hidden
            ),
            freeze_m6=True,
        )
        .to(
            device
        )
    )

    model.load_state_dict(
        payload[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()
    model.m6.eval()

    # ========================================================
    # Architecture contract
    # ========================================================

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
        set(
            trainable_names
        )
        !=
        expected_trainable
    ):

        raise RuntimeError(
            "Unexpected trainable "
            f"parameters: {trainable_names}"
        )

    trainable_numel = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    if trainable_numel != 386:

        raise RuntimeError(
            "Unexpected trainable "
            f"numel={trainable_numel}"
        )

    alpha_a = float(
        model.base_raw_alpha_a
        .detach()
        .cpu()
        .item()
    )

    if (
        model.base_raw_alpha_a
        .requires_grad
        or alpha_a != 0.0
    ):

        raise RuntimeError(
            "Path-A output gate invalid: "
            f"{alpha_a}"
        )

    frozen_m6 = sum(
        1
        for parameter
        in model.m6.parameters()
        if not parameter.requires_grad
    )

    if frozen_m6 != 38:

        raise RuntimeError(
            "Unexpected frozen M6 "
            f"tensor count={frozen_m6}"
        )

    print()
    print(
        "========== D2-1 "
        "ARCHITECTURE CONTRACT =========="
    )

    print(
        "TRAINABLE_NAMES:",
        trainable_names,
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
        "FROZEN_M6_PARAMETER_TENSORS:",
        frozen_m6,
    )

    print(
        "✅ D2-1 architecture "
        "contract PASS"
    )

    return model


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.rollout_steps != 4:

        raise ValueError(
            "D2-2a is locked to H4."
        )

    lock = LOCKED[
        args.split_label
    ]

    if (
        os.path.basename(
            args.split
        )
        !=
        lock[
            "split_name"
        ]
    ):

        raise RuntimeError(
            "split filename does not "
            "match split_label"
        )

    if (
        os.path.basename(
            args.stats
        )
        !=
        lock[
            "stats_name"
        ]
    ):

        raise RuntimeError(
            "stats filename does not "
            "match split_label"
        )

    for path in [
        args.split,
        args.stats,
        args.d2_checkpoint,
        ORIGINAL_PATH,
    ]:

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                path
            )

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 100
    )

    print(
        "D2-2a CANONICAL PATH-B "
        "UTILITY-ROW REGENERATION"
    )

    print(
        "=" * 100
    )

    print(
        "Stage: CONTROL / PREPARATION"
    )

    print(
        "Main neural operator training: OFF"
    )

    print(
        "Utility classifier training: OFF"
    )

    print(
        "Utility Gate: OFF"
    )

    print(
        "TRAIN + VAL only"
    )

    print(
        "TEST access: FORBIDDEN"
    )

    print(
        "Closed M10-2a generate_rows "
        "reused unchanged: YES"
    )

    print(
        "Utility-label equation changed: NO"
    )

    print(
        "Reliability-target equation changed: NO"
    )

    print(
        f"Canonical interface: "
        f"dx={DX}, "
        f"dy={DY}, "
        f"dt={DT}"
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
        "Device:",
        device,
    )

    print()
    print(
        "========== PROVENANCE =========="
    )

    split_sha = require_sha(
        args.split,
        lock[
            "split_sha"
        ],
        "SPLIT",
    )

    stats_sha = require_sha(
        args.stats,
        lock[
            "stats_sha"
        ],
        "STATS",
    )

    d2_sha = sha256_file(
        args.d2_checkpoint
    )

    source_sha = sha256_file(
        ORIGINAL_PATH
    )

    print(
        "D2_CHECKPOINT_SHA256:",
        d2_sha,
    )

    print(
        "CLOSED_M10_2A_GENERATOR_SHA256:",
        source_sha,
    )

    payload = torch.load(
        args.d2_checkpoint,
        map_location=device,
    )

    if (
        int(
            payload.get(
                "seed",
                -1,
            )
        )
        !=
        args.seed
    ):

        raise RuntimeError(
            "checkpoint seed mismatch"
        )

    if (
        payload.get(
            "split_sha256"
        )
        !=
        split_sha
    ):

        raise RuntimeError(
            "checkpoint split SHA mismatch"
        )

    if (
        payload.get(
            "stats_sha256"
        )
        !=
        stats_sha
    ):

        raise RuntimeError(
            "checkpoint stats SHA mismatch"
        )

    if (
        payload.get(
            "m6_checkpoint_sha256"
        )
        !=
        lock[
            "m6_sha"
        ]
    ):

        raise RuntimeError(
            "checkpoint M6 provenance mismatch"
        )

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as handle:

        split = json.load(
            handle
        )

    # ========================================================
    # TRAIN / VAL ONLY.
    # TEST is intentionally never instantiated.
    # ========================================================

    train_base = original.RBCDataset(
        split_config=(
            split[
                "train"
            ]
        ),
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=4,
        return_params=True,
    )

    val_base = original.RBCDataset(
        split_config=(
            split[
                "val"
            ]
        ),
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=4,
        return_params=True,
    )

    train_dataset = (
        original.M10MultiStepParamDataset(
            train_base,
            context_length=4,
        )
    )

    val_dataset = (
        original.M10MultiStepParamDataset(
            val_base,
            context_length=4,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print()
    print(
        "========== DATA =========="
    )

    print(
        "TRAIN_SAMPLES:",
        len(
            train_dataset
        ),
    )

    print(
        "VAL_SAMPLES:",
        len(
            val_dataset
        ),
    )

    (
        field_mean,
        field_std,
    ) = (
        original.audit.base
        .build_field_stats(
            args.stats
        )
    )

    model = build_d2(
        payload,
        field_mean,
        field_std,
        lock,
        device,
    )

    print()
    print(
        "🔥 Generating TRAIN "
        "canonical utility rows..."
    )

    train_df = original.generate_rows(
        split_name="train",
        loader=train_loader,
        model=model,
        device=device,
        rollout_steps=4,
        max_batches=(
            args.max_train_batches
        ),
    )

    print(
        "🔥 Generating VAL "
        "canonical utility rows..."
    )

    val_df = original.generate_rows(
        split_name="val",
        loader=val_loader,
        model=model,
        device=device,
        rollout_steps=4,
        max_batches=(
            args.max_val_batches
        ),
    )

    if (
        train_df.empty
        or val_df.empty
    ):

        raise RuntimeError(
            "Generated utility rows "
            "are empty."
        )

    required = {
        "step",
        "q_target",
        "utility_gain",
        "path_b_helps",
        "m6_b_error",
        "b_only_b_error",
        "alpha_b",
        "path_b_scale",
        "path_b_raw_rms",
        "cap_active",
        "legacy__log1p_path_b_rms",
        "legacy__logRa",
        "legacy__logPr",
    }

    missing = (
        required.difference(
            train_df.columns
        )
        |
        required.difference(
            val_df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Missing expected columns: "
            f"{sorted(missing)}"
        )

    summary = (
        pd.concat(
            [
                train_df,
                val_df,
            ],
            ignore_index=True,
        )
        .groupby(
            [
                "split",
                "step",
            ],
            as_index=False,
        )
        .agg(
            count=(
                "path_b_helps",
                "size",
            ),
            help_fraction=(
                "path_b_helps",
                "mean",
            ),
            utility_gain_mean=(
                "utility_gain",
                "mean",
            ),
            utility_gain_std=(
                "utility_gain",
                "std",
            ),
            q_target_mean=(
                "q_target",
                "mean",
            ),
            q_target_std=(
                "q_target",
                "std",
            ),
            alpha_b_mean=(
                "alpha_b",
                "mean",
            ),
            path_b_raw_rms_mean=(
                "path_b_raw_rms",
                "mean",
            ),
            cap_active_fraction=(
                "cap_active",
                "mean",
            ),
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "CANONICAL UTILITY ROW SUMMARY"
    )

    print(
        "path_b_helps = 1 iff "
        "normalized-B RMSE(M6) > "
        "normalized-B RMSE(D2-B-only)"
    )

    print(
        "=" * 100
    )

    print(
        summary.to_string(
            index=False
        )
    )

    output_dir = (
        args.output_dir
        if os.path.isabs(
            args.output_dir
        )
        else os.path.join(
            PROJECT_ROOT,
            args.output_dir,
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    prefix = os.path.join(
        output_dir,
        args.run_name,
    )

    train_path = (
        prefix
        +
        "_train_rows.csv"
    )

    val_path = (
        prefix
        +
        "_val_rows.csv"
    )

    summary_path = (
        prefix
        +
        "_summary.csv"
    )

    metadata_path = (
        prefix
        +
        "_metadata.json"
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    val_df.to_csv(
        val_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    interface = payload[
        "dimensional_interface"
    ]

    stabilization = payload[
        "stabilization"
    ]

    metadata = {
        "experiment":
            (
                "D2-2a Canonical Path-B "
                "Utility-Row Regeneration"
            ),

        "stage":
            "control_preparation",

        "main_operator_training":
            False,

        "utility_classifier_training":
            False,

        "utility_gate_enabled":
            False,

        "test_split_accessed":
            False,

        "closed_m10_2a_generate_rows_reused_unchanged":
            True,

        "source_generator":
            ORIGINAL_PATH,

        "source_generator_sha256":
            source_sha,

        "utility_label":
            (
                "path_b_helps=1 iff "
                "RMSE_B(base_next,GT_next)-"
                "RMSE_B(D2_Bonly_next,GT_next)>0"
            ),

        "utility_metric_space":
            "normalized buoyancy RMSE",

        "utility_tie_rule":
            "strict >0; zero gain -> 0",

        "reliability_target":
            (
                "q=1/(1+"
                "RMS(PB_rollout_raw-PB_GT_raw)/"
                "(RMS(PB_GT_raw)+1e-6))"
            ),

        "split_label":
            args.split_label,

        "seed":
            args.seed,

        "split_sha256":
            split_sha,

        "stats_sha256":
            stats_sha,

        "m6_checkpoint_sha256":
            lock[
                "m6_sha"
            ],

        "d2_checkpoint":
            args.d2_checkpoint,

        "d2_checkpoint_sha256":
            d2_sha,

        "dimensional_interface": {
            "dx":
                float(
                    interface[
                        "dx"
                    ]
                ),

            "dy":
                float(
                    interface[
                        "dy"
                    ]
                ),

            "dt":
                float(
                    interface[
                        "dt"
                    ]
                ),

            "rate_to_increment":
                bool(
                    interface[
                        "rate_to_increment"
                    ]
                ),
        },

        "path_b_rms_cap":
            float(
                stabilization[
                    "path_b_rms_cap"
                ]
            ),

        "train_rows":
            len(
                train_df
            ),

        "val_rows":
            len(
                val_df
            ),

        "max_train_batches":
            args.max_train_batches,

        "max_val_batches":
            args.max_val_batches,
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
        train_path,
        val_path,
        summary_path,
        metadata_path,
    ]:

        print(
            " ",
            path,
        )

    print(
        "✅ D2-2a finished."
    )


if __name__ == "__main__":
    main()
