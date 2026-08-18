import argparse
import csv
import json
import math
import os
import sys

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


from datasets.rbc_dataset import RBCDataset
from training.metrics import FieldWiseRelativeL2Loss

from models.operators.fno2d_d2_1_dimconsistent_bonly import (
    D21DimConsistentBOnlyFNO2d,
)

from scripts.train_m10_2_bonly_rmscap_h4 import (
    M10MultiStepParamDataset,
    autoregressive_multistep_loss_m10,
    build_field_stats,
    build_probe_batch,
    evaluate,
    extract_m6_state,
    set_seed,
    sha256_file,
    summarize_gate_probe,
)

from scripts.train_m6_fieldwise_encoder_h4 import (
    make_rollout_weights,
)


# ============================================================
# Locked D2-1 contract
# ============================================================

DX_CANONICAL = 1.0 / 64.0
DY_CANONICAL = 1.0 / 63.0
DT_CANONICAL = 0.25

EXPECTED_TRAINABLE_NUMEL = 386
EXPECTED_FROZEN_M6_TENSORS = 38


LOCKED_D2_SUPPORT = {
    "unseen_pr": {
        "cap":
            5.3869654465e-01,

        "split_basename":
            "unseen_pr_split.json",

        "stats_basename":
            "rbc_field_stats_unseen_pr.json",

        "calibration_csv": (
            "outputs/tables/"
            "d2_0_canonical_pathb_support_unseen_pr.csv"
        ),
    },

    "unseen_ra": {
        "cap":
            3.9047183691e-01,

        "split_basename":
            "unseen_ra_split.json",

        "stats_basename":
            "rbc_field_stats_unseen_ra.json",

        "calibration_csv": (
            "outputs/tables/"
            "d2_0_canonical_pathb_support_unseen_ra.csv"
        ),
    },
}


PARENT_D2_1_TRAINER = (
    "scripts/train_d2_1_dimconsistent_bonly_h4.py"
)

PARENT_D2_1_TRAINER_SHA256 = (
    "446bd05b3d886e0e291ef0881ce426c090327a491f7279c03fc18c22cf38eb6a"
)

LEGACY_ALPHA_MAX = 0.25

LEGACY_PATH_B_RMS_CAP = {
    "unseen_pr": 2.180076360160806,
    "unseen_ra": 1.5787383958464702,
}

LOCKED_CAPACITY_MATCH_ALPHA = {
    "unseen_pr": 1.0117367476234869,
    "unseen_ra": 1.0107889011534232,
}


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "D2-3A capacity-matched canonical Path-B "
            "StateParam H4 causal control."
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
        "--split_label",
        required=True,
        choices=[
            "unseen_pr",
            "unseen_ra",
        ],
    )

    parser.add_argument(
        "--run_name",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--alpha_max",
        type=float,
        default=None,
        help=(
            "Locked split-specific capacity-matched alpha_max. "
            "Omit to use the predeclared value; any supplied value "
            "must match it exactly."
        ),
    )

    parser.add_argument(
        "--conditioner_hidden",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--path_b_rms_eps",
        type=float,
        default=1e-12,
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
        "--checkpoint_dir",
        default="checkpoints/d2_3a_capacity_match",
    )

    return parser.parse_args()


# ============================================================
# D2-0 support verification
# ============================================================

def verify_locked_contract(
    args,
):
    config = LOCKED_D2_SUPPORT[
        args.split_label
    ]

    if (
        os.path.basename(
            args.split
        )
        != config[
            "split_basename"
        ]
    ):
        raise RuntimeError(
            "split mismatch.\n"
            f"Expected: {config['split_basename']}\n"
            f"Actual: {args.split}"
        )

    if (
        os.path.basename(
            args.stats
        )
        != config[
            "stats_basename"
        ]
    ):
        raise RuntimeError(
            "stats mismatch.\n"
            f"Expected: {config['stats_basename']}\n"
            f"Actual: {args.stats}"
        )

    calibration_csv = config[
        "calibration_csv"
    ]

    if not os.path.exists(
        calibration_csv
    ):
        raise FileNotFoundError(
            calibration_csv
        )

    with open(
        calibration_csv,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(
                f
            )
        )

    overall_rows = [
        row
        for row in rows
        if row.get(
            "scope"
        )
        == "overall"
    ]

    if len(
        overall_rows
    ) != 1:
        raise RuntimeError(
            "D2-0 CSV must contain exactly "
            "one scope=overall row."
        )

    overall = overall_rows[
        0
    ]

    if (
        overall.get(
            "split_label"
        )
        != args.split_label
    ):
        raise RuntimeError(
            "D2-0 calibration "
            "split_label mismatch."
        )

    csv_cap = float(
        overall[
            "canonical_max"
        ]
    )

    locked_cap = float(
        config[
            "cap"
        ]
    )

    if not math.isclose(
        csv_cap,
        locked_cap,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError(
            "D2-0 cap mismatch.\n"
            f"Locked: {locked_cap:.12e}\n"
            f"CSV:    {csv_cap:.12e}"
        )

    return {
        "cap":
            locked_cap,

        "calibration_csv":
            calibration_csv,

        "calibration_sha256":
            sha256_file(
                calibration_csv
            ),
    }


# ============================================================
# Epoch-0 interface audit
# ============================================================

def epoch0_interface_audit(
    model,
    probe_batch,
    device,
):
    model.eval()
    model.m6.eval()

    (
        context_norm,
        _,
        param,
    ) = probe_batch

    context_norm = (
        context_norm.to(
            device
        )
    )

    param = param.to(
        device
    )

    (
        batch_size,
        context_len,
        channels,
        h,
        w,
    ) = context_norm.shape

    model_input = (
        context_norm.reshape(
            batch_size,
            context_len * channels,
            h,
            w,
        )
    )

    with torch.no_grad():

        (
            d2_delta,
            components,
        ) = model(
            model_input,
            params=param,
            return_components=True,
        )

        m6_delta = model.m6(
            model_input
        )

    # --------------------------------------------------------
    # Epoch 0 must be EXACTLY the M6 rebase apart from tiny
    # floating-point roundoff because alpha_B == 0.
    # --------------------------------------------------------

    epoch0_m6_diff = float(
        (
            d2_delta
            -
            m6_delta
        )
        .abs()
        .max()
        .cpu()
    )

    if epoch0_m6_diff > 1e-7:
        raise RuntimeError(
            "Epoch-0 D2 output must equal pure M6.\n"
            f"max_abs_diff="
            f"{epoch0_m6_diff:.12e}"
        )

    # --------------------------------------------------------
    # Explicit rate -> increment audit.
    # --------------------------------------------------------

    path_b_increment = components[
        "path_b_increment_norm_raw"
    ]

    path_b_rate = components[
        "path_b_rate_norm_raw_canonical_grid"
    ]

    interface_error = float(
        (
            path_b_increment
            -
            DT_CANONICAL
            * path_b_rate
        )
        .abs()
        .max()
        .cpu()
    )

    if interface_error > 1e-6:
        raise RuntimeError(
            "Canonical interface failed.\n"
            "Expected: P_increment = dt * P_rate\n"
            f"max_abs_error="
            f"{interface_error:.12e}"
        )

    return {
        "epoch0_output_minus_m6_max_abs":
            epoch0_m6_diff,

        "rate_to_increment_max_abs_error":
            interface_error,

        "path_b_rms_raw_mean":
            float(
                components[
                    "path_b_rms_raw"
                ]
                .mean()
                .cpu()
            ),

        "path_b_rms_raw_max":
            float(
                components[
                    "path_b_rms_raw"
                ]
                .max()
                .cpu()
            ),

        "path_b_cap_active_fraction":
            float(
                components[
                    "path_b_cap_active"
                ]
                .float()
                .mean()
                .cpu()
            ),

        "path_b_scale_min":
            float(
                components[
                    "path_b_scale"
                ]
                .min()
                .cpu()
            ),
    }


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    epoch,
    val_loss,
    best_val_loss,
    best_epoch,
    args,
    cap,
    split_sha,
    stats_sha,
    m6_sha,
    calibration_info,
    trainable_names,
    trainable_numel,
    gate_summary,
    interface_audit,
):
    payload = {
        "experiment": (
            "D2-3A-CapacityMatched-"
            "DimConsistent-StateParam-BOnly-H4"
        ),

        "stage": (
            "causal_control_"
            "capacity_matched_path_b"
        ),

        "parent_experiment": (
            "D2-1-DimConsistent-"
            "StateParam-BOnly-H4"
        ),

        "comparison_target": (
            "D2-1 canonical alpha_max=0.25 "
            "and legacy M10-2 B-only"
        ),

        "factorial_mode":
            "stateparam",

        "epoch":
            int(
                epoch
            ),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict": (
            None
            if optimizer is None
            else optimizer.state_dict()
        ),

        "val_loss":
            float(
                val_loss
            ),

        "best_val_loss":
            float(
                best_val_loss
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        # ----------------------------------------------------
        # Backbone / topology
        # ----------------------------------------------------

        "backbone": (
            "audited_frozen_"
            "M6_FieldWiseEncoder_H4"
        ),

        "backbone_frozen":
            True,

        "physics_topology": {
            "path_a_output_injection":
                False,

            "buoyancy_anomaly_state_feature":
                True,

            "path_b": (
                "dt*(-u_dot_grad_b)/"
                "buoyancy_std -> "
                "normalized finite-step delta"
            ),

            "path_b_only_output_physics":
                True,

            "direct_ux_path":
                False,

            "direct_uy_path":
                False,

            "direct_pressure_path":
                False,
        },

        # ----------------------------------------------------
        # D2 dimensional interface
        # ----------------------------------------------------

        "dimensional_interface": {
            "status":
                "canonical_finite_step",

            "dx":
                DX_CANONICAL,

            "dy":
                DY_CANONICAL,

            "dt":
                DT_CANONICAL,

            "rate_to_increment":
                True,

            "path_b_formula": (
                "dt/std_b * "
                "(-u_x*d_x(b)-u_y*d_y(b))"
            ),

            "state_conditioner_uses_"
            "canonical_raw_path_b":
                True,

            "path_a_state_feature_"
            "unchanged_from_m10_2":
                True,

            "source": (
                "D0/D1 audited RBC "
                "representation interface"
            ),
        },

        # ----------------------------------------------------
        # StateParam
        # ----------------------------------------------------

        "conditioner": {
            "mode":
                "stateparam",

            "hidden_dim":
                int(
                    args.conditioner_hidden
                ),

            "parameter_format": (
                "[log10(Ra), log10(Pr)]"
            ),

            "dynamic_output_zero_initialized":
                True,

            "path_b_summary_representation": (
                "canonical finite-step raw Path-B"
            ),
        },

        "gate_parameterization": (
            "alpha_max*tanh("
            "base_raw+tanh("
            "conditioner(condition)))"
        ),

        "alpha_max":
            float(
                args.alpha_max
            ),

        "capacity_match_control": {
            "control_type":
                "max_injection_capacity_match",

            "legacy_alpha_max":
                LEGACY_ALPHA_MAX,

            "legacy_path_b_rms_cap":
                float(
                    LEGACY_PATH_B_RMS_CAP[
                        args.split_label
                    ]
                ),

            "canonical_path_b_rms_cap":
                float(
                    cap
                ),

            "matched_alpha_max":
                float(
                    args.alpha_max
                ),

            "capacity_equation": (
                "alpha_match*C_dc = "
                "legacy_alpha_max*C_old"
            ),

            "legacy_max_capacity":
                float(
                    LEGACY_ALPHA_MAX
                    * LEGACY_PATH_B_RMS_CAP[
                        args.split_label
                    ]
                ),

            "canonical_max_capacity":
                float(
                    args.alpha_max
                    * cap
                ),

            "parent_d2_1_trainer":
                PARENT_D2_1_TRAINER,

            "parent_d2_1_trainer_sha256":
                PARENT_D2_1_TRAINER_SHA256,

            "val_tuned":
                False,
        },

        # ----------------------------------------------------
        # Canonical RMSCap
        # ----------------------------------------------------

        "stabilization": {
            "target": (
                "canonical Path-B "
                "residual injection only"
            ),

            "type":
                "per_sample_spatial_rms_cap",

            "support_source": (
                "D2-0 split-specific "
                "TRAIN-ONLY full H4 "
                "GT support"
            ),

            "path_b_rms_cap":
                float(
                    cap
                ),

            "path_b_rms_eps":
                float(
                    args.path_b_rms_eps
                ),

            "state_conditioner_uses_"
            "canonical_raw_path_b":
                True,

            "calibration_csv":
                calibration_info[
                    "calibration_csv"
                ],

            "calibration_sha256":
                calibration_info[
                    "calibration_sha256"
                ],
        },

        # ----------------------------------------------------
        # Grid / time
        # ----------------------------------------------------

        "dx":
            DX_CANONICAL,

        "dy":
            DY_CANONICAL,

        "dt":
            DT_CANONICAL,

        # ----------------------------------------------------
        # Training protocol
        # ----------------------------------------------------

        "prediction_type":
            "normalized_delta",

        "delta_definition": (
            "pred_next_norm="
            "current_state_norm+"
            "pred_delta_norm"
        ),

        "training_type":
            "H4 free-autoregressive",

        "loss": (
            "weighted H4 "
            "free-autoregressive "
            "FieldWiseRelativeL2Loss"
        ),

        "pde_loss":
            False,

        "utility_gate":
            False,

        "rollout_steps":
            int(
                args.rollout_steps
            ),

        "rollout_weights": [
            float(
                x
            )
            for x in make_rollout_weights(
                args.rollout_steps
            ).tolist()
        ],

        "epochs":
            int(
                args.epochs
            ),

        "batch_size":
            int(
                args.batch_size
            ),

        "learning_rate":
            float(
                args.lr
            ),

        "optimizer":
            "Adam",

        "weight_decay":
            0.0,

        "seed":
            int(
                args.seed
            ),

        # ----------------------------------------------------
        # Provenance
        # ----------------------------------------------------

        "split_label":
            args.split_label,

        "split":
            args.split,

        "split_sha256":
            split_sha,

        "stats":
            args.stats,

        "stats_sha256":
            stats_sha,

        "m6_checkpoint":
            args.m6_checkpoint,

        "m6_checkpoint_sha256":
            m6_sha,

        # ----------------------------------------------------
        # Audit
        # ----------------------------------------------------

        "trainable_parameter_names":
            trainable_names,

        "trainable_parameter_count":
            int(
                trainable_numel
            ),

        "frozen_m6_parameter_tensors":
            EXPECTED_FROZEN_M6_TENSORS,

        "epoch0_interface_audit":
            interface_audit,

        "gate_probe_summary":
            gate_summary,
    }

    torch.save(
        payload,
        path,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    expected_alpha = (
        LOCKED_CAPACITY_MATCH_ALPHA[
            args.split_label
        ]
    )

    if args.alpha_max is None:
        args.alpha_max = expected_alpha
    elif not math.isclose(
        float(args.alpha_max),
        float(expected_alpha),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "D2-3A alpha_max is predeclared and locked.\n"
            f"split={args.split_label}\n"
            f"expected={expected_alpha:.16f}\n"
            f"actual={args.alpha_max:.16f}"
        )

    parent_trainer_path = os.path.join(
        PROJECT_ROOT,
        PARENT_D2_1_TRAINER,
    )

    if not os.path.exists(
        parent_trainer_path
    ):
        raise FileNotFoundError(
            parent_trainer_path
        )

    parent_trainer_sha = sha256_file(
        parent_trainer_path
    )

    if (
        parent_trainer_sha
        !=
        PARENT_D2_1_TRAINER_SHA256
    ):
        raise RuntimeError(
            "Parent D2-1 trainer SHA mismatch.\n"
            f"expected={PARENT_D2_1_TRAINER_SHA256}\n"
            f"actual={parent_trainer_sha}"
        )

    if args.rollout_steps != 4:
        raise ValueError(
            "D2-3A is locked to "
            "rollout_steps=4."
        )

    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.lr <= 0
    ):
        raise ValueError(
            "epochs, batch_size and lr "
            "must be positive."
        )

    if args.path_b_rms_eps <= 0:
        raise ValueError(
            "path_b_rms_eps must be positive."
        )

    for path in [
        args.split,
        args.stats,
        args.m6_checkpoint,
    ]:
        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                path
            )

    calibration_info = (
        verify_locked_contract(
            args
        )
    )

    cap = calibration_info[
        "cap"
    ]

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "========================================"
    )
    print(
        "D2-3A CAPACITY-MATCHED "
        "CANONICAL PATH-B CONTROL"
    )
    print(
        "========================================"
    )

    print(
        "Stage: CONTROL / CAUSAL ATTRIBUTION"
    )
    print(
        "Training: ON"
    )
    print(
        "Utility Gate: OFF"
    )
    print(
        "PDE loss: OFF"
    )
    print(
        "TEST access: NOT USED"
    )
    print(
        "Mode: stateparam"
    )
    print(
        "Device:",
        device,
    )
    print(
        "Seed:",
        args.seed,
    )
    print(
        "dx:",
        DX_CANONICAL,
    )
    print(
        "dy:",
        DY_CANONICAL,
    )
    print(
        "dt:",
        DT_CANONICAL,
    )
    print(
        "Canonical RMSCap:",
        cap,
    )
    print(
        "Conditioner sees canonical RAW Path-B: True"
    )
    print(
        "alpha_max:",
        args.alpha_max,
    )
    print(
        "Legacy Path-B cap:",
        LEGACY_PATH_B_RMS_CAP[
            args.split_label
        ],
    )
    print(
        "Matched max capacity:",
        args.alpha_max * cap,
    )
    print(
        "Parent D2-1 trainer SHA:",
        parent_trainer_sha,
    )

    # ========================================================
    # Provenance
    # ========================================================

    split_sha = sha256_file(
        args.split
    )

    stats_sha = sha256_file(
        args.stats
    )

    m6_sha = sha256_file(
        args.m6_checkpoint
    )

    print()
    print(
        "========== PROVENANCE =========="
    )
    print(
        "SPLIT_SHA256:",
        split_sha,
    )
    print(
        "STATS_SHA256:",
        stats_sha,
    )
    print(
        "M6_CHECKPOINT_SHA256:",
        m6_sha,
    )
    print(
        "D2_CALIBRATION_SHA256:",
        calibration_info[
            "calibration_sha256"
        ],
    )

    # ========================================================
    # Data
    # ========================================================

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(
            f
        )

    (
        field_mean,
        field_std,
    ) = build_field_stats(
        args.stats
    )

    train_base = RBCDataset(
        split_config=split[
            "train"
        ],
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=args.rollout_steps,
        return_params=True,
    )

    val_base = RBCDataset(
        split_config=split[
            "val"
        ],
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=args.rollout_steps,
        return_params=True,
    )

    train_dataset = (
        M10MultiStepParamDataset(
            train_base,
            context_length=4,
        )
    )

    val_dataset = (
        M10MultiStepParamDataset(
            val_base,
            context_length=4,
        )
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
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
        sample_context,
        sample_y_seq,
        sample_param,
    ) = train_dataset[
        0
    ]

    print(
        "CONTEXT_SHAPE:",
        tuple(
            sample_context.shape
        ),
    )
    print(
        "Y_SEQ_SHAPE:",
        tuple(
            sample_y_seq.shape
        ),
    )
    print(
        "PARAM_SHAPE:",
        tuple(
            sample_param.shape
        ),
    )

    probe_batch = (
        build_probe_batch(
            val_dataset,
            probe_size=12,
        )
    )

    # ========================================================
    # Model
    # ========================================================

    set_seed(
        args.seed
    )

    model = (
        D21DimConsistentBOnlyFNO2d(
            field_mean=field_mean,
            field_std=field_std,
            dx=DX_CANONICAL,
            dy=DY_CANONICAL,
            dt=DT_CANONICAL,
            path_b_rms_cap=cap,
            path_b_rms_eps=(
                args.path_b_rms_eps
            ),
            alpha_max=(
                args.alpha_max
            ),
            conditioner_hidden=(
                args.conditioner_hidden
            ),
            freeze_m6=True,
        )
        .to(
            device
        )
    )

    m6_checkpoint = torch.load(
        args.m6_checkpoint,
        map_location=device,
    )

    model.load_m6_state_dict(
        extract_m6_state(
            m6_checkpoint
        )
    )

    # ========================================================
    # Parameter audit
    # ========================================================

    trainable_names = (
        model.trainable_parameter_names()
    )

    trainable_numel = (
        model.trainable_parameter_count()
    )

    if (
        trainable_numel
        != EXPECTED_TRAINABLE_NUMEL
    ):
        raise RuntimeError(
            "Unexpected trainable parameter count.\n"
            f"Expected: "
            f"{EXPECTED_TRAINABLE_NUMEL}\n"
            f"Actual: "
            f"{trainable_numel}"
        )

    frozen_m6_tensors = (
        model.frozen_m6_parameter_tensor_count()
    )

    if (
        frozen_m6_tensors
        != EXPECTED_FROZEN_M6_TENSORS
    ):
        raise RuntimeError(
            "Unexpected frozen M6 tensor count.\n"
            f"Expected: "
            f"{EXPECTED_FROZEN_M6_TENSORS}\n"
            f"Actual: "
            f"{frozen_m6_tensors}"
        )

    m6_trainable = [
        name
        for (
            name,
            parameter,
        )
        in model.m6.named_parameters()
        if parameter.requires_grad
    ]

    if m6_trainable:
        raise RuntimeError(
            "M6 is not fully frozen: "
            f"{m6_trainable}"
        )

    print()
    print(
        "========== PARAMETER AUDIT =========="
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
        "FROZEN_M6_PARAMETER_TENSORS:",
        frozen_m6_tensors,
    )

    # ========================================================
    # Epoch-0 interface audit
    # ========================================================

    interface_audit = (
        epoch0_interface_audit(
            model,
            probe_batch,
            device,
        )
    )

    print()
    print(
        "========== EPOCH-0 INTERFACE AUDIT =========="
    )

    print(
        json.dumps(
            interface_audit,
            indent=2,
            sort_keys=True,
        )
    )

    # ========================================================
    # Training objects
    # ========================================================

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    criterion = (
        FieldWiseRelativeL2Loss()
    )

    rollout_weights = (
        make_rollout_weights(
            args.rollout_steps
        )
        .to(
            device
        )
    )

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.lr,
    )

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True,
    )

    best_path = os.path.join(
        args.checkpoint_dir,
        f"{args.run_name}_best.pth",
    )

    last_path = os.path.join(
        args.checkpoint_dir,
        f"{args.run_name}_last.pth",
    )

    # ========================================================
    # Epoch 0 = pure M6
    # ========================================================

    initial_val = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        rollout_weights=rollout_weights,
        device=device,
        max_batches=(
            args.max_val_batches
        ),
    )

    gate_summary = (
        summarize_gate_probe(
            model,
            probe_batch,
            device,
        )
    )

    best_val_loss = (
        initial_val
    )

    best_epoch = 0

    save_checkpoint(
        best_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        val_loss=initial_val,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        args=args,
        cap=cap,
        split_sha=split_sha,
        stats_sha=stats_sha,
        m6_sha=m6_sha,
        calibration_info=(
            calibration_info
        ),
        trainable_names=(
            trainable_names
        ),
        trainable_numel=(
            trainable_numel
        ),
        gate_summary=(
            gate_summary
        ),
        interface_audit=(
            interface_audit
        ),
    )

    print()
    print(
        "========== EPOCH 0 / PURE-M6 START =========="
    )

    print(
        f"VAL_LOSS="
        f"{initial_val:.12f}"
    )

    print(
        "GATE_PROBE:",
        json.dumps(
            gate_summary,
            sort_keys=True,
        ),
    )

    print(
        "BEST_EPOCH=0"
    )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()
        model.m6.eval()

        train_loss_sum = 0.0
        train_samples = 0

        for (
            batch_idx,
            batch,
        ) in enumerate(
            train_loader
        ):
            if (
                args.max_train_batches
                is not None
                and batch_idx
                >= args.max_train_batches
            ):
                break

            (
                context_norm,
                y_seq_norm,
                param,
            ) = batch

            context_norm = (
                context_norm.to(
                    device
                )
            )

            y_seq_norm = (
                y_seq_norm.to(
                    device
                )
            )

            param = param.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                loss,
                _,
            ) = (
                autoregressive_multistep_loss_m10(
                    model=model,
                    context_norm=(
                        context_norm
                    ),
                    y_seq_norm=(
                        y_seq_norm
                    ),
                    param=param,
                    criterion=criterion,
                    rollout_weights=(
                        rollout_weights
                    ),
                )
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite loss "
                    f"at epoch={epoch}, "
                    f"batch={batch_idx}: "
                    f"{loss}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = (
                context_norm.shape[
                    0
                ]
            )

            train_loss_sum += (
                float(
                    loss.item()
                )
                * batch_size
            )

            train_samples += (
                batch_size
            )

        if train_samples == 0:
            raise RuntimeError(
                "Training loader produced "
                "zero samples."
            )

        train_loss = (
            train_loss_sum
            / train_samples
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            rollout_weights=(
                rollout_weights
            ),
            device=device,
            max_batches=(
                args.max_val_batches
            ),
        )

        gate_summary = (
            summarize_gate_probe(
                model,
                probe_batch,
                device,
            )
        )

        improved = (
            val_loss
            < best_val_loss
        )

        if improved:
            best_val_loss = (
                val_loss
            )

            best_epoch = (
                epoch
            )

            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                best_val_loss=(
                    best_val_loss
                ),
                best_epoch=(
                    best_epoch
                ),
                args=args,
                cap=cap,
                split_sha=split_sha,
                stats_sha=stats_sha,
                m6_sha=m6_sha,
                calibration_info=(
                    calibration_info
                ),
                trainable_names=(
                    trainable_names
                ),
                trainable_numel=(
                    trainable_numel
                ),
                gate_summary=(
                    gate_summary
                ),
                interface_audit=(
                    interface_audit
                ),
            )

        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            best_val_loss=(
                best_val_loss
            ),
            best_epoch=(
                best_epoch
            ),
            args=args,
            cap=cap,
            split_sha=split_sha,
            stats_sha=stats_sha,
            m6_sha=m6_sha,
            calibration_info=(
                calibration_info
            ),
            trainable_names=(
                trainable_names
            ),
            trainable_numel=(
                trainable_numel
            ),
            gate_summary=(
                gate_summary
            ),
            interface_audit=(
                interface_audit
            ),
        )

        print(
            f"[Epoch "
            f"{epoch:02d}/"
            f"{args.epochs:02d}] "
            f"train={train_loss:.12f} "
            f"val={val_loss:.12f} "
            f"B="
            f"{gate_summary['alpha_b_mean']:+.6e}"
            f"±"
            f"{gate_summary['alpha_b_std']:.2e} "
            f"dyn="
            f"{gate_summary['dynamic_raw_abs_mean']:.3e} "
            f"capFrac="
            f"{gate_summary['path_b_cap_active_fraction']:.3f} "
            f"rawBmax="
            f"{gate_summary['path_b_rms_raw_max']:.3e} "
            f"safeBmax="
            f"{gate_summary['path_b_rms_safe_max']:.3e} "
            f"minScale="
            f"{gate_summary['path_b_scale_min']:.3e} "
            f"best_epoch="
            f"{best_epoch}"
            + (
                "  ✅ BEST"
                if improved
                else ""
            )
        )

    # ========================================================
    # Complete
    # ========================================================

    print()
    print(
        "========== D2-3A COMPLETE =========="
    )

    print(
        "EXPERIMENT:",
        (
            "D2-3A-CapacityMatched-"
            "DimConsistent-StateParam-BOnly-H4"
        ),
    )

    print(
        "SPLIT_LABEL:",
        args.split_label,
    )

    print(
        "INITIAL_VAL:",
        f"{initial_val:.12f}",
    )

    print(
        "BEST_VAL:",
        f"{best_val_loss:.12f}",
    )

    print(
        "BEST_EPOCH:",
        best_epoch,
    )

    print(
        "BEST_CHECKPOINT:",
        best_path,
    )

    print(
        "LAST_CHECKPOINT:",
        last_path,
    )


if __name__ == "__main__":
    main()
