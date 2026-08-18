from __future__ import annotations

import argparse
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
    sys.path.insert(0, PROJECT_ROOT)


from datasets.rbc_dataset import RBCDataset
from training.metrics import FieldWiseRelativeL2Loss

from models.operators.fno2d_r3_pde_coupling import (
    R3PDECouplingFNO2d,
)

from models.operators.fno2d_r3_state_adaptive import (
    R3StateAdaptivePDECouplingFNO2d,
)

from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
)

from scripts.train_m6_fieldwise_encoder_h4 import (
    make_rollout_weights,
)

from scripts.train_r3_1_pde_coupling_h4 import (
    EXPECTED_FROZEN_M6_TENSORS,
    R3MultiStepParamDataset,
    autoregressive_multistep_loss_r3,
    build_probe_batch,
    evaluate,
    extract_m6_state,
    set_seed,
    sha256_file,
)


R3_2_CONTRACT_PATH = os.path.join(
    PROJECT_ROOT,
    "configs",
    "r3",
    "r3_2_state_adaptive_contract.json",
)

EXPECTED_ACTIVE_TERMS = (
    "buoyancy_advection",
    "buoyancy_forcing",
)

EXPECTED_TRAINABLE_NUMEL = 914

EXPECTED_TRAINABLE_NAMES = {
    "conditioner.net.0.weight",
    "conditioner.net.0.bias",
    "conditioner.net.2.weight",
    "conditioner.net.2.bias",
    "conditioner.net.4.weight",
    "conditioner.net.4.bias",
}

EXPECTED_FROZEN_RAW_ALPHA_NAMES = {
    "raw_alpha.buoyancy_advection",
    "raw_alpha.buoyancy_forcing",
}


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "R3-2 matched State-Adaptive Dimension-Valid "
            "Coupling H4 trainer."
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
        choices=(
            "unseen_pr",
            "unseen_ra",
        ),
    )

    parser.add_argument(
        "--parent_checkpoint",
        required=True,
        help=(
            "Split-matched frozen R3-1b Canonical "
            "best checkpoint."
        ),
    )

    parser.add_argument(
        "--m6_checkpoint",
        required=True,
        help=(
            "Independent audited M6 checkpoint used only "
            "to verify the M6 embedded in R3-1b."
        ),
    )

    parser.add_argument(
        "--adaptation_mode",
        required=True,
        choices=(
            "paramonly",
            "stateparam",
        ),
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
        "--checkpoint_dir",
        default="checkpoints/r3_2",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Run contract/data/parent/epoch0/gradient audits. "
            "No optimizer step and no checkpoint saving."
        ),
    )

    return parser.parse_args()


# ============================================================
# Path / state helpers
# ============================================================

def project_path(path):

    if os.path.isabs(path):
        return os.path.abspath(path)

    return os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            path,
        )
    )


def extract_model_state(checkpoint):

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):
        return checkpoint[
            "model_state_dict"
        ]

    if (
        isinstance(checkpoint, dict)
        and
        "state_dict" in checkpoint
    ):
        return checkpoint[
            "state_dict"
        ]

    if (
        isinstance(checkpoint, dict)
        and
        "model" in checkpoint
        and
        isinstance(
            checkpoint["model"],
            dict,
        )
    ):
        return checkpoint[
            "model"
        ]

    if (
        isinstance(checkpoint, dict)
        and checkpoint
        and all(
            torch.is_tensor(v)
            for v in checkpoint.values()
        )
    ):
        return checkpoint

    raise RuntimeError(
        "Could not locate model state_dict "
        "inside parent checkpoint."
    )


def assert_state_dict_exact(
    left,
    right,
    *,
    label,
):

    left_keys = list(
        left.keys()
    )

    right_keys = list(
        right.keys()
    )

    if left_keys != right_keys:
        raise RuntimeError(
            f"{label}: state_dict key mismatch."
        )

    for key in left_keys:

        if not torch.equal(
            left[key],
            right[key],
        ):
            max_abs = float(
                (
                    left[key]
                    -
                    right[key]
                )
                .abs()
                .max()
                .detach()
                .cpu()
            )

            raise RuntimeError(
                f"{label}: tensor mismatch "
                f"at {key!r}; "
                f"max_abs={max_abs:.12e}"
            )


# ============================================================
# Frozen R3-2 contract
# ============================================================

def verify_contract(args):

    if not os.path.exists(
        R3_2_CONTRACT_PATH
    ):
        raise FileNotFoundError(
            R3_2_CONTRACT_PATH
        )

    with open(
        R3_2_CONTRACT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        contract = json.load(f)

    if (
        contract.get(
            "contract_status"
        )
        !=
        "LOCKED_BEFORE_FORMAL_TRAINING"
    ):
        raise RuntimeError(
            "Unexpected R3-2 contract status."
        )

    parent_cfg = contract[
        "parent"
    ]

    if (
        parent_cfg[
            "required_parent_representation"
        ]
        !=
        "canonical"
    ):
        raise RuntimeError(
            "R3-2 parent must be canonical."
        )

    causal = contract[
        "causal_decomposition"
    ]

    if (
        causal[
            "primary_contrast"
        ]
        !=
        "R3-2b - R3-2a"
    ):
        raise RuntimeError(
            "R3-2 primary contrast changed."
        )

    if (
        causal[
            "primary_contrast_interpretation"
        ]
        !=
        "value of current-state information"
    ):
        raise RuntimeError(
            "R3-2 causal interpretation changed."
        )

    frozen_cfg = contract[
        "frozen_r3_1_structure"
    ]

    if set(
        frozen_cfg[
            "active_pde_terms"
        ]
    ) != set(
        EXPECTED_ACTIVE_TERMS
    ):
        raise RuntimeError(
            "R3-2 active PDE terms changed."
        )

    alpha_max_by_term = {
        term_name:
            float(
                frozen_cfg[
                    "formal_alpha_max_by_term"
                ][
                    term_name
                ]
            )
        for term_name
        in EXPECTED_ACTIVE_TERMS
    }

    if (
        frozen_cfg[
            "alpha_capacity_learnable"
        ]
        is not False
    ):
        raise RuntimeError(
            "R3-2 alpha capacity must remain fixed."
        )

    train_policy = contract[
        "trainable_parameter_policy"
    ]

    if (
        train_policy[
            "only_conditioner_trainable"
        ]
        is not True
    ):
        raise RuntimeError(
            "R3-2 only-conditioner-trainable "
            "policy changed."
        )

    formal = contract[
        "formal_training_protocol"
    ]

    locked_values = {
        "epochs":
            int(
                formal["epochs"]
            ),
        "batch_size":
            int(
                formal["batch_size"]
            ),
        "rollout_steps":
            int(
                formal["rollout_steps"]
            ),
        "seed":
            int(
                formal["seed"]
            ),
        "lr":
            float(
                formal["learning_rate"]
            ),
    }

    actual_values = {
        "epochs":
            int(args.epochs),
        "batch_size":
            int(args.batch_size),
        "rollout_steps":
            int(args.rollout_steps),
        "seed":
            int(args.seed),
        "lr":
            float(args.lr),
    }

    for key in locked_values:

        expected = locked_values[
            key
        ]

        actual = actual_values[
            key
        ]

        if isinstance(
            expected,
            float,
        ):
            matched = math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        else:
            matched = (
                actual
                ==
                expected
            )

        if not matched:
            raise RuntimeError(
                "R3-2 formal training protocol "
                f"mismatch for {key}: "
                f"expected={expected}, "
                f"actual={actual}"
            )

    data_cfg = contract[
        "data_protocol"
    ][
        args.split_label
    ]

    expected_split = project_path(
        data_cfg[
            "split"
        ]
    )

    expected_stats = project_path(
        data_cfg[
            "stats"
        ]
    )

    if (
        project_path(
            args.split
        )
        !=
        expected_split
    ):
        raise RuntimeError(
            "R3-2 split path differs from "
            "locked contract."
        )

    if (
        project_path(
            args.stats
        )
        !=
        expected_stats
    ):
        raise RuntimeError(
            "R3-2 stats path differs from "
            "locked contract."
        )

    parent_locked = contract[
        "parent_checkpoints"
    ][
        args.split_label
    ]

    expected_parent_path = (
        project_path(
            parent_locked[
                "checkpoint"
            ]
        )
    )

    if (
        project_path(
            args.parent_checkpoint
        )
        !=
        expected_parent_path
    ):
        raise RuntimeError(
            "R3-2 parent checkpoint path "
            "differs from locked contract."
        )

    test_embargo = contract[
        "test_embargo"
    ]

    if (
        test_embargo[
            "test_access_during_formal_training"
        ]
        is not False
    ):
        raise RuntimeError(
            "R3-2 TEST embargo changed."
        )

    return (
        contract,
        alpha_max_by_term,
        parent_locked,
    )


# ============================================================
# Parent checkpoint audit
# ============================================================

def audit_parent_checkpoint(
    *,
    checkpoint,
    checkpoint_sha,
    parent_locked,
    args,
    alpha_max_by_term,
):

    expected_sha = (
        parent_locked[
            "sha256"
        ]
    )

    if (
        checkpoint_sha
        !=
        expected_sha
    ):
        raise RuntimeError(
            "R3-1b parent checkpoint SHA256 "
            "does not match frozen R3-2 contract.\n"
            f"Expected={expected_sha}\n"
            f"Actual={checkpoint_sha}"
        )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Formal R3-1b checkpoint must "
            "contain metadata."
        )

    if (
        checkpoint.get(
            "stage"
        )
        !=
        "R3-1b"
    ):
        raise RuntimeError(
            "Parent checkpoint is not R3-1b."
        )

    if (
        checkpoint.get(
            "representation_mode"
        )
        !=
        "canonical"
    ):
        raise RuntimeError(
            "R3-2 parent checkpoint must "
            "use canonical representation."
        )

    if (
        checkpoint.get(
            "split_label"
        )
        !=
        args.split_label
    ):
        raise RuntimeError(
            "R3-2 parent checkpoint split "
            "does not match current run."
        )

    if (
        checkpoint.get(
            "test_accessed"
        )
        is not False
    ):
        raise RuntimeError(
            "Parent checkpoint provenance "
            "does not certify TEST embargo."
        )

    correction = checkpoint.get(
        "correction",
        {},
    )

    checkpoint_alpha = correction.get(
        "alpha_max_by_term"
    )

    if checkpoint_alpha is None:
        raise RuntimeError(
            "Parent checkpoint missing "
            "alpha_max_by_term metadata."
        )

    normalized_checkpoint_alpha = {
        key:
            float(value)
        for (
            key,
            value,
        )
        in checkpoint_alpha.items()
    }

    if (
        normalized_checkpoint_alpha
        !=
        alpha_max_by_term
    ):
        raise RuntimeError(
            "Parent alpha capacity differs "
            "from frozen R3-2 contract."
        )

    expected_best_epoch = int(
        parent_locked[
            "best_epoch"
        ]
    )

    if (
        int(
            checkpoint.get(
                "best_epoch",
                -1,
            )
        )
        !=
        expected_best_epoch
    ):
        raise RuntimeError(
            "Parent checkpoint best_epoch "
            "does not match frozen contract."
        )

    expected_best_val = float(
        parent_locked[
            "best_val_loss"
        ]
    )

    actual_best_val = float(
        checkpoint.get(
            "best_val_loss"
        )
    )

    if not math.isclose(
        actual_best_val,
        expected_best_val,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise RuntimeError(
            "Parent checkpoint best_val_loss "
            "does not match frozen contract."
        )


# ============================================================
# Parameter audit
# ============================================================

def audit_trainable_contract(
    model,
):

    trainable_names = (
        model.trainable_parameter_names()
    )

    trainable_numel = (
        model.trainable_parameter_count()
    )

    if (
        set(trainable_names)
        !=
        EXPECTED_TRAINABLE_NAMES
    ):
        raise RuntimeError(
            "Unexpected R3-2 trainable names:\n"
            f"{trainable_names}"
        )

    if (
        trainable_numel
        !=
        EXPECTED_TRAINABLE_NUMEL
    ):
        raise RuntimeError(
            "Unexpected R3-2 trainable numel.\n"
            f"Expected={EXPECTED_TRAINABLE_NUMEL}\n"
            f"Actual={trainable_numel}"
        )

    frozen_raw_names = set(
        model.frozen_parent_raw_alpha_names()
    )

    if (
        frozen_raw_names
        !=
        EXPECTED_FROZEN_RAW_ALPHA_NAMES
    ):
        raise RuntimeError(
            "Frozen parent raw-alpha names "
            "do not match R3-2 contract."
        )

    if any(
        p.requires_grad
        for p in model.raw_alpha.parameters()
    ):
        raise RuntimeError(
            "R3-1b parent raw_alpha is trainable."
        )

    if any(
        p.requires_grad
        for p in model.m6.parameters()
    ):
        raise RuntimeError(
            "M6 backbone is trainable."
        )

    frozen_m6_tensors = (
        model
        .frozen_m6_parameter_tensor_count()
    )

    if (
        frozen_m6_tensors
        !=
        EXPECTED_FROZEN_M6_TENSORS
    ):
        raise RuntimeError(
            "Unexpected frozen M6 tensor count.\n"
            f"Expected={EXPECTED_FROZEN_M6_TENSORS}\n"
            f"Actual={frozen_m6_tensors}"
        )

    return (
        trainable_names,
        trainable_numel,
        frozen_m6_tensors,
    )


# ============================================================
# Deterministic R3-1b parent reproduction probe
# ============================================================

def rms_per_sample(x):

    return torch.sqrt(
        torch.mean(
            x * x,
            dim=(-2, -1),
        )
        +
        1.0e-12
    )


def summarize_probe_r3_2(
    model,
    probe_batch,
    device,
    *,
    parent_model=None,
    require_parent_identity=False,
):

    model.eval()
    model.m6.eval()

    (
        context_norm,
        _,
        param,
    ) = probe_batch

    context_norm = (
        context_norm.to(device)
    )

    param = param.to(device)

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
            output,
            comp,
        ) = model(
            model_input,
            params=param,
            return_components=True,
        )

    full_features = (
        comp[
            "full_conditioner_features"
        ]
    )

    conditioner_input = (
        comp[
            "conditioner_input"
        ]
    )

    if (
        model.adaptation_mode
        ==
        "paramonly"
    ):

        if (
            torch.count_nonzero(
                conditioner_input[
                    :,
                    :8,
                ]
            ).item()
            !=
            0
        ):
            raise RuntimeError(
                "R3-2a ParamOnly state features "
                "are not exact zero."
            )

        if not torch.equal(
            conditioner_input[
                :,
                8:,
            ],
            full_features[
                :,
                8:,
            ],
        ):
            raise RuntimeError(
                "R3-2a parameter features changed "
                "during state masking."
            )

    elif (
        model.adaptation_mode
        ==
        "stateparam"
    ):

        if not torch.equal(
            conditioner_input,
            full_features,
        ):
            raise RuntimeError(
                "R3-2b StateParam must expose "
                "the full feature vector."
            )

    else:
        raise RuntimeError(
            "Unexpected adaptation mode."
        )

    if not torch.equal(
        full_features[
            :,
            8:,
        ],
        param.to(
            device=full_features.device,
            dtype=full_features.dtype,
        ),
    ):
        raise RuntimeError(
            "R3-2 conditioner parameter features "
            "do not equal [logRa,logPr]."
        )

    delta_raw = (
        comp[
            "delta_raw"
        ]
    )

    summary = {
        "adaptation_mode":
            model.adaptation_mode,

        "representation_mode":
            "canonical",

        "trainable_parameter_count":
            model.trainable_parameter_count(),

        "delta_raw_abs_max":
            float(
                delta_raw
                .abs()
                .max()
                .detach()
                .cpu()
            ),

        "delta_raw_rms":
            float(
                torch.sqrt(
                    torch.mean(
                        delta_raw
                        *
                        delta_raw
                    )
                    +
                    1.0e-12
                )
                .detach()
                .cpu()
            ),

        "state_feature_abs_mean":
            float(
                full_features[
                    :,
                    :8,
                ]
                .abs()
                .mean()
                .detach()
                .cpu()
            ),

        "conditioner_state_feature_abs_mean":
            float(
                conditioner_input[
                    :,
                    :8,
                ]
                .abs()
                .mean()
                .detach()
                .cpu()
            ),

        "parent_identity_required":
            bool(
                require_parent_identity
            ),
    }

    for term_name in (
        model.ACTIVE_TERMS
    ):

        parent_alpha = (
            comp[
                "parent_gate_values"
            ][
                term_name
            ]
        )

        effective_alpha = (
            comp[
                "effective_gate_values"
            ][
                term_name
            ]
        )

        signal = (
            comp[
                "compiled_terms"
            ][
                term_name
            ]
        )

        signal_rms = (
            rms_per_sample(
                signal
            )
        )

        summary[
            f"{term_name}_parent_alpha"
        ] = float(
            parent_alpha
            .detach()
            .cpu()
        )

        summary[
            f"{term_name}_effective_alpha_mean"
        ] = float(
            effective_alpha
            .mean()
            .detach()
            .cpu()
        )

        summary[
            f"{term_name}_effective_alpha_min"
        ] = float(
            effective_alpha
            .min()
            .detach()
            .cpu()
        )

        summary[
            f"{term_name}_effective_alpha_max"
        ] = float(
            effective_alpha
            .max()
            .detach()
            .cpu()
        )

        summary[
            f"{term_name}_signal_rms_mean"
        ] = float(
            signal_rms
            .mean()
            .detach()
            .cpu()
        )

    if parent_model is None:

        if require_parent_identity:
            raise RuntimeError(
                "Parent model required for "
                "epoch-0 identity audit."
            )

        summary[
            "output_minus_parent_max_abs"
        ] = None

        return summary

    parent_model.eval()
    parent_model.m6.eval()

    with torch.no_grad():

        (
            parent_output,
            parent_comp,
        ) = parent_model(
            model_input,
            params=param,
            return_components=True,
        )

    max_abs = float(
        (
            output
            -
            parent_output
        )
        .abs()
        .max()
        .detach()
        .cpu()
    )

    summary[
        "output_minus_parent_max_abs"
    ] = max_abs

    if require_parent_identity:

        if not torch.equal(
            output,
            parent_output,
        ):
            raise RuntimeError(
                "R3-2 epoch-0 output does not "
                "exactly reproduce R3-1b parent.\n"
                f"max_abs={max_abs:.12e}"
            )

        if (
            torch.count_nonzero(
                delta_raw
            ).item()
            !=
            0
        ):
            raise RuntimeError(
                "R3-2 epoch-0 delta_raw "
                "must be exact zero."
            )

        if not torch.equal(
            comp[
                "base_delta_norm"
            ],
            parent_comp[
                "base_delta_norm"
            ],
        ):
            raise RuntimeError(
                "R3-2 epoch-0 M6 base delta "
                "differs from R3-1b."
            )

        if not torch.equal(
            comp[
                "physics_residual_norm"
            ],
            parent_comp[
                "physics_residual_norm"
            ],
        ):
            raise RuntimeError(
                "R3-2 epoch-0 physics residual "
                "differs from R3-1b."
            )

        for term_name in (
            model.ACTIVE_TERMS
        ):

            if not torch.equal(
                comp[
                    "compiled_terms"
                ][
                    term_name
                ],
                parent_comp[
                    "compiled_terms"
                ][
                    term_name
                ],
            ):
                raise RuntimeError(
                    "R3-2 epoch-0 canonical signal "
                    f"differs for {term_name}."
                )

            if not torch.equal(
                comp[
                    "term_corrections"
                ][
                    term_name
                ],
                parent_comp[
                    "term_corrections"
                ][
                    term_name
                ],
            ):
                raise RuntimeError(
                    "R3-2 epoch-0 term correction "
                    f"differs for {term_name}."
                )

            parent_alpha = (
                parent_comp[
                    "gate_values"
                ][
                    term_name
                ]
            )

            effective_alpha = (
                comp[
                    "effective_gate_values"
                ][
                    term_name
                ]
            )

            if not torch.equal(
                effective_alpha,
                parent_alpha.expand_as(
                    effective_alpha
                ),
            ):
                raise RuntimeError(
                    "R3-2 epoch-0 effective gate "
                    f"differs for {term_name}."
                )

    return summary


# ============================================================
# Deterministic one-batch gradient audit
#
# IMPORTANT:
# A separate deterministic batch is built directly from the
# dataset so the shuffled formal train-loader RNG is untouched.
# ============================================================

def build_gradient_audit_batch(
    dataset,
    batch_size,
):

    n = min(
        int(batch_size),
        len(dataset),
    )

    if n <= 0:
        raise RuntimeError(
            "Cannot build gradient audit "
            "from empty TRAIN dataset."
        )

    contexts = []
    y_seqs = []
    params = []

    for idx in range(n):

        (
            context_norm,
            y_seq_norm,
            param,
        ) = dataset[idx]

        contexts.append(
            context_norm
        )

        y_seqs.append(
            y_seq_norm
        )

        params.append(
            param
        )

    return (
        torch.stack(
            contexts,
            dim=0,
        ),
        torch.stack(
            y_seqs,
            dim=0,
        ),
        torch.stack(
            params,
            dim=0,
        ),
    )


def run_gradient_audit(
    *,
    model,
    train_dataset,
    criterion,
    rollout_weights,
    device,
    batch_size,
):

    (
        context_norm,
        y_seq_norm,
        param,
    ) = build_gradient_audit_batch(
        train_dataset,
        batch_size,
    )

    context_norm = (
        context_norm.to(device)
    )

    y_seq_norm = (
        y_seq_norm.to(device)
    )

    param = param.to(device)

    model.zero_grad(
        set_to_none=True
    )

    model.train()
    model.m6.eval()

    (
        loss,
        _,
    ) = autoregressive_multistep_loss_r3(
        model=model,
        context_norm=context_norm,
        y_seq_norm=y_seq_norm,
        param=param,
        criterion=criterion,
        rollout_weights=rollout_weights,
    )

    if not torch.isfinite(
        loss
    ):
        raise RuntimeError(
            "Gradient audit produced "
            "non-finite loss."
        )

    loss.backward()

    conditioner_missing_grad = []

    conditioner_grad_stats = {}

    forbidden_grad_names = []

    for (
        name,
        parameter,
    ) in model.named_parameters():

        grad = parameter.grad

        if name.startswith(
            "conditioner."
        ):

            if grad is None:
                conditioner_missing_grad.append(
                    name
                )
                continue

            if not torch.isfinite(
                grad
            ).all():
                raise RuntimeError(
                    "Non-finite conditioner gradient "
                    f"for {name}."
                )

            conditioner_grad_stats[
                name
            ] = {
                "max_abs":
                    float(
                        grad
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    ),

                "l2":
                    float(
                        torch.linalg.vector_norm(
                            grad
                        )
                        .detach()
                        .cpu()
                    ),
            }

        else:

            if grad is not None:
                forbidden_grad_names.append(
                    name
                )

    if conditioner_missing_grad:
        raise RuntimeError(
            "Some conditioner parameters have "
            "grad=None during R3-2 gradient audit:\n"
            f"{conditioner_missing_grad}"
        )

    if forbidden_grad_names:
        raise RuntimeError(
            "Gradient leaked outside conditioner:\n"
            f"{forbidden_grad_names}"
        )

    final_names = (
        "conditioner.net.4.weight",
        "conditioner.net.4.bias",
    )

    final_signal = sum(
        conditioner_grad_stats[
            name
        ][
            "max_abs"
        ]
        for name in final_names
    )

    # Because the final layer is zero-initialized, gradients of
    # earlier conditioner layers may be exactly zero on this FIRST
    # backward. The final layer itself must receive real signal.
    if final_signal <= 0.0:
        raise RuntimeError(
            "Zero-initialized R3-2 final layer "
            "received no gradient signal."
        )

    raw_alpha_grad_present = any(
        parameter.grad is not None
        for parameter
        in model.raw_alpha.parameters()
    )

    m6_grad_present = any(
        parameter.grad is not None
        for parameter
        in model.m6.parameters()
    )

    if raw_alpha_grad_present:
        raise RuntimeError(
            "Frozen parent raw_alpha received grad."
        )

    if m6_grad_present:
        raise RuntimeError(
            "Frozen M6 received grad."
        )

    summary = {
        "loss":
            float(
                loss
                .detach()
                .cpu()
            ),

        "conditioner_missing_grad":
            conditioner_missing_grad,

        "forbidden_grad_names":
            forbidden_grad_names,

        "raw_alpha_grad_present":
            raw_alpha_grad_present,

        "m6_grad_present":
            m6_grad_present,

        "final_layer_gradient_signal":
            float(
                final_signal
            ),

        "conditioner_gradients":
            conditioner_grad_stats,
    }

    model.zero_grad(
        set_to_none=True
    )

    model.eval()
    model.m6.eval()

    return summary


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
    initial_val,
    parent_val_loss,
    parent_val_reproduction_abs_diff,
    args,
    contract_sha,
    split_sha,
    stats_sha,
    parent_sha,
    m6_sha,
    trainable_names,
    trainable_numel,
    frozen_m6_tensors,
    probe_summary,
    gradient_audit_summary,
):

    stage = (
        "R3-2a"
        if
        args.adaptation_mode
        ==
        "paramonly"
        else
        "R3-2b"
    )

    role = (
        "capacity_matched_paramonly_control"
        if
        args.adaptation_mode
        ==
        "paramonly"
        else
        "stateparam_formal_candidate"
    )

    payload = {
        "experiment":
            "R3-2-Matched-State-Adaptive-"
            "Dimension-Valid-Coupling-H4",

        "stage":
            stage,

        "adaptation_mode":
            args.adaptation_mode,

        "representation_mode":
            "canonical",

        "role":
            role,

        "primary_contrast":
            "R3-2b - R3-2a",

        "primary_contrast_interpretation":
            "value of current-state information",

        "epoch":
            int(epoch),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            (
                None
                if optimizer is None
                else
                optimizer.state_dict()
            ),

        "val_loss":
            float(val_loss),

        "best_val_loss":
            float(best_val_loss),

        "best_epoch":
            int(best_epoch),

        "initial_val":
            float(initial_val),

        "parent_val_loss":
            float(parent_val_loss),

        "parent_val_reproduction_abs_diff":
            float(
                parent_val_reproduction_abs_diff
            ),

        "architecture_contract":
            R3_2_CONTRACT_PATH,

        "architecture_contract_sha256":
            contract_sha,

        "parent_stage":
            "R3-1b",

        "parent_representation":
            "canonical",

        "parent_checkpoint":
            args.parent_checkpoint,

        "parent_checkpoint_sha256":
            parent_sha,

        "m6_checkpoint":
            args.m6_checkpoint,

        "m6_checkpoint_sha256":
            m6_sha,

        "backbone":
            "audited_frozen_M6_"
            "FieldWiseEncoder_H4",

        "backbone_frozen":
            True,

        "parent_raw_alpha_frozen":
            True,

        "active_terms":
            list(
                model.ACTIVE_TERMS
            ),

        "alpha_max_by_term": {
            term_name:
                float(
                    model.alpha_max_for_term(
                        term_name
                    )
                )
            for term_name
            in model.ACTIVE_TERMS
        },

        "gate_parameterization":
            (
                "alpha_j(s_t,Ra,Pr)="
                "alpha_max_j*tanh("
                "raw_alpha_parent_j+delta_raw_j)"
            ),

        "conditioner": {
            "architecture":
                "10-32-16-2_GELU",

            "input_dim":
                10,

            "hidden_dims":
                [32, 16],

            "output_dim":
                2,

            "final_layer_zero_initialized":
                True,

            "trainable_numel":
                EXPECTED_TRAINABLE_NUMEL,

            "state_features_visible":
                (
                    args.adaptation_mode
                    ==
                    "stateparam"
                ),

            "param_features_visible":
                True,

            "paramonly_state_mask":
                (
                    "features[0:8]=exact_zero"
                    if
                    args.adaptation_mode
                    ==
                    "paramonly"
                    else
                    None
                ),
        },

        "disabled": {
            "utility_gate":
                True,

            "utility_labels":
                True,

            "rmscap":
                True,

            "pde_loss":
                True,

            "new_pde_terms":
                True,

            "learnable_spatial_phi":
                True,

            "horizon_conditioning":
                True,

            "temporal_d1_d2_features":
                True,
        },

        "prediction_type":
            "normalized_finite_step_delta",

        "training_type":
            "H4_free_autoregressive",

        "rollout_steps":
            int(
                args.rollout_steps
            ),

        "rollout_weights": [
            float(x)
            for x
            in make_rollout_weights(
                args.rollout_steps
            ).tolist()
        ],

        "epochs":
            int(args.epochs),

        "batch_size":
            int(args.batch_size),

        "learning_rate":
            float(args.lr),

        "optimizer":
            "Adam",

        "weight_decay":
            0.0,

        "seed":
            int(args.seed),

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

        "trainable_parameter_names":
            trainable_names,

        "trainable_parameter_count":
            int(
                trainable_numel
            ),

        "frozen_parent_raw_alpha_names":
            sorted(
                EXPECTED_FROZEN_RAW_ALPHA_NAMES
            ),

        "frozen_m6_parameter_tensors":
            int(
                frozen_m6_tensors
            ),

        "initial_probe_summary":
            probe_summary,

        "gradient_audit":
            gradient_audit_summary,

        "test_accessed":
            False,
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

    (
        contract,
        alpha_max_by_term,
        parent_locked,
    ) = verify_contract(
        args
    )

    for path in (
        args.split,
        args.stats,
        args.parent_checkpoint,
        args.m6_checkpoint,
        R3_2_CONTRACT_PATH,
    ):

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
        else
        "cpu"
    )

    stage = (
        "R3-2a PARAMONLY CONTROL"
        if
        args.adaptation_mode
        ==
        "paramonly"
        else
        "R3-2b STATEPARAM FORMAL CANDIDATE"
    )

    print(
        "=" * 108
    )

    print(
        "R3-2 MATCHED STATE-ADAPTIVE "
        "DIMENSION-VALID COUPLING H4"
    )

    print(
        "=" * 108
    )

    print(
        "Stage:",
        stage,
    )

    print(
        "Adaptation mode:",
        args.adaptation_mode,
    )

    print(
        "Representation: canonical"
    )

    print(
        "Parent: frozen split-matched R3-1b"
    )

    print(
        "M6 frozen: True"
    )

    print(
        "Parent raw_alpha frozen: True"
    )

    print(
        "Only conditioner trainable: True"
    )

    print(
        "Trainable conditioner numel:",
        EXPECTED_TRAINABLE_NUMEL,
    )

    print(
        "Utility Gate: OFF"
    )

    print(
        "RMSCap: OFF"
    )

    print(
        "PDE loss: OFF"
    )

    print(
        "New PDE paths: OFF"
    )

    print(
        "TEST access: FORBIDDEN"
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Dry run:",
        args.dry_run,
    )

    contract_sha = sha256_file(
        R3_2_CONTRACT_PATH
    )

    split_sha = sha256_file(
        args.split
    )

    stats_sha = sha256_file(
        args.stats
    )

    parent_sha = sha256_file(
        args.parent_checkpoint
    )

    m6_sha = sha256_file(
        args.m6_checkpoint
    )

    print()
    print(
        "========== PROVENANCE =========="
    )

    print(
        "CONTRACT_SHA256:",
        contract_sha,
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
        "PARENT_R3_1B_SHA256:",
        parent_sha,
    )

    print(
        "M6_CHECKPOINT_SHA256:",
        m6_sha,
    )

    # --------------------------------------------------------
    # Parent checkpoint
    # --------------------------------------------------------

    parent_checkpoint = torch.load(
        args.parent_checkpoint,
        map_location=device,
    )

    audit_parent_checkpoint(
        checkpoint=parent_checkpoint,
        checkpoint_sha=parent_sha,
        parent_locked=parent_locked,
        args=args,
        alpha_max_by_term=(
            alpha_max_by_term
        ),
    )

    parent_state = (
        extract_model_state(
            parent_checkpoint
        )
    )

    # --------------------------------------------------------
    # Data: TRAIN + VAL ONLY
    # --------------------------------------------------------

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    train_base = RBCDataset(
        split_config=split["train"],
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=args.rollout_steps,
        return_params=True,
    )

    val_base = RBCDataset(
        split_config=split["val"],
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=args.rollout_steps,
        return_params=True,
    )

    train_dataset = (
        R3MultiStepParamDataset(
            train_base,
            context_length=4,
        )
    )

    val_dataset = (
        R3MultiStepParamDataset(
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
        len(train_dataset),
    )

    print(
        "VAL_SAMPLES:",
        len(val_dataset),
    )

    (
        sample_context,
        sample_y_seq,
        sample_param,
    ) = train_dataset[0]

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

    probe_batch = build_probe_batch(
        val_dataset,
        probe_size=12,
    )

    # --------------------------------------------------------
    # Canonical metadata
    # --------------------------------------------------------

    metadata = (
        build_rbc_canonical_metadata(
            args.stats
        )
    )

    # --------------------------------------------------------
    # Build independent frozen R3-1b parent
    # --------------------------------------------------------

    parent_model = (
        R3PDECouplingFNO2d(
            canonical_metadata=metadata,
            representation_mode="canonical",
            alpha_max_by_term=(
                alpha_max_by_term
            ),
            freeze_m6=True,
        )
        .to(device)
    )

    parent_model.load_state_dict(
        parent_state,
        strict=True,
    )

    parent_model.eval()
    parent_model.m6.eval()

    # --------------------------------------------------------
    # Independently verify embedded M6.
    # --------------------------------------------------------

    m6_checkpoint = torch.load(
        args.m6_checkpoint,
        map_location=device,
    )

    independent_m6_state = (
        extract_m6_state(
            m6_checkpoint
        )
    )

    assert_state_dict_exact(
        parent_model.m6.state_dict(),
        independent_m6_state,
        label=(
            "R3-1b embedded M6 "
            "vs independent audited M6"
        ),
    )

    print()
    print(
        "✅ Parent R3-1b embedded M6 "
        "exactly matches audited M6"
    )

    # --------------------------------------------------------
    # Build R3-2 child.
    #
    # Reset seed immediately before construction so separate
    # ParamOnly / StateParam formal jobs obtain the same
    # conditioner initialization under seed 42.
    # --------------------------------------------------------

    set_seed(
        args.seed
    )

    model = (
        R3StateAdaptivePDECouplingFNO2d(
            canonical_metadata=metadata,
            adaptation_mode=(
                args.adaptation_mode
            ),
            alpha_max_by_term=(
                alpha_max_by_term
            ),
        )
        .to(device)
    )

    model.load_parent_r3_1b_state_dict(
        parent_state
    )

    # --------------------------------------------------------
    # Parameter audit
    # --------------------------------------------------------

    (
        trainable_names,
        trainable_numel,
        frozen_m6_tensors,
    ) = audit_trainable_contract(
        model
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
        "FROZEN_PARENT_RAW_ALPHA:",
        sorted(
            EXPECTED_FROZEN_RAW_ALPHA_NAMES
        ),
    )

    print(
        "FROZEN_M6_PARAMETER_TENSORS:",
        frozen_m6_tensors,
    )

    # --------------------------------------------------------
    # Exact epoch-0 parent identity audit
    # --------------------------------------------------------

    probe_summary = (
        summarize_probe_r3_2(
            model,
            probe_batch,
            device,
            parent_model=(
                parent_model
            ),
            require_parent_identity=True,
        )
    )

    print()
    print(
        "========== EPOCH-0 / PARENT AUDIT =========="
    )

    print(
        json.dumps(
            probe_summary,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "✅ R3-2 epoch0 exactly reproduces "
        "frozen R3-1b parent"
    )

    # --------------------------------------------------------
    # Shared criterion / H4 weights
    # --------------------------------------------------------

    criterion = (
        FieldWiseRelativeL2Loss()
    )

    rollout_weights = (
        make_rollout_weights(
            args.rollout_steps
        )
        .to(device)
    )

    # --------------------------------------------------------
    # One-batch backward audit.
    #
    # NO optimizer step.
    # Separate deterministic batch -> formal train shuffle
    # generator is untouched.
    # --------------------------------------------------------

    gradient_audit_summary = (
        run_gradient_audit(
            model=model,
            train_dataset=(
                train_dataset
            ),
            criterion=criterion,
            rollout_weights=(
                rollout_weights
            ),
            device=device,
            batch_size=args.batch_size,
        )
    )

    print()
    print(
        "========== 1-BATCH GRADIENT AUDIT =========="
    )

    print(
        json.dumps(
            gradient_audit_summary,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "✅ Conditioner gradients present"
    )

    print(
        "✅ Final zero-init layer receives "
        "nonzero gradient signal"
    )

    print(
        "✅ Parent raw_alpha grad = None"
    )

    print(
        "✅ M6 grad = None"
    )

    # --------------------------------------------------------
    # Dry-run stops BEFORE optimization / checkpoint.
    # --------------------------------------------------------

    if args.dry_run:

        print()
        print(
            "✅ R3-2 DRY-RUN PASS"
        )

        print(
            "No optimizer step performed."
        )

        print(
            "No checkpoint written."
        )

        print(
            "TEST was not accessed."
        )

        return

    # --------------------------------------------------------
    # Formal training
    # --------------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.lr,
        weight_decay=0.0,
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

    # --------------------------------------------------------
    # Epoch 0:
    #
    # R3-2 exactly equals split-matched frozen R3-1b.
    #
    # Therefore initial VAL must reproduce the parent
    # checkpoint's own best-checkpoint validation value.
    # --------------------------------------------------------

    initial_val = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        rollout_weights=(
            rollout_weights
        ),
        device=device,
        max_batches=None,
    )

    parent_val_loss = float(
        parent_checkpoint[
            "val_loss"
        ]
    )

    parent_val_reproduction_abs_diff = abs(
        initial_val
        -
        parent_val_loss
    )

    if (
        parent_val_reproduction_abs_diff
        >
        1.0e-10
    ):
        raise RuntimeError(
            "R3-2 epoch-0 VAL does not "
            "reproduce frozen R3-1b checkpoint.\n"
            f"R3-2 initial={initial_val:.12f}\n"
            f"R3-1b parent={parent_val_loss:.12f}\n"
            f"abs_diff="
            f"{parent_val_reproduction_abs_diff:.12e}"
        )

    print()
    print(
        "========== EPOCH 0 / R3-1b PARENT START =========="
    )

    print(
        "R3_2_INITIAL_VAL:",
        f"{initial_val:.12f}",
    )

    print(
        "R3_1B_PARENT_VAL:",
        f"{parent_val_loss:.12f}",
    )

    print(
        "VAL_REPRODUCTION_ABS_DIFF:",
        f"{parent_val_reproduction_abs_diff:.12e}",
    )

    print(
        "✅ Epoch-0 VAL exactly reproduces "
        "the frozen parent within tolerance"
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
        best_val_loss=(
            best_val_loss
        ),
        best_epoch=best_epoch,
        initial_val=initial_val,
        parent_val_loss=(
            parent_val_loss
        ),
        parent_val_reproduction_abs_diff=(
            parent_val_reproduction_abs_diff
        ),
        args=args,
        contract_sha=contract_sha,
        split_sha=split_sha,
        stats_sha=stats_sha,
        parent_sha=parent_sha,
        m6_sha=m6_sha,
        trainable_names=(
            trainable_names
        ),
        trainable_numel=(
            trainable_numel
        ),
        frozen_m6_tensors=(
            frozen_m6_tensors
        ),
        probe_summary=(
            probe_summary
        ),
        gradient_audit_summary=(
            gradient_audit_summary
        ),
    )

    # Parent no longer needed after exact epoch-0 audit.
    del parent_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Epochs 1..20
    # --------------------------------------------------------

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

            (
                context_norm,
                y_seq_norm,
                param,
            ) = batch

            context_norm = (
                context_norm.to(device)
            )

            y_seq_norm = (
                y_seq_norm.to(device)
            )

            param = param.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                loss,
                _,
            ) = (
                autoregressive_multistep_loss_r3(
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
                    "Non-finite R3-2 loss at "
                    f"epoch={epoch}, "
                    f"batch={batch_idx}: "
                    f"{loss}"
                )

            loss.backward()

            # Gradient containment must remain true.
            if any(
                parameter.grad
                is not None
                for parameter
                in model.raw_alpha.parameters()
            ):
                raise RuntimeError(
                    "Frozen parent raw_alpha received "
                    "gradient during formal training."
                )

            if any(
                parameter.grad
                is not None
                for parameter
                in model.m6.parameters()
            ):
                raise RuntimeError(
                    "Frozen M6 received gradient "
                    "during formal training."
                )

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()

            bs = (
                context_norm.shape[0]
            )

            train_loss_sum += (
                float(
                    loss.item()
                )
                *
                bs
            )

            train_samples += bs

        if train_samples == 0:
            raise RuntimeError(
                "Training loader produced "
                "zero samples."
            )

        train_loss = (
            train_loss_sum
            /
            train_samples
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            rollout_weights=(
                rollout_weights
            ),
            device=device,
            max_batches=None,
        )

        epoch_probe_summary = (
            summarize_probe_r3_2(
                model,
                probe_batch,
                device,
                parent_model=None,
                require_parent_identity=False,
            )
        )

        improved = (
            val_loss
            <
            best_val_loss
        )

        if improved:

            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

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
                initial_val=(
                    initial_val
                ),
                parent_val_loss=(
                    parent_val_loss
                ),
                parent_val_reproduction_abs_diff=(
                    parent_val_reproduction_abs_diff
                ),
                args=args,
                contract_sha=(
                    contract_sha
                ),
                split_sha=split_sha,
                stats_sha=stats_sha,
                parent_sha=parent_sha,
                m6_sha=m6_sha,
                trainable_names=(
                    trainable_names
                ),
                trainable_numel=(
                    trainable_numel
                ),
                frozen_m6_tensors=(
                    frozen_m6_tensors
                ),
                probe_summary=(
                    epoch_probe_summary
                ),
                gradient_audit_summary=(
                    gradient_audit_summary
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
            best_epoch=best_epoch,
            initial_val=initial_val,
            parent_val_loss=(
                parent_val_loss
            ),
            parent_val_reproduction_abs_diff=(
                parent_val_reproduction_abs_diff
            ),
            args=args,
            contract_sha=contract_sha,
            split_sha=split_sha,
            stats_sha=stats_sha,
            parent_sha=parent_sha,
            m6_sha=m6_sha,
            trainable_names=(
                trainable_names
            ),
            trainable_numel=(
                trainable_numel
            ),
            frozen_m6_tensors=(
                frozen_m6_tensors
            ),
            probe_summary=(
                epoch_probe_summary
            ),
            gradient_audit_summary=(
                gradient_audit_summary
            ),
        )

        adv_alpha = (
            epoch_probe_summary[
                "buoyancy_advection_"
                "effective_alpha_mean"
            ]
        )

        forcing_alpha = (
            epoch_probe_summary[
                "buoyancy_forcing_"
                "effective_alpha_mean"
            ]
        )

        delta_raw_rms = (
            epoch_probe_summary[
                "delta_raw_rms"
            ]
        )

        print(
            f"[Epoch {epoch:02d}/"
            f"{args.epochs:02d}] "
            f"train={train_loss:.12f} "
            f"val={val_loss:.12f} "
            f"delta_raw_rms="
            f"{delta_raw_rms:.6e} "
            f"alpha_adv_mean="
            f"{adv_alpha:+.6e} "
            f"alpha_force_mean="
            f"{forcing_alpha:+.6e} "
            f"best_epoch={best_epoch}"
            +
            (
                "  ✅ BEST"
                if improved
                else
                ""
            )
        )

    print()
    print(
        "========== R3-2 COMPLETE =========="
    )

    print(
        "ADAPTATION_MODE:",
        args.adaptation_mode,
    )

    print(
        "INITIAL_R3_1B_VAL:",
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

    print(
        "TEST_ACCESS:",
        "FORBIDDEN / NOT ACCESSED",
    )


if __name__ == "__main__":
    main()
