from __future__ import annotations

import argparse
import json
import math
import os
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


from training.normalization import (
    FieldWiseNormalizer,
)

from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
)

from models.operators.fno2d_r3_state_adaptive import (
    R3StateAdaptivePDECouplingFNO2d,
)

# ------------------------------------------------------------
# Reuse the frozen R3-1 formal rollout infrastructure.
#
# We deliberately DO NOT rewrite:
#   - TEST window construction
#   - physical-space normalization / denormalization
#   - aggregate Rel-L2 / MSE accumulation
#   - full H1-H16 curve
#   - global growth / AUC
#
# Only the R3-2 model loader, matched contrasts and
# per-sample adaptive-gate statistics are new.
# ------------------------------------------------------------

from evaluation.evaluate_cross_param_rollout_r3_1 import (
    RolloutDataset,
    assert_same_m6,
    build_m6,
    build_r3,
    compare_alpha_maps,
    error_rows,
    load_checkpoint,
    load_locked_contract,
    make_global_growth_table,
    resolve_path,
    sha256_file,
    update_error_stats,
)


# ============================================================
# Frozen R3-2 evaluation identities
# ============================================================

DISPLAY_CANONICAL = (
    "R3-1b-Canonical"
)

DISPLAY_PARAMONLY = (
    "R3-2a-ParamOnly"
)

DISPLAY_STATEPARAM = (
    "R3-2b-StateParam"
)

MODEL_ORDER = (
    DISPLAY_CANONICAL,
    DISPLAY_PARAMONLY,
    DISPLAY_STATEPARAM,
)

ACTIVE_TERMS = (
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


# ============================================================
# Frozen contracts / registry
# ============================================================

R3_0_CONTRACT = (
    "configs/r3/"
    "r3_0_architecture_contract.json"
)

R3_2_CONTRACT = (
    "configs/r3/"
    "r3_2_state_adaptive_contract.json"
)

R3_2_REGISTRY = (
    "configs/r3/"
    "r3_2_pretest_checkpoint_registry.json"
)

EXPECTED_R3_0_CONTRACT_SHA256 = (
    "7ba1809042b2b97ed3c1ec7dab299337"
    "c339f54d57a913086db5fefbaedc79fe"
)


# ============================================================
# Split-specific resources frozen BEFORE R3-2 TEST
# ============================================================

LOCKED_RESOURCES = {

    "unseen_pr": {

        "split":
            "data/splits/"
            "unseen_pr_split.json",

        "split_sha256":
            "ca4f1707c913c880b33398bdf17906ae"
            "70ea4dba77662a76d12de5e319cf86be",

        "stats":
            "data/stats/"
            "rbc_field_stats_unseen_pr.json",

        "stats_sha256":
            "299828fb4c986f54493a552fdea8871e"
            "114fc6dd0b756c45dde0117340b54233",

        "m6":
            "checkpoints/cross_param/"
            "m6_fieldwise_encoder_h4_"
            "unseen_pr_best.pth",

        "m6_sha256":
            "f7a91e4661703127d90c5a7e7ae68645"
            "6f96ef3525e720dd605f422047001cab",

        "r3_1b":
            "checkpoints/r3_1/"
            "r3_1b_canonical_h4_"
            "unseen_pr_best.pth",
    },

    "unseen_ra": {

        "split":
            "data/splits/"
            "unseen_ra_split.json",

        "split_sha256":
            "475d3092bb9d0ad16f023088446419b5"
            "651b1186d369ccfe0019c841f1fd8e36",

        "stats":
            "data/stats/"
            "rbc_field_stats_unseen_ra.json",

        "stats_sha256":
            "a96b1a01cf25d7b9910e01abd4de5672"
            "078e6ec3f6a6dda8a96a19e1a022d5af",

        "m6":
            "checkpoints/cross_param/"
            "m6_fieldwise_encoder_h4_"
            "unseen_ra_best.pth",

        "m6_sha256":
            "3d0c1571dcc57e65b1cad45fbdcae72e"
            "2b737249033d379426dc616a6c414e53",

        "r3_1b":
            "checkpoints/r3_1/"
            "r3_1b_canonical_h4_"
            "unseen_ra_best.pth",
    },
}


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "R3-2 matched formal TEST rollout: "
            "R3-1b Canonical vs "
            "R3-2a ParamOnly vs "
            "R3-2b StateParam."
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
            "Audit registry/checkpoints/"
            "model interfaces only. "
            "Does NOT construct or access "
            "the TEST rollout dataset."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/tables/"
            "r3_2_rollout"
        ),
    )

    return parser.parse_args()


# ============================================================
# Basic helpers
# ============================================================

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


def audit_locked_file(
    path,
    expected_sha,
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

    if actual != expected_sha:

        raise RuntimeError(
            f"{label} SHA256 mismatch.\n"
            f"Expected={expected_sha}\n"
            f"Actual={actual}\n"
            f"Path={path}"
        )

    return actual


# ============================================================
# Pre-TEST registry audit
# ============================================================

def load_and_audit_registry(
    split_label,
):

    registry_path = resolve_path(
        R3_2_REGISTRY
    )

    contract_path = resolve_path(
        R3_2_CONTRACT
    )

    if not os.path.exists(
        registry_path
    ):
        raise FileNotFoundError(
            registry_path
        )

    if not os.path.exists(
        contract_path
    ):
        raise FileNotFoundError(
            contract_path
        )

    registry = load_json(
        registry_path
    )

    if (
        registry.get("stage")
        !=
        "R3-2"
    ):
        raise RuntimeError(
            "Unexpected R3-2 registry stage."
        )

    if (
        registry.get(
            "registry_status"
        )
        !=
        "LOCKED_BEFORE_FORMAL_TEST"
    ):
        raise RuntimeError(
            "R3-2 pre-TEST registry "
            "is not locked."
        )

    policy = (
        registry[
            "selection_policy"
        ]
    )

    if (
        policy[
            "selection_source"
        ]
        !=
        "validation_only"
    ):
        raise RuntimeError(
            "R3-2 checkpoint selection "
            "is not VAL-only."
        )

    if (
        policy[
            "test_used_for_model_selection"
        ]
        is not False
    ):
        raise RuntimeError(
            "TEST was used for "
            "R3-2 model selection."
        )

    if (
        policy[
            "test_accessed_during_training"
        ]
        is not False
    ):
        raise RuntimeError(
            "TEST was accessed during "
            "R3-2 training."
        )

    primary = (
        registry[
            "primary_formal_contrast"
        ]
    )

    if (
        primary[
            "comparison"
        ]
        !=
        "R3-2b - R3-2a"
    ):
        raise RuntimeError(
            "R3-2 primary formal "
            "contrast changed."
        )

    actual_contract_sha = (
        sha256_file(
            contract_path
        )
    )

    frozen_contract_sha = (
        registry[
            "frozen_code_provenance"
        ][
            "r3_2_contract_sha256"
        ]
    )

    if (
        actual_contract_sha
        !=
        frozen_contract_sha
    ):
        raise RuntimeError(
            "R3-2 contract SHA differs "
            "from pre-TEST registry."
        )

    entries = (
        registry[
            "formal_checkpoints"
        ][
            split_label
        ]
    )

    for stage in (
        "R3-2a",
        "R3-2b",
    ):

        entry = entries[
            stage
        ]

        checkpoint_path = (
            resolve_path(
                entry[
                    "checkpoint"
                ]
            )
        )

        audit_locked_file(
            checkpoint_path,
            entry[
                "sha256"
            ],
            (
                f"{split_label}/"
                f"{stage}"
            ),
        )

        if (
            entry[
                "trainable_parameter_count"
            ]
            !=
            914
        ):
            raise RuntimeError(
                f"{split_label}/{stage}: "
                "registry trainable "
                "numel changed."
            )

        if (
            entry[
                "test_accessed"
            ]
            is not False
        ):
            raise RuntimeError(
                f"{split_label}/{stage}: "
                "registry says TEST accessed."
            )

    return (
        registry,
        entries,
        registry_path,
        contract_path,
        actual_contract_sha,
    )


# ============================================================
# R3-2 checkpoint metadata audit
# ============================================================

def audit_r3_2_checkpoint(
    payload,
    *,
    expected_stage,
    expected_mode,
    split_label,
    expected_entry,
    r3_2_contract_sha,
    alpha_map,
    path,
):

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "R3-2 checkpoint lacks "
            "metadata payload: "
            f"{path}"
        )

    checks = {

        "stage":
            expected_stage,

        "adaptation_mode":
            expected_mode,

        "representation_mode":
            "canonical",

        "split_label":
            split_label,

        "seed":
            42,

        "trainable_parameter_count":
            914,

        "test_accessed":
            False,

        "architecture_contract_sha256":
            r3_2_contract_sha,

        "parent_checkpoint_sha256":
            expected_entry[
                "parent_r3_1b_sha256"
            ],
    }

    for key, expected in (
        checks.items()
    ):

        actual = payload.get(
            key
        )

        if actual != expected:

            raise RuntimeError(
                "R3-2 checkpoint metadata "
                f"mismatch: {path}\n"
                f"key={key}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            )

    if (
        int(
            payload.get(
                "epoch",
                -1,
            )
        )
        !=
        int(
            expected_entry[
                "best_epoch"
            ]
        )
    ):
        raise RuntimeError(
            "R3-2 checkpoint epoch "
            "differs from registry: "
            f"{path}"
        )

    if (
        int(
            payload.get(
                "best_epoch",
                -1,
            )
        )
        !=
        int(
            expected_entry[
                "best_epoch"
            ]
        )
    ):
        raise RuntimeError(
            "R3-2 best_epoch differs "
            "from registry: "
            f"{path}"
        )

    for key in (
        "val_loss",
        "best_val_loss",
    ):

        actual = float(
            payload[
                key
            ]
        )

        expected = float(
            expected_entry[
                "best_val_loss"
            ]
        )

        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise RuntimeError(
                f"R3-2 {key} differs "
                "from registry: "
                f"{path}"
            )

    stored_alpha = (
        payload.get(
            "alpha_max_by_term"
        )
    )

    if stored_alpha is None:
        raise RuntimeError(
            "R3-2 checkpoint missing "
            "alpha_max_by_term: "
            f"{path}"
        )

    compare_alpha_maps(
        stored_alpha,
        alpha_map,
        path=path,
    )

    if (
        payload.get(
            "primary_contrast"
        )
        !=
        "R3-2b - R3-2a"
    ):
        raise RuntimeError(
            "R3-2 primary contrast "
            "metadata mismatch: "
            f"{path}"
        )


# ============================================================
# R3-2 model builder
# ============================================================

def build_r3_2(
    *,
    adaptation_mode,
    checkpoint_path,
    expected_entry,
    metadata,
    alpha_map,
    r3_2_contract_sha,
    split_label,
    device,
):

    model = (
        R3StateAdaptivePDECouplingFNO2d(
            canonical_metadata=(
                metadata
            ),
            adaptation_mode=(
                adaptation_mode
            ),
            alpha_max_by_term=(
                alpha_map
            ),
        )
        .to(device)
    )

    (
        payload,
        state,
    ) = load_checkpoint(
        checkpoint_path,
        device,
    )

    expected_stage = (
        "R3-2a"
        if
        adaptation_mode
        ==
        "paramonly"
        else
        "R3-2b"
    )

    audit_r3_2_checkpoint(
        payload,
        expected_stage=(
            expected_stage
        ),
        expected_mode=(
            adaptation_mode
        ),
        split_label=(
            split_label
        ),
        expected_entry=(
            expected_entry
        ),
        r3_2_contract_sha=(
            r3_2_contract_sha
        ),
        alpha_map=(
            alpha_map
        ),
        path=(
            checkpoint_path
        ),
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()
    model.m6.eval()

    if (
        model.trainable_parameter_count()
        !=
        EXPECTED_TRAINABLE_NUMEL
    ):
        raise RuntimeError(
            f"{expected_stage}: "
            "trainable numel != 914"
        )

    if (
        set(
            model
            .trainable_parameter_names()
        )
        !=
        EXPECTED_TRAINABLE_NAMES
    ):
        raise RuntimeError(
            f"{expected_stage}: "
            "trainable topology changed."
        )

    if any(
        parameter.requires_grad
        for parameter
        in model.raw_alpha.parameters()
    ):
        raise RuntimeError(
            f"{expected_stage}: "
            "frozen parent raw_alpha "
            "is trainable."
        )

    if any(
        parameter.requires_grad
        for parameter
        in model.m6.parameters()
    ):
        raise RuntimeError(
            f"{expected_stage}: "
            "embedded M6 is trainable."
        )

    compare_alpha_maps(
        model.alpha_max_by_term,
        alpha_map,
        path=checkpoint_path,
    )

    return (
        model,
        payload,
    )


# ============================================================
# Exact parent-state audits
# ============================================================

def assert_same_parent_raw_alpha(
    canonical,
    child,
    child_name,
):

    for term in (
        ACTIVE_TERMS
    ):

        parent_value = (
            canonical
            .raw_alpha[
                term
            ]
            .detach()
        )

        child_value = (
            child
            .raw_alpha[
                term
            ]
            .detach()
        )

        if not torch.equal(
            parent_value,
            child_value,
        ):
            raise RuntimeError(
                f"{child_name}: "
                "frozen parent raw_alpha "
                f"differs for {term}."
            )

    print(
        f"✅ {child_name}: "
        "frozen parent raw_alpha "
        "exactly matches R3-1b"
    )


# ============================================================
# Synthetic interface audit
#
# No TEST data are touched.
# ============================================================

def synthetic_interface_audit(
    *,
    paramonly,
    stateparam,
    alpha_map,
    device,
):

    x = torch.zeros(
        2,
        16,
        256,
        64,
        dtype=torch.float32,
        device=device,
    )

    params = torch.tensor(
        [
            [
                6.0,
                -0.3010300,
            ],
            [
                8.0,
                0.3010300,
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    required = {
        "effective_gate_values",
        "delta_raw",
        "full_conditioner_features",
        "conditioner_input",
    }

    with torch.no_grad():

        for name, model in (
            (
                DISPLAY_PARAMONLY,
                paramonly,
            ),
            (
                DISPLAY_STATEPARAM,
                stateparam,
            ),
        ):

            (
                output,
                info,
            ) = model(
                x,
                params=params,
                return_components=True,
            )

            if not torch.isfinite(
                output
            ).all():
                raise RuntimeError(
                    f"{name}: "
                    "non-finite synthetic output."
                )

            missing = (
                required
                -
                set(info)
            )

            if missing:
                raise RuntimeError(
                    f"{name}: missing "
                    "return_components keys: "
                    f"{missing}"
                )

            for term in (
                ACTIVE_TERMS
            ):

                gate = (
                    info[
                        "effective_gate_values"
                    ][
                        term
                    ]
                    .reshape(-1)
                )

                if (
                    gate.numel()
                    !=
                    x.shape[0]
                ):
                    raise RuntimeError(
                        f"{name}/{term}: "
                        "expected per-sample "
                        "gate [B]."
                    )

                if not torch.isfinite(
                    gate
                ).all():
                    raise RuntimeError(
                        f"{name}/{term}: "
                        "non-finite gate."
                    )

                bound = float(
                    alpha_map[
                        term
                    ]
                )

                max_abs = float(
                    gate
                    .abs()
                    .max()
                    .cpu()
                )

                if (
                    max_abs
                    >
                    bound
                    +
                    1.0e-12
                ):
                    raise RuntimeError(
                        f"{name}/{term}: "
                        "gate exceeds frozen "
                        "alpha_max."
                    )

    print(
        "✅ R3-2 synthetic "
        "forward/interface audit PASS"
    )


# ============================================================
# Per-sample adaptive-gate accumulation
# ============================================================

def new_gate_bucket():

    return {

        "n": 0,

        "adv_sum": 0.0,
        "adv_sq_sum": 0.0,
        "adv_min": float("inf"),
        "adv_max": float("-inf"),
        "adv_abs_capacity_sum": 0.0,
        "adv_sat99_n": 0,

        "forcing_sum": 0.0,
        "forcing_sq_sum": 0.0,
        "forcing_min": float("inf"),
        "forcing_max": float("-inf"),
        "forcing_abs_capacity_sum": 0.0,
        "forcing_sat99_n": 0,
    }


def _gate_vector(
    info,
    term,
    batch_size,
):

    if (
        "effective_gate_values"
        in info
    ):
        gate = (
            info[
                "effective_gate_values"
            ][
                term
            ]
        )

    else:
        # R3-1b canonical:
        # one global scalar gate.
        gate = (
            info[
                "gate_values"
            ][
                term
            ]
        )

    gate = (
        gate
        .detach()
        .double()
        .reshape(-1)
    )

    # Expand the R3-1b global scalar so n means
    # number of samples for all three models.
    if (
        gate.numel() == 1
        and
        batch_size > 1
    ):
        gate = gate.expand(
            batch_size
        )

    if (
        gate.numel()
        !=
        batch_size
    ):
        raise RuntimeError(
            "Gate shape mismatch "
            f"for {term}: "
            f"numel={gate.numel()}, "
            f"batch_size={batch_size}"
        )

    return (
        gate
        .cpu()
        .numpy()
    )


def update_gate_stats(
    gate_stats,
    model_name,
    step,
    info,
    batch_size,
    alpha_map,
):

    key = (
        model_name,
        step,
    )

    if key not in (
        gate_stats
    ):
        gate_stats[
            key
        ] = new_gate_bucket()

    bucket = (
        gate_stats[
            key
        ]
    )

    adv_values = (
        _gate_vector(
            info,
            "buoyancy_advection",
            batch_size,
        )
    )

    forcing_values = (
        _gate_vector(
            info,
            "buoyancy_forcing",
            batch_size,
        )
    )

    adv_cap = float(
        alpha_map[
            "buoyancy_advection"
        ]
    )

    forcing_cap = float(
        alpha_map[
            "buoyancy_forcing"
        ]
    )

    bucket[
        "n"
    ] += int(
        batch_size
    )

    for (
        prefix,
        values,
        cap,
    ) in (
        (
            "adv",
            adv_values,
            adv_cap,
        ),
        (
            "forcing",
            forcing_values,
            forcing_cap,
        ),
    ):

        bucket[
            f"{prefix}_sum"
        ] += float(
            np.sum(
                values
            )
        )

        bucket[
            f"{prefix}_sq_sum"
        ] += float(
            np.sum(
                values * values
            )
        )

        bucket[
            f"{prefix}_min"
        ] = min(
            bucket[
                f"{prefix}_min"
            ],
            float(
                np.min(
                    values
                )
            ),
        )

        bucket[
            f"{prefix}_max"
        ] = max(
            bucket[
                f"{prefix}_max"
            ],
            float(
                np.max(
                    values
                )
            ),
        )

        ratios = (
            np.abs(
                values
            )
            /
            max(
                cap,
                1.0e-30,
            )
        )

        bucket[
            f"{prefix}_abs_capacity_sum"
        ] += float(
            np.sum(
                ratios
            )
        )

        bucket[
            f"{prefix}_sat99_n"
        ] += int(
            np.sum(
                ratios
                >=
                0.99
            )
        )


def gate_rows(
    gate_stats,
    model_order,
    max_horizon,
):

    rows = []

    for model_name in (
        model_order
    ):

        for step in range(
            1,
            max_horizon + 1,
        ):

            bucket = (
                gate_stats[
                    (
                        model_name,
                        step,
                    )
                ]
            )

            n = bucket[
                "n"
            ]

            row = {
                "model":
                    model_name,

                "horizon":
                    step,
            }

            for (
                prefix,
                long_name,
            ) in (
                (
                    "adv",
                    "advection",
                ),
                (
                    "forcing",
                    "forcing",
                ),
            ):

                mean = (
                    bucket[
                        f"{prefix}_sum"
                    ]
                    /
                    n
                )

                var = max(
                    0.0,
                    (
                        bucket[
                            f"{prefix}_sq_sum"
                        ]
                        /
                        n
                        -
                        mean
                        *
                        mean
                    ),
                )

                row[
                    f"alpha_{long_name}_mean"
                ] = mean

                row[
                    f"alpha_{long_name}_std"
                ] = math.sqrt(
                    var
                )

                row[
                    f"alpha_{long_name}_min"
                ] = (
                    bucket[
                        f"{prefix}_min"
                    ]
                )

                row[
                    f"alpha_{long_name}_max"
                ] = (
                    bucket[
                        f"{prefix}_max"
                    ]
                )

                row[
                    f"alpha_{long_name}_"
                    "mean_abs_capacity_ratio"
                ] = (
                    bucket[
                        f"{prefix}_"
                        "abs_capacity_sum"
                    ]
                    /
                    n
                )

                row[
                    f"alpha_{long_name}_"
                    "sat99_fraction"
                ] = (
                    bucket[
                        f"{prefix}_"
                        "sat99_n"
                    ]
                    /
                    n
                )

            rows.append(
                row
            )

    return rows


# ============================================================
# Free-autoregressive rollout
#
# Same physical-space protocol as frozen R3-1.
# ============================================================

def rollout_one_model(
    *,
    model_name,
    model,
    x0_phys,
    future_phys,
    param,
    normalizer,
    max_horizon,
    error_stats,
    gate_stats,
    alpha_map,
):

    x_norm = (
        normalizer
        .normalize_x(
            x0_phys
        )
    )

    for step in range(
        1,
        max_horizon + 1,
    ):

        current_norm = (
            x_norm[
                :,
                -4:,
                :,
                :,
            ]
        )

        (
            pred_delta_norm,
            info,
        ) = model(
            x_norm,
            params=param,
            return_components=True,
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

        true_phys = (
            future_phys[
                :,
                step - 1,
                :,
                :,
                :,
            ]
        )

        update_error_stats(
            error_stats,
            model_name,
            step,
            pred_next_phys,
            true_phys,
        )

        update_gate_stats(
            gate_stats,
            model_name,
            step,
            info,
            batch_size=(
                x0_phys.shape[0]
            ),
            alpha_map=(
                alpha_map
            ),
        )

        # Free autoregressive feedback.
        x_norm = torch.cat(
            [
                x_norm[
                    :,
                    4:,
                    :,
                    :,
                ],
                pred_next_norm,
            ],
            dim=1,
        )


# ============================================================
# Matched difference table
# ============================================================

def make_difference_table(
    summary_df,
):

    pairs = [

        # Secondary:
        (
            DISPLAY_PARAMONLY,
            DISPLAY_CANONICAL,
        ),

        # Secondary:
        (
            DISPLAY_STATEPARAM,
            DISPLAY_CANONICAL,
        ),

        # PRIMARY:
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
                "horizon",
                "field",
            ],
            suffixes=(
                "_a",
                "_b",
            ),
        )

        for _, row in (
            merged.iterrows()
        ):

            rows.append(
                {
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

                    "field":
                        row[
                            "field"
                        ],

                    "rel_l2_diff_percent_point":
                        (
                            row[
                                "rel_l2_percent_a"
                            ]
                            -
                            row[
                                "rel_l2_percent_b"
                            ]
                        ),

                    "mse_diff":
                        (
                            row[
                                "mse_a"
                            ]
                            -
                            row[
                                "mse_b"
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Frozen R3-1b trajectory reproduction audit
#
# Formal R3-2 evaluation MUST reproduce the already-frozen
# R3-1b H1-H16 curve before interpreting any new comparison.
# ============================================================

def reproduce_r3_1_reference(
    *,
    curve_df,
    split_label,
    seed,
    max_horizon,
):

    reference_path = (
        resolve_path(
            os.path.join(
                "outputs",
                "tables",
                "r3_1_rollout",
                (
                    f"r3_1_rollout_"
                    f"{split_label}_"
                    f"seed{seed}_"
                    f"curve_h1_h"
                    f"{max_horizon}.csv"
                ),
            )
        )
    )

    if not os.path.exists(
        reference_path
    ):
        raise FileNotFoundError(
            "Frozen R3-1 reference curve "
            "is required for formal "
            "reproduction audit: "
            f"{reference_path}"
        )

    reference = pd.read_csv(
        reference_path
    )

    reference = reference[
        reference[
            "model"
        ]
        ==
        DISPLAY_CANONICAL
    ][
        [
            "horizon",
            "field",
            "rel_l2_percent",
            "mse",
        ]
    ]

    current = curve_df[
        curve_df[
            "model"
        ]
        ==
        DISPLAY_CANONICAL
    ][
        [
            "horizon",
            "field",
            "rel_l2_percent",
            "mse",
        ]
    ]

    merged = current.merge(
        reference,
        on=[
            "horizon",
            "field",
        ],
        suffixes=(
            "_current",
            "_reference",
        ),
        how="inner",
    )

    # 4 fields + global = 5 rows / horizon.
    expected_rows = (
        max_horizon
        *
        5
    )

    if (
        len(merged)
        !=
        expected_rows
    ):
        raise RuntimeError(
            "Incomplete R3-1b "
            "reproduction table: "
            f"expected={expected_rows}, "
            f"actual={len(merged)}"
        )

    max_rel = float(
        np.max(
            np.abs(
                merged[
                    "rel_l2_percent_current"
                ]
                -
                merged[
                    "rel_l2_percent_reference"
                ]
            )
        )
    )

    max_mse = float(
        np.max(
            np.abs(
                merged[
                    "mse_current"
                ]
                -
                merged[
                    "mse_reference"
                ]
            )
        )
    )

    if max_rel > 2.0e-9:
        raise RuntimeError(
            "R3-2 evaluator does not "
            "reproduce frozen R3-1b "
            "Rel-L2 curve: "
            f"max_abs={max_rel:.12e}"
        )

    if max_mse > 2.0e-12:
        raise RuntimeError(
            "R3-2 evaluator does not "
            "reproduce frozen R3-1b "
            "MSE curve: "
            f"max_abs={max_mse:.12e}"
        )

    return {
        "reference_path":
            reference_path,

        "max_abs_rel_l2_diff":
            max_rel,

        "max_abs_mse_diff":
            max_mse,
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Formal evaluator contract
    # --------------------------------------------------------

    if args.seed != 42:
        raise ValueError(
            "Formal R3-2 evaluation "
            "is locked to seed=42."
        )

    if args.batch_size != 4:
        raise ValueError(
            "Formal R3-2 evaluation "
            "is locked to batch_size=4."
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
        [
            1,
            4,
            8,
            16,
        ]
    ):
        raise ValueError(
            "R3-2 formal reporting "
            "horizons are locked to "
            "1,4,8,16."
        )

    max_horizon = 16

    resource = (
        LOCKED_RESOURCES[
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

    r3_0_contract_path = (
        resolve_path(
            R3_0_CONTRACT
        )
    )

    # --------------------------------------------------------
    # Locked SHA provenance
    # --------------------------------------------------------

    audit_locked_file(
        split_path,
        resource[
            "split_sha256"
        ],
        (
            f"{args.split_label} "
            "split"
        ),
    )

    audit_locked_file(
        stats_path,
        resource[
            "stats_sha256"
        ],
        (
            f"{args.split_label} "
            "stats"
        ),
    )

    audit_locked_file(
        m6_path,
        resource[
            "m6_sha256"
        ],
        (
            f"{args.split_label} "
            "M6"
        ),
    )

    actual_r3_0_sha = (
        audit_locked_file(
            r3_0_contract_path,
            EXPECTED_R3_0_CONTRACT_SHA256,
            "R3-0 contract",
        )
    )

    (
        registry,
        entries,
        registry_path,
        r3_2_contract_path,
        r3_2_contract_sha,
    ) = (
        load_and_audit_registry(
            args.split_label
        )
    )

    parent_expected_sha = (
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
        parent_expected_sha
    ):
        raise RuntimeError(
            "R3-2a/R3-2b do not "
            "share the same frozen "
            "R3-1b parent."
        )

    r3_1b_sha = (
        audit_locked_file(
            r3_1b_path,
            parent_expected_sha,
            (
                f"{args.split_label} "
                "R3-1b parent"
            ),
        )
    )

    paramonly_path = (
        resolve_path(
            entries[
                "R3-2a"
            ][
                "checkpoint"
            ]
        )
    )

    stateparam_path = (
        resolve_path(
            entries[
                "R3-2b"
            ][
                "checkpoint"
            ]
        )
    )

    paramonly_sha = (
        sha256_file(
            paramonly_path
        )
    )

    stateparam_sha = (
        sha256_file(
            stateparam_path
        )
    )

    registry_sha = (
        sha256_file(
            registry_path
        )
    )

    # --------------------------------------------------------
    # R3-0 fixed capacities
    # --------------------------------------------------------

    (
        _,
        alpha_map,
    ) = load_locked_contract(
        r3_0_contract_path
    )

    # --------------------------------------------------------
    # Device / deterministic label
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

    print(
        "=" * 118
    )

    print(
        "R3-2 MATCHED FORMAL TEST "
        "ROLLOUT EVALUATOR"
    )

    print(
        "=" * 118
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
        "Models:",
        MODEL_ORDER,
    )

    print(
        "Primary contrast: "
        "R3-2b-StateParam "
        "- R3-2a-ParamOnly"
    )

    print(
        "Negative difference = "
        "StateParam better"
    )

    print(
        "Horizons: h=1,4,8,16; "
        "full curve h=1..16"
    )

    print(
        "Formal TEST windows: "
        "all legal windows, stride 1"
    )

    print(
        "Checkpoint selection: "
        "frozen before TEST"
    )

    print(
        "Audit only:",
        args.audit_only,
    )

    print()
    print(
        "========== LOCKED PROVENANCE =========="
    )

    print(
        "R3_0_CONTRACT_SHA256:",
        actual_r3_0_sha,
    )

    print(
        "R3_2_CONTRACT_SHA256:",
        r3_2_contract_sha,
    )

    print(
        "R3_2_REGISTRY_SHA256:",
        registry_sha,
    )

    print(
        "SPLIT_SHA256:",
        resource[
            "split_sha256"
        ],
    )

    print(
        "STATS_SHA256:",
        resource[
            "stats_sha256"
        ],
    )

    print(
        "M6_SHA256:",
        resource[
            "m6_sha256"
        ],
    )

    print(
        "R3_1B_SHA256:",
        r3_1b_sha,
    )

    print(
        "R3_2A_SHA256:",
        paramonly_sha,
    )

    print(
        "R3_2B_SHA256:",
        stateparam_sha,
    )

    # --------------------------------------------------------
    # Build metadata + models.
    #
    # This part is allowed in --audit_only.
    # No TEST dataset has been constructed.
    # --------------------------------------------------------

    metadata = (
        build_rbc_canonical_metadata(
            stats_path
        )
    )

    (
        m6,
        _,
    ) = build_m6(
        m6_path,
        device,
    )

    # build_r3 only needs split_label + seed
    # from the R3-1 evaluator args object.
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
    ) = build_r3(
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
            actual_r3_0_sha
        ),
        args=(
            r3_args
        ),
        device=(
            device
        ),
    )

    (
        paramonly,
        paramonly_payload,
    ) = build_r3_2(
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

    (
        stateparam,
        stateparam_payload,
    ) = build_r3_2(
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

    # --------------------------------------------------------
    # Exact embedded M6 audit
    # --------------------------------------------------------

    assert_same_m6(
        m6,
        canonical,
        DISPLAY_CANONICAL,
    )

    assert_same_m6(
        m6,
        paramonly,
        DISPLAY_PARAMONLY,
    )

    assert_same_m6(
        m6,
        stateparam,
        DISPLAY_STATEPARAM,
    )

    # --------------------------------------------------------
    # Exact frozen parent raw-alpha audit
    # --------------------------------------------------------

    assert_same_parent_raw_alpha(
        canonical,
        paramonly,
        DISPLAY_PARAMONLY,
    )

    assert_same_parent_raw_alpha(
        canonical,
        stateparam,
        DISPLAY_STATEPARAM,
    )

    # --------------------------------------------------------
    # Synthetic model-interface audit
    # --------------------------------------------------------

    synthetic_interface_audit(
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

    for name, payload in (
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
            f"{name}: "
            "best_val="
            f"{payload.get('best_val_loss', payload.get('val_loss'))} "
            "best_epoch="
            f"{payload.get('best_epoch', payload.get('epoch'))}"
        )

    # --------------------------------------------------------
    # AUDIT-ONLY STOP.
    #
    # Crucially:
    # split_config["test"] has NOT been read into a rollout
    # dataset at this point.
    # --------------------------------------------------------

    if args.audit_only:

        print()
        print(
            "✅ R3-2 EVALUATOR "
            "AUDIT-ONLY PASS"
        )

        print(
            "✅ Registry/checkpoints/"
            "model interfaces verified"
        )

        print(
            "✅ TEST rollout dataset "
            "was NOT constructed or accessed"
        )

        return

    # ========================================================
    # TEST access starts ONLY here.
    # ========================================================

    print()
    print(
        "TEST access: YES "
        "(formal post-training evaluation)"
    )

    if (
        args.max_samples
        is not None
    ):

        print(
            "⚠️ DEBUG TEST ONLY: "
            "max_samples =",
            args.max_samples,
        )

        print(
            "⚠️ This run is NOT "
            "the formal full TEST result."
        )

    split_config = (
        load_json(
            split_path
        )
    )

    dataset = RolloutDataset(
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

    # Formal all-legal H16 TEST support previously frozen
    # in R3-1: 2430 windows for each split.
    if (
        args.max_samples
        is None
        and
        len(dataset)
        !=
        2430
    ):
        raise RuntimeError(
            "Formal R3-2 TEST must "
            "contain exactly 2430 "
            "all-legal H16 windows; "
            f"got {len(dataset)}"
        )

    loader = DataLoader(
        dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=0,
    )

    normalizer = (
        FieldWiseNormalizer(
            stats_path
        )
        .to(device)
    )

    models = {

        DISPLAY_CANONICAL:
            canonical,

        DISPLAY_PARAMONLY:
            paramonly,

        DISPLAY_STATEPARAM:
            stateparam,
    }

    error_stats = {}

    gate_stats = {}

    print()
    print(
        "🔥 Starting R3-2 "
        "free-autoregressive TEST rollout..."
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

                rollout_one_model(
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
                    error_stats=(
                        error_stats
                    ),
                    gate_stats=(
                        gate_stats
                    ),
                    alpha_map=(
                        alpha_map
                    ),
                )

            if (
                batch_idx
                %
                10
                ==
                0
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
    # Tables
    # ========================================================

    curve_df = pd.DataFrame(
        error_rows(
            error_stats,
            MODEL_ORDER,
            max_horizon,
        )
    )

    summary_df = (
        curve_df[
            curve_df[
                "horizon"
            ]
            .isin(
                requested_horizons
            )
        ]
        .copy()
    )

    gates_df = pd.DataFrame(
        gate_rows(
            gate_stats,
            MODEL_ORDER,
            max_horizon,
        )
    )

    growth_df = (
        make_global_growth_table(
            curve_df,
            MODEL_ORDER,
            max_horizon,
        )
    )

    diff_df = (
        make_difference_table(
            summary_df
        )
    )

    for df in (
        curve_df,
        summary_df,
        gates_df,
        growth_df,
        diff_df,
    ):

        df.insert(
            0,
            "seed",
            args.seed,
        )

        df.insert(
            0,
            "split",
            args.split_label,
        )

    # ========================================================
    # Frozen R3-1b reproduction
    # ========================================================

    if (
        args.max_samples
        is None
    ):

        reproduction = (
            reproduce_r3_1_reference(
                curve_df=(
                    curve_df
                ),
                split_label=(
                    args.split_label
                ),
                seed=(
                    args.seed
                ),
                max_horizon=(
                    max_horizon
                ),
            )
        )

        print()
        print(
            "========== FROZEN "
            "R3-1b REPRODUCTION =========="
        )

        print(
            "max_abs Rel-L2 difference = "
            f"{reproduction['max_abs_rel_l2_diff']:.12e}"
        )

        print(
            "max_abs MSE difference    = "
            f"{reproduction['max_abs_mse_diff']:.12e}"
        )

        print(
            "✅ R3-2 evaluator reproduces "
            "frozen R3-1b rollout"
        )

    else:

        reproduction = {
            "skipped":
                True,

            "reason":
                (
                    "debug max_samples run "
                    "is not comparable to "
                    "frozen full R3-1 curve"
                ),
        }

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
        f"r3_2_rollout_"
        f"{args.split_label}_"
        f"seed{args.seed}"
    )

    paths = {

        "summary":
            os.path.join(
                output_dir,
                f"{prefix}_summary.csv",
            ),

        "curve":
            os.path.join(
                output_dir,
                (
                    f"{prefix}_"
                    "curve_h1_h16.csv"
                ),
            ),

        "growth":
            os.path.join(
                output_dir,
                (
                    f"{prefix}_"
                    "global_growth_auc.csv"
                ),
            ),

        "gates":
            os.path.join(
                output_dir,
                (
                    f"{prefix}_"
                    "gate_curve.csv"
                ),
            ),

        "diff":
            os.path.join(
                output_dir,
                (
                    f"{prefix}_"
                    "differences.csv"
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

    summary_df.to_csv(
        paths[
            "summary"
        ],
        index=False,
    )

    curve_df.to_csv(
        paths[
            "curve"
        ],
        index=False,
    )

    growth_df.to_csv(
        paths[
            "growth"
        ],
        index=False,
    )

    gates_df.to_csv(
        paths[
            "gates"
        ],
        index=False,
    )

    diff_df.to_csv(
        paths[
            "diff"
        ],
        index=False,
    )

    metadata_out = {

        "experiment":
            "R3-2-matched-formal-"
            "test-rollout",

        "formal_run":
            (
                args.max_samples
                is None
            ),

        "split_label":
            args.split_label,

        "seed":
            args.seed,

        "models":
            list(
                MODEL_ORDER
            ),

        "primary_comparison":
            (
                "R3-2b-StateParam "
                "- R3-2a-ParamOnly"
            ),

        "primary_interpretation":
            (
                "incremental value of "
                "current-state information "
                "beyond the capacity-matched "
                "parameter-only conditioner"
            ),

        "negative_primary_difference_means":
            "StateParam better",

        "horizons_reported":
            requested_horizons,

        "full_max_horizon":
            max_horizon,

        "formal_test_windows":
            (
                None
                if
                args.max_samples
                is not None
                else
                len(dataset)
            ),

        "window_policy":
            (
                "all legal TEST windows; "
                "stride 1; same H16 support "
                "for h1/h4/h8/h16"
            ),

        "rollout_protocol":
            (
                "free-autoregressive; "
                "predicted next state is "
                "fed back into the next context"
            ),

        "metric_space":
            (
                "denormalized physical/"
                "field-value space"
            ),

        "formal_alpha_max_by_term":
            alpha_map,

        "r3_0_contract_sha256":
            actual_r3_0_sha,

        "r3_2_contract_sha256":
            r3_2_contract_sha,

        "r3_2_registry_sha256":
            registry_sha,

        "split_sha256":
            resource[
                "split_sha256"
            ],

        "stats_sha256":
            resource[
                "stats_sha256"
            ],

        "m6_checkpoint_sha256":
            resource[
                "m6_sha256"
            ],

        "r3_1b_checkpoint_sha256":
            r3_1b_sha,

        "r3_2a_checkpoint_sha256":
            paramonly_sha,

        "r3_2b_checkpoint_sha256":
            stateparam_sha,

        "r3_1b_reproduction":
            reproduction,

        # These diagnostics are declared in code
        # BEFORE any R3-2 TEST result is seen.
        "gate_diagnostics_predeclared_before_test": {

            "per_horizon_mean_std_min_max":
                True,

            "mean_abs_capacity_ratio":
                True,

            "fraction_abs_gate_ge_0.99_capacity":
                True,
        },

        "checkpoint_set_frozen_before_test":
            True,

        "test_accessed":
            True,
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

    global_wide = (
        summary_df[
            summary_df[
                "field"
            ]
            ==
            "global"
        ]
        .pivot_table(
            index="model",
            columns="horizon",
            values=(
                "rel_l2_percent"
            ),
            aggfunc="first",
        )
        .reindex(
            MODEL_ORDER
        )
    )

    print()
    print(
        "================ "
        "GLOBAL REL-L2 (%) "
        "================"
    )

    print(
        global_wide
        .to_string()
    )

    primary_diff = (
        diff_df[
            (
                diff_df[
                    "comparison"
                ]
                ==
                (
                    f"{DISPLAY_STATEPARAM} "
                    f"- {DISPLAY_PARAMONLY}"
                )
            )
            &
            (
                diff_df[
                    "field"
                ]
                ==
                "global"
            )
        ][
            [
                "horizon",
                "rel_l2_diff_percent_point",
            ]
        ]
    )

    print()
    print(
        "========== "
        "R3-2b StateParam "
        "- R3-2a ParamOnly "
        "| GLOBAL Rel-L2 pp "
        "=========="
    )

    print(
        "Negative = StateParam better."
    )

    print(
        primary_diff
        .to_string(
            index=False
        )
    )

    print()
    print(
        "================ "
        "GLOBAL ERROR GROWTH / AUC "
        "================"
    )

    print(
        growth_df
        .drop(
            columns=[
                "split",
                "seed",
            ]
        )
        .to_string(
            index=False
        )
    )

    gate_key = (
        gates_df[
            gates_df[
                "horizon"
            ]
            .isin(
                requested_horizons
            )
        ][
            [
                "model",
                "horizon",

                "alpha_advection_mean",
                "alpha_advection_std",
                "alpha_advection_mean_abs_capacity_ratio",
                "alpha_advection_sat99_fraction",

                "alpha_forcing_mean",
                "alpha_forcing_std",
                "alpha_forcing_mean_abs_capacity_ratio",
                "alpha_forcing_sat99_fraction",
            ]
        ]
    )

    print()
    print(
        "================ "
        "EFFECTIVE GATE DIAGNOSTICS "
        "================"
    )

    print(
        gate_key
        .to_string(
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

        print(
            f"  {key}: {path}"
        )

    print()

    if (
        args.max_samples
        is None
    ):

        print(
            "✅ FORMAL R3-2 "
            "TEST ROLLOUT COMPLETE"
        )

    else:

        print(
            "✅ DEBUG R3-2 "
            "TEST ROLLOUT COMPLETE "
            "(NOT FORMAL)"
        )


if __name__ == "__main__":
    main()
