from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


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
    CONTEXT_LENGTH,
    DTYPE,
)

from training.normalization import (
    FieldWiseNormalizer,
)

from models.operators.fno2d_fieldwise import (
    FieldWiseFNO2d,
)

from models.operators.fno2d_r3_pde_coupling import (
    R3PDECouplingFNO2d,
)

from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
)


# ============================================================
# Frozen R3-1 evaluation contract
# ============================================================

DISPLAY_M6 = "M6"
DISPLAY_NAIVE = "R3-1a-Naive"
DISPLAY_CANONICAL = "R3-1b-Canonical"

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
# TEST rollout dataset
#
# Protocol intentionally follows the mature M10 evaluator:
#
#   physical history
#       -> normalize
#       -> free autoregressive prediction
#       -> denormalize predicted next state
#       -> physical-space error
#
# Every sample has the SAME max_horizon support, so h=1/4/8/16
# are evaluated on the same window set.
# ============================================================

class RolloutDataset(Dataset):

    def __init__(
        self,
        split_config,
        max_horizon=16,
        max_samples=None,
    ):
        self.split_config = split_config
        self.max_horizon = int(
            max_horizon
        )

        self.max_samples = (
            max_samples
        )

        self.data_store = {}
        self.index = []

        self._load_data()

    @staticmethod
    def _parse_ra_pr(
        group_name,
    ):
        ra_match = re.search(
            r"[Rr]a_?([0-9.eE+-]+)",
            group_name,
        )

        pr_match = re.search(
            r"[Pp]r_?([0-9.eE+-]+)",
            group_name,
        )

        if (
            ra_match is None
            or
            pr_match is None
        ):
            raise ValueError(
                "Cannot parse Ra/Pr "
                f"from group: {group_name}"
            )

        return (
            float(
                ra_match.group(1)
            ),
            float(
                pr_match.group(1)
            ),
        )

    @classmethod
    def _make_param(
        cls,
        group_name,
    ):
        ra, pr = cls._parse_ra_pr(
            group_name
        )

        return torch.tensor(
            [
                math.log10(ra),
                math.log10(pr),
            ],
            dtype=DTYPE,
        )

    def _load_data(
        self,
    ):
        print(
            "📦 Loading R3-1 TEST rollout data: "
            f"{len(self.split_config)} group batches"
        )

        with h5py.File(
            DATA_PATH,
            "r",
        ) as f:

            for item in (
                self.split_config
            ):

                group_name = (
                    item["group"]
                )

                traj_indices = (
                    item["trajectories"]
                )

                if group_name not in f:
                    raise KeyError(
                        "Group not found in dataset: "
                        f"{group_name}"
                    )

                group = f[
                    group_name
                ]

                fields_data = [
                    group[field][:]
                    for field
                    in FIELD_ORDER
                ]

                stacked = torch.tensor(
                    np.stack(
                        fields_data,
                        axis=0,
                    ),
                    dtype=DTYPE,
                )

                # [field, traj, time, X, Y]
                selected = stacked[
                    :,
                    traj_indices,
                ]

                # ->
                # [traj, time, field, X, Y]
                data = selected.permute(
                    1,
                    2,
                    0,
                    3,
                    4,
                ).contiguous()

                self.data_store[
                    group_name
                ] = data

                param = (
                    self._make_param(
                        group_name
                    )
                )

                (
                    num_traj,
                    num_steps,
                    _,
                    _,
                    _,
                ) = data.shape

                max_start = (
                    num_steps
                    -
                    CONTEXT_LENGTH
                    -
                    self.max_horizon
                    +
                    1
                )

                if max_start <= 0:
                    raise RuntimeError(
                        "Not enough time steps in "
                        f"{group_name} for "
                        f"horizon={self.max_horizon}"
                    )

                for local_traj_idx in range(
                    num_traj
                ):

                    for t0 in range(
                        max_start
                    ):

                        self.index.append(
                            {
                                "group":
                                    group_name,

                                "traj":
                                    local_traj_idx,

                                "t0":
                                    t0,

                                "param":
                                    param,
                            }
                        )

        if (
            self.max_samples
            is not None
        ):
            self.index = (
                self.index[
                    : int(
                        self.max_samples
                    )
                ]
            )

        print(
            "✅ TEST rollout samples:",
            len(self.index),
        )

    def __len__(
        self,
    ):
        return len(
            self.index
        )

    def __getitem__(
        self,
        idx,
    ):
        item = (
            self.index[
                idx
            ]
        )

        data = (
            self.data_store[
                item["group"]
            ]
        )

        history = data[
            item["traj"],
            item["t0"]:
            item["t0"]
            +
            CONTEXT_LENGTH,
        ]

        future = data[
            item["traj"],
            item["t0"]
            +
            CONTEXT_LENGTH:
            item["t0"]
            +
            CONTEXT_LENGTH
            +
            self.max_horizon,
        ]

        _, _, h, w = (
            history.shape
        )

        x0_phys = (
            history.reshape(
                CONTEXT_LENGTH * 4,
                h,
                w,
            )
        )

        return (
            x0_phys,
            future,
            item["param"],
        )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "R3-1 matched cross-parameter TEST rollout: "
            "M6 vs R3-1a Naive vs R3-1b Canonical."
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
            "Formal evaluation must omit this."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/tables/"
            "r3_1_rollout"
        ),
    )

    return parser.parse_args()


# ============================================================
# Path / provenance helpers
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


def extract_state_dict(
    payload,
):
    if (
        isinstance(
            payload,
            dict,
        )
        and
        "model_state_dict"
        in payload
    ):
        return payload[
            "model_state_dict"
        ]

    if isinstance(
        payload,
        dict,
    ):
        return payload

    raise TypeError(
        "Unsupported checkpoint format."
    )


def load_checkpoint(
    path,
    device,
):
    payload = torch.load(
        path,
        map_location=device,
    )

    return (
        payload,
        extract_state_dict(
            payload
        ),
    )


# ============================================================
# Recursive metadata lookup
#
# Skip large state dictionaries so provenance search stays cheap.
# ============================================================

_SKIP_METADATA_KEYS = {
    "model_state_dict",
    "optimizer_state_dict",
    "state_dict",
}


def find_metadata_key(
    obj,
    target_key,
):
    if not isinstance(
        obj,
        dict,
    ):
        return None

    if target_key in obj:
        return obj[
            target_key
        ]

    for key, value in (
        obj.items()
    ):

        if key in (
            _SKIP_METADATA_KEYS
        ):
            continue

        if isinstance(
            value,
            dict,
        ):
            found = (
                find_metadata_key(
                    value,
                    target_key,
                )
            )

            if found is not None:
                return found

    return None


# ============================================================
# Locked R3 contract
# ============================================================

def load_locked_contract(
    contract_path,
):
    with open(
        contract_path,
        "r",
        encoding="utf-8",
    ) as f:
        contract = json.load(
            f
        )

    policy = (
        contract[
            "correction"
        ][
            "alpha_upper_bound_policy"
        ]
    )

    if (
        policy.get(
            "formal_values_status"
        )
        !=
        "LOCKED"
    ):
        raise RuntimeError(
            "R3 formal alpha capacity "
            "is not LOCKED."
        )

    if (
        policy.get(
            "shared_between_r3_1a_and_r3_1b"
        )
        is not True
    ):
        raise RuntimeError(
            "R3 matched capacity policy "
            "is not shared across arms."
        )

    if (
        policy.get(
            "arm_specific_tuning_forbidden"
        )
        is not True
    ):
        raise RuntimeError(
            "Arm-specific tuning must be forbidden."
        )

    if (
        policy.get(
            "selection_uses_validation_performance"
        )
        is not False
    ):
        raise RuntimeError(
            "Formal capacity must not be "
            "selected from VAL performance."
        )

    if (
        policy.get(
            "selection_uses_test"
        )
        is not False
    ):
        raise RuntimeError(
            "Formal capacity must not be "
            "selected using TEST."
        )

    alpha_map_raw = (
        policy[
            "formal_alpha_max_by_term"
        ]
    )

    alpha_map = {
        term:
            float(
                alpha_map_raw[
                    term
                ]
            )
        for term
        in ACTIVE_TERMS
    }

    if (
        set(alpha_map_raw)
        !=
        set(ACTIVE_TERMS)
    ):
        raise RuntimeError(
            "Locked alpha map does not "
            "match R3 active terms."
        )

    active_from_contract = {
        name
        for name, cfg
        in contract[
            "active_routing"
        ].items()
        if cfg["active"]
    }

    if (
        active_from_contract
        !=
        set(ACTIVE_TERMS)
    ):
        raise RuntimeError(
            "Contract active routing does "
            "not match R3-1 evaluator."
        )

    return (
        contract,
        alpha_map,
    )


# ============================================================
# Model builders
# ============================================================

def build_m6(
    checkpoint_path,
    device,
):

    model = FieldWiseFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
    ).to(device)

    (
        payload,
        state,
    ) = load_checkpoint(
        checkpoint_path,
        device,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    return (
        model,
        payload,
    )


def compare_alpha_maps(
    actual,
    expected,
    *,
    path,
):

    if (
        set(actual)
        !=
        set(expected)
    ):
        raise RuntimeError(
            "Checkpoint alpha map keys mismatch: "
            f"{path}"
        )

    for term in expected:

        if not math.isclose(
            float(
                actual[
                    term
                ]
            ),
            float(
                expected[
                    term
                ]
            ),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise RuntimeError(
                "Checkpoint alpha capacity mismatch: "
                f"{path}: "
                f"{term}="
                f"{actual[term]}, "
                "expected="
                f"{expected[term]}"
            )


def audit_r3_checkpoint(
    payload,
    *,
    representation_mode,
    expected_stage,
    split_label,
    seed,
    alpha_map,
    contract_sha,
    path,
):

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "R3 checkpoint must contain "
            "metadata payload: "
            f"{path}"
        )

    # These were frozen by the R3 shared trainer.
    if (
        payload.get(
            "representation_mode"
        )
        !=
        representation_mode
    ):
        raise RuntimeError(
            "representation_mode mismatch: "
            f"{path}"
        )

    if (
        payload.get(
            "stage"
        )
        !=
        expected_stage
    ):
        raise RuntimeError(
            "stage mismatch: "
            f"{path}: "
            f"{payload.get('stage')} "
            f"!= {expected_stage}"
        )

    # Check optional top-level metadata if present.
    if (
        "split_label"
        in payload
        and
        payload[
            "split_label"
        ]
        !=
        split_label
    ):
        raise RuntimeError(
            "split_label mismatch: "
            f"{path}"
        )

    if (
        "seed"
        in payload
        and
        int(
            payload["seed"]
        )
        !=
        int(seed)
    ):
        raise RuntimeError(
            "seed mismatch: "
            f"{path}"
        )

    stored_alpha = (
        find_metadata_key(
            payload,
            "alpha_max_by_term",
        )
    )

    if stored_alpha is not None:
        compare_alpha_maps(
            stored_alpha,
            alpha_map,
            path=path,
        )

    stored_contract_sha = (
        find_metadata_key(
            payload,
            "contract_sha256",
        )
    )

    if stored_contract_sha is None:
        stored_contract_sha = (
            find_metadata_key(
                payload,
                "contract_sha",
            )
        )

    if (
        stored_contract_sha
        is not None
        and
        str(
            stored_contract_sha
        )
        !=
        str(
            contract_sha
        )
    ):
        raise RuntimeError(
            "Contract SHA mismatch: "
            f"{path}: "
            f"{stored_contract_sha} "
            f"!= {contract_sha}"
        )


def build_r3(
    *,
    representation_mode,
    checkpoint_path,
    metadata,
    alpha_map,
    contract_sha,
    args,
    device,
):

    model = R3PDECouplingFNO2d(
        canonical_metadata=metadata,
        representation_mode=(
            representation_mode
        ),
        alpha_max_by_term=(
            alpha_map
        ),
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        freeze_m6=True,
    ).to(device)

    (
        payload,
        state,
    ) = load_checkpoint(
        checkpoint_path,
        device,
    )

    expected_stage = (
        "R3-1a"
        if representation_mode
        == "naive"
        else
        "R3-1b"
    )

    audit_r3_checkpoint(
        payload,
        representation_mode=(
            representation_mode
        ),
        expected_stage=(
            expected_stage
        ),
        split_label=(
            args.split_label
        ),
        seed=(
            args.seed
        ),
        alpha_map=(
            alpha_map
        ),
        contract_sha=(
            contract_sha
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
        2
    ):
        raise RuntimeError(
            "R3 checkpoint does not have "
            "exactly two trainable scalar gates."
        )

    if set(
        model.trainable_parameter_names()
    ) != {
        "raw_alpha.buoyancy_advection",
        "raw_alpha.buoyancy_forcing",
    }:
        raise RuntimeError(
            "R3 trainable parameter topology mismatch."
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


def assert_same_m6(
    reference_m6,
    r3_model,
    model_name,
):

    ref = (
        reference_m6
        .state_dict()
    )

    got = (
        r3_model
        .m6
        .state_dict()
    )

    if (
        ref.keys()
        !=
        got.keys()
    ):
        raise RuntimeError(
            f"{model_name}: embedded M6 "
            "state keys differ."
        )

    for key in ref:

        if not torch.equal(
            ref[key],
            got[key],
        ):
            max_diff = float(
                (
                    ref[key]
                    -
                    got[key]
                )
                .abs()
                .max()
                .item()
            )

            raise RuntimeError(
                f"{model_name}: embedded M6 "
                f"differs at {key}; "
                "max_abs_diff="
                f"{max_diff:.6e}"
            )

    print(
        f"✅ {model_name}: embedded frozen M6 "
        "exactly matches audited M6"
    )


# ============================================================
# Error accumulation
# ============================================================

def new_error_bucket(
    device,
):

    return {
        "field_sse":
            torch.zeros(
                4,
                dtype=torch.float64,
                device=device,
            ),

        "field_target_sq":
            torch.zeros(
                4,
                dtype=torch.float64,
                device=device,
            ),

        "field_numel":
            torch.zeros(
                4,
                dtype=torch.float64,
                device=device,
            ),

        "global_sse":
            torch.tensor(
                0.0,
                dtype=torch.float64,
                device=device,
            ),

        "global_target_sq":
            torch.tensor(
                0.0,
                dtype=torch.float64,
                device=device,
            ),

        "global_numel":
            torch.tensor(
                0.0,
                dtype=torch.float64,
                device=device,
            ),
    }


def update_error_stats(
    stats,
    model_name,
    step,
    pred_phys,
    true_phys,
):

    key = (
        model_name,
        step,
    )

    if key not in stats:
        stats[key] = (
            new_error_bucket(
                pred_phys.device
            )
        )

    bucket = (
        stats[key]
    )

    diff = (
        pred_phys
        -
        true_phys
    )

    for c in range(4):

        diff_c = (
            diff[
                :,
                c,
                :,
                :,
            ]
            .double()
        )

        true_c = (
            true_phys[
                :,
                c,
                :,
                :,
            ]
            .double()
        )

        bucket[
            "field_sse"
        ][c] += torch.sum(
            diff_c ** 2
        )

        bucket[
            "field_target_sq"
        ][c] += torch.sum(
            true_c ** 2
        )

        bucket[
            "field_numel"
        ][c] += (
            diff_c.numel()
        )

    bucket[
        "global_sse"
    ] += torch.sum(
        diff.double() ** 2
    )

    bucket[
        "global_target_sq"
    ] += torch.sum(
        true_phys.double() ** 2
    )

    bucket[
        "global_numel"
    ] += diff.numel()


# ============================================================
# Global-gate statistics
# ============================================================

def new_gate_bucket():

    return {
        "n": 0,

        "adv_sum": 0.0,
        "adv_sq_sum": 0.0,
        "adv_min": float("inf"),
        "adv_max": float("-inf"),

        "forcing_sum": 0.0,
        "forcing_sq_sum": 0.0,
        "forcing_min": float("inf"),
        "forcing_max": float("-inf"),
    }


def update_gate_stats(
    gate_stats,
    model_name,
    step,
    info,
):

    key = (
        model_name,
        step,
    )

    if key not in gate_stats:
        gate_stats[key] = (
            new_gate_bucket()
        )

    bucket = (
        gate_stats[key]
    )

    adv = float(
        info[
            "gate_values"
        ][
            "buoyancy_advection"
        ]
        .detach()
        .double()
        .cpu()
        .item()
    )

    forcing = float(
        info[
            "gate_values"
        ][
            "buoyancy_forcing"
        ]
        .detach()
        .double()
        .cpu()
        .item()
    )

    bucket["n"] += 1

    bucket[
        "adv_sum"
    ] += adv

    bucket[
        "adv_sq_sum"
    ] += adv * adv

    bucket[
        "adv_min"
    ] = min(
        bucket[
            "adv_min"
        ],
        adv,
    )

    bucket[
        "adv_max"
    ] = max(
        bucket[
            "adv_max"
        ],
        adv,
    )

    bucket[
        "forcing_sum"
    ] += forcing

    bucket[
        "forcing_sq_sum"
    ] += (
        forcing
        *
        forcing
    )

    bucket[
        "forcing_min"
    ] = min(
        bucket[
            "forcing_min"
        ],
        forcing,
    )

    bucket[
        "forcing_max"
    ] = max(
        bucket[
            "forcing_max"
        ],
        forcing,
    )


# ============================================================
# Free-autoregressive rollout
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

            info = None

        else:

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

        if info is not None:

            update_gate_stats(
                gate_stats,
                model_name,
                step,
                info,
            )

        # Free autoregressive:
        # feed the model's own predicted state back.
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
# Tables
# ============================================================

def error_rows(
    error_stats,
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
                error_stats[
                    (
                        model_name,
                        step,
                    )
                ]
            )

            for c, field in enumerate(
                FIELD_ORDER
            ):

                rel = torch.sqrt(
                    bucket[
                        "field_sse"
                    ][c]
                    /
                    (
                        bucket[
                            "field_target_sq"
                        ][c]
                        +
                        1.0e-12
                    )
                ) * 100.0

                mse = (
                    bucket[
                        "field_sse"
                    ][c]
                    /
                    bucket[
                        "field_numel"
                    ][c]
                )

                rows.append(
                    {
                        "model":
                            model_name,

                        "horizon":
                            step,

                        "field":
                            field,

                        "rel_l2_percent":
                            float(
                                rel.item()
                            ),

                        "mse":
                            float(
                                mse.item()
                            ),
                    }
                )

            global_rel = torch.sqrt(
                bucket[
                    "global_sse"
                ]
                /
                (
                    bucket[
                        "global_target_sq"
                    ]
                    +
                    1.0e-12
                )
            ) * 100.0

            global_mse = (
                bucket[
                    "global_sse"
                ]
                /
                bucket[
                    "global_numel"
                ]
            )

            rows.append(
                {
                    "model":
                        model_name,

                    "horizon":
                        step,

                    "field":
                        "global",

                    "rel_l2_percent":
                        float(
                            global_rel.item()
                        ),

                    "mse":
                        float(
                            global_mse.item()
                        ),
                }
            )

    return rows


def gate_rows(
    gate_stats,
    model_order,
    max_horizon,
):

    rows = []

    for model_name in (
        model_order
    ):

        if (
            model_name
            ==
            DISPLAY_M6
        ):
            continue

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

            adv_mean = (
                bucket[
                    "adv_sum"
                ]
                /
                n
            )

            adv_var = max(
                0.0,
                (
                    bucket[
                        "adv_sq_sum"
                    ]
                    /
                    n
                    -
                    adv_mean
                    *
                    adv_mean
                ),
            )

            forcing_mean = (
                bucket[
                    "forcing_sum"
                ]
                /
                n
            )

            forcing_var = max(
                0.0,
                (
                    bucket[
                        "forcing_sq_sum"
                    ]
                    /
                    n
                    -
                    forcing_mean
                    *
                    forcing_mean
                ),
            )

            rows.append(
                {
                    "model":
                        model_name,

                    "horizon":
                        step,

                    "alpha_advection_mean":
                        adv_mean,

                    "alpha_advection_std":
                        math.sqrt(
                            adv_var
                        ),

                    "alpha_advection_min":
                        bucket[
                            "adv_min"
                        ],

                    "alpha_advection_max":
                        bucket[
                            "adv_max"
                        ],

                    "alpha_forcing_mean":
                        forcing_mean,

                    "alpha_forcing_std":
                        math.sqrt(
                            forcing_var
                        ),

                    "alpha_forcing_min":
                        bucket[
                            "forcing_min"
                        ],

                    "alpha_forcing_max":
                        bucket[
                            "forcing_max"
                        ],
                }
            )

    return rows


def make_global_growth_table(
    curve_df,
    model_order,
    max_horizon,
):

    rows = []

    for model_name in (
        model_order
    ):

        sub = curve_df[
            (
                curve_df[
                    "model"
                ]
                ==
                model_name
            )
            &
            (
                curve_df[
                    "field"
                ]
                ==
                "global"
            )
        ].sort_values(
            "horizon"
        )

        x = (
            sub[
                "horizon"
            ]
            .to_numpy(
                dtype=float
            )
        )

        y = (
            sub[
                "rel_l2_percent"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if (
            len(y)
            !=
            max_horizon
        ):
            raise RuntimeError(
                "Incomplete global curve "
                f"for {model_name}"
            )

        auc = float(
            np.sum(
                0.5
                *
                (
                    y[:-1]
                    +
                    y[1:]
                )
                *
                np.diff(x)
            )
        )

        span = float(
            x[-1]
            -
            x[0]
        )

        rows.append(
            {
                "model":
                    model_name,

                "global_rel_l2_h1":
                    float(
                        y[0]
                    ),

                (
                    f"global_rel_l2_"
                    f"h{max_horizon}"
                ):
                    float(
                        y[-1]
                    ),

                "global_error_growth_abs":
                    float(
                        y[-1]
                        -
                        y[0]
                    ),

                "global_error_growth_ratio":
                    float(
                        y[-1]
                        /
                        max(
                            y[0],
                            1.0e-12,
                        )
                    ),

                "global_error_growth_slope":
                    float(
                        (
                            y[-1]
                            -
                            y[0]
                        )
                        /
                        max(
                            span,
                            1.0,
                        )
                    ),

                "global_auc_trapz":
                    auc,

                "global_auc_mean_over_span":
                    float(
                        auc
                        /
                        max(
                            span,
                            1.0,
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


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
# Main
# ============================================================

def main():

    args = parse_args()

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

    contract_path = resolve_path(
        args.contract
    )

    for path in (
        split_path,
        stats_path,
        m6_path,
        naive_path,
        canonical_path,
        contract_path,
    ):

        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                path
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
        not requested_horizons
        or
        requested_horizons[
            0
        ]
        <
        1
    ):
        raise ValueError(
            "Invalid --horizons"
        )

    max_horizon = max(
        requested_horizons
    )

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

    (
        _,
        alpha_map,
    ) = load_locked_contract(
        contract_path
    )

    contract_sha = (
        sha256_file(
            contract_path
        )
    )

    split_sha = (
        sha256_file(
            split_path
        )
    )

    stats_sha = (
        sha256_file(
            stats_path
        )
    )

    m6_sha = (
        sha256_file(
            m6_path
        )
    )

    naive_sha = (
        sha256_file(
            naive_path
        )
    )

    canonical_sha = (
        sha256_file(
            canonical_path
        )
    )

    print(
        "=" * 110
    )

    print(
        "R3-1 MATCHED TEST ROLLOUT EVALUATION"
    )

    print(
        "=" * 110
    )

    print(
        "Device:",
        device,
    )

    print(
        "Split:",
        args.split_label,
    )

    print(
        "Seed label:",
        args.seed,
    )

    print(
        "Models:",
        MODEL_ORDER,
    )

    print(
        "Horizons reported:",
        requested_horizons,
    )

    print(
        "Full rollout curve:",
        f"h=1..{max_horizon}",
    )

    print(
        "Formal alpha max by term:",
        alpha_map,
    )

    print(
        "TEST access: YES "
        "(formal post-training evaluation)"
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
            "⚠️ This run is NOT the formal full TEST result."
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
        "M6_SHA256:",
        m6_sha,
    )

    print(
        "NAIVE_SHA256:",
        naive_sha,
    )

    print(
        "CANONICAL_SHA256:",
        canonical_sha,
    )

    # ========================================================
    # TEST dataset
    # ========================================================

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as f:

        split_config = (
            json.load(
                f
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

    metadata = (
        build_rbc_canonical_metadata(
            stats_path
        )
    )

    # ========================================================
    # Models
    # ========================================================

    (
        m6,
        _,
    ) = build_m6(
        m6_path,
        device,
    )

    (
        naive,
        naive_payload,
    ) = build_r3(
        representation_mode="naive",
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
        args=args,
        device=device,
    )

    (
        canonical,
        canonical_payload,
    ) = build_r3(
        representation_mode="canonical",
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
        args=args,
        device=device,
    )

    assert_same_m6(
        m6,
        naive,
        DISPLAY_NAIVE,
    )

    assert_same_m6(
        m6,
        canonical,
        DISPLAY_CANONICAL,
    )

    print()
    print(
        "========== CHECKPOINT MODEL SELECTION =========="
    )

    for name, payload in (
        (
            DISPLAY_NAIVE,
            naive_payload,
        ),
        (
            DISPLAY_CANONICAL,
            canonical_payload,
        ),
    ):

        print(
            f"{name}: "
            "best_val="
            f"{payload.get('best_val_loss', payload.get('val_loss', 'NA'))}, "
            "best_epoch="
            f"{payload.get('best_epoch', payload.get('epoch', 'NA'))}"
        )

    models = {
        DISPLAY_M6:
            m6,

        DISPLAY_NAIVE:
            naive,

        DISPLAY_CANONICAL:
            canonical,
    }

    # ========================================================
    # Full free-autoregressive TEST rollout
    # ========================================================

    error_stats = {}
    gate_stats = {}

    print()
    print(
        "🔥 Starting free-autoregressive TEST rollout..."
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

    summary_df = curve_df[
        curve_df[
            "horizon"
        ]
        .isin(
            requested_horizons
        )
    ].copy()

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
        "r3_1_rollout_"
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
                    f"curve_h1_h"
                    f"{max_horizon}.csv"
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
            "R3-1-matched-test-rollout",

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

        "horizons_reported":
            requested_horizons,

        "full_max_horizon":
            max_horizon,

        "formal_alpha_max_by_term":
            alpha_map,

        "contract_sha256":
            contract_sha,

        "split_sha256":
            split_sha,

        "stats_sha256":
            stats_sha,

        "m6_checkpoint_sha256":
            m6_sha,

        "naive_checkpoint_sha256":
            naive_sha,

        "canonical_checkpoint_sha256":
            canonical_sha,

        "test_accessed":
            True,

        "rollout_protocol":
            (
                "free-autoregressive; "
                "predicted next state is fed back "
                "into the next context"
            ),

        "metric_space":
            (
                "denormalized physical/field-value space"
            ),

        "comparison_of_interest":
            (
                "R3-1b-Canonical - R3-1a-Naive; "
                "negative Rel-L2 difference means "
                "Canonical is better"
            ),
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
    # Compact human-readable output
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

    key_diff = diff_df[
        (
            diff_df[
                "comparison"
            ]
            ==
            (
                f"{DISPLAY_CANONICAL} "
                f"- {DISPLAY_NAIVE}"
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

    print()
    print(
        "========== "
        "R3-1b Canonical - R3-1a Naive "
        "| GLOBAL Rel-L2 pp "
        "=========="
    )

    print(
        "Negative = Canonical better."
    )

    print(
        key_diff.to_string(
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
        growth_df.drop(
            columns=[
                "split",
                "seed",
            ]
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "================ "
        "FINAL LEARNED GATES "
        "================"
    )

    gate_h1 = gates_df[
        gates_df[
            "horizon"
        ]
        ==
        1
    ][
        [
            "model",
            "alpha_advection_mean",
            "alpha_forcing_mean",
        ]
    ]

    print(
        gate_h1.to_string(
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
            "✅ FORMAL R3-1 TEST ROLLOUT COMPLETE"
        )
    else:
        print(
            "✅ DEBUG R3-1 TEST ROLLOUT COMPLETE "
            "(NOT FORMAL)"
        )


if __name__ == "__main__":
    main()
