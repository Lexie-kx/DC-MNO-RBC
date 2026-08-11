import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from constants import (
    B_IDX,
    UX_IDX,
    UY_IDX,
)


# ============================================================
# Reuse audited M10 evaluator / model builders
# ============================================================

AUDIT_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_m10_physics_audit_v2.py",
)

spec = importlib.util.spec_from_file_location(
    "m10_physics_audit_v2",
    AUDIT_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Cannot import {AUDIT_PATH}"
    )

audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


# ============================================================
# Locked provenance
# ============================================================

EXPECTED_SPLIT_SHA = (
    "475d3092bb9d0ad16f023088446419b5651b1186d369ccfe0019c841f1fd8e36"
)

EXPECTED_STATS_SHA = (
    "a96b1a01cf25d7b9910e01abd4de5672078e6ec3f6a6dda8a96a19e1a022d5af"
)

EXPECTED_M6_SHA = (
    "3d0c1571dcc57e65b1cad45fbdcae72e2b737249033d379426dc616a6c414e53"
)

EXPECTED_M10_SHA = (
    "450a2dcff725154e484f746d2dc883429c10aa5036fadf63944bb4fcb7bd8d76"
)


# ============================================================
# Exact M10-2b feature definitions
# ============================================================

LEGACY_NAMES = [
    "b_mean",
    "b_rms",
    "ux_mean",
    "ux_rms",
    "uy_mean",
    "uy_rms",
    "log1p_path_a_rms",
    "log1p_path_b_rms",
    "logRa",
    "logPr",
]

TEMPORAL_NAMES = (
    LEGACY_NAMES
    +
    [
        "path_b_scale",
        "d1_b_rms",
        "d1_ux_rms",
        "d1_uy_rms",
        "d2_b_rms",
        "d2_ux_rms",
        "d2_uy_rms",
        "m6_delta_b_rms",
        "m6_delta_uy_rms",
    ]
)

LEGACY_COLUMNS = [
    f"legacy__{x}"
    for x in LEGACY_NAMES
]

TEMPORAL_ALPHA_COLUMNS = (
    [
        f"temporal__{x}"
        for x in TEMPORAL_NAMES
    ]
    +
    [
        "alpha_b",
    ]
)


MODEL_ORDER = [
    "M6",
    "B-only",
    "UtilityGate-Legacy10",
    "UtilityGate-TemporalAlpha20",
]


# ============================================================
# FINAL PHYSICS AUDIT PROTOCOL
#
# Physics discretization is deliberately separate from the
# frozen model discretization.
#
# These are FD comparative-proxy settings only.
# They DO NOT alter the trained neural operator.
# ============================================================

PRIMARY_PHYSICS_MODELS = [
    "M6",
    "B-only",
    "UtilityGate-Legacy10",
]

PHYSICS_DX = 1.0 / 64.0
PHYSICS_DY = 1.0 / 63.0
PHYSICS_DT = 0.25


# ============================================================
# COUPLING-REGION PROTOCOL
#
# PREDECLARED before inspecting regional results:
#
#   region = top 20% spatial points of
#            |u . grad b|
#
# computed from the GT PREVIOUS state independently for
# every snapshot.
#
# The mask therefore:
#   * is model-independent;
#   * is causal relative to next-state prediction;
#   * is not tuned using prediction errors.
# ============================================================

HIGH_ADV_FRACTION = 0.20


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "M10-2c closed-loop utility-gate intervention "
            "on validation only."
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
        "--m10_1a_checkpoint",
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
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_horizon",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--classifier_epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--dx",
        type=float,
        default=1.0 / 64.0,
    )

    parser.add_argument(
        "--dy",
        type=float,
        default=1.0 / 64.0,
    )

    parser.add_argument(
        "--alpha_max",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--conditioner_hidden",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--path_b_rms_cap",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--path_b_rms_eps",
        type=float,
        default=1.0e-12,
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/tables/"
            "m10_2c_closedloop"
        ),
    )

    return parser.parse_args()


# ============================================================
# Reproducibility / provenance
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


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
            f"actual={actual}"
        )


# ============================================================
# Classifier
# Exact architecture used in M10-2b
# ============================================================

class UtilityClassifier(nn.Module):

    def __init__(
        self,
        input_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                32,
            ),
            nn.GELU(),

            nn.Linear(
                32,
                16,
            ),
            nn.GELU(),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        return (
            self.net(x)
            .squeeze(-1)
        )


def roc_auc_score_simple(
    y_true,
    score,
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    score = np.asarray(
        score,
        dtype=np.float64,
    )

    pos = (
        y_true == 1
    )

    neg = (
        y_true == 0
    )

    n_pos = int(
        pos.sum()
    )

    n_neg = int(
        neg.sum()
    )

    if (
        n_pos == 0
        or n_neg == 0
    ):
        return float("nan")

    ranks = pd.Series(
        score
    ).rank(
        method="average"
    ).to_numpy()

    rank_sum_pos = float(
        ranks[pos].sum()
    )

    auc = (
        rank_sum_pos
        -
        n_pos
        * (
            n_pos + 1
        )
        / 2.0
    ) / (
        n_pos
        * n_neg
    )

    return float(auc)


def fit_classifier(
    train_df,
    val_df,
    columns,
    *,
    seed,
    epochs,
):

    set_seed(seed)

    train_x = (
        train_df[
            columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    val_x = (
        val_df[
            columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    train_y = (
        train_df[
            "path_b_helps"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    val_y = (
        val_df[
            "path_b_helps"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    mean = train_x.mean(
        axis=0,
        keepdims=True,
    )

    std = train_x.std(
        axis=0,
        keepdims=True,
    )

    std[
        std < 1.0e-6
    ] = 1.0

    train_z = (
        train_x
        -
        mean
    ) / std

    val_z = (
        val_x
        -
        mean
    ) / std

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    x_train = torch.tensor(
        train_z,
        dtype=torch.float32,
        device=device,
    )

    x_val = torch.tensor(
        val_z,
        dtype=torch.float32,
        device=device,
    )

    y_train = torch.tensor(
        train_y,
        dtype=torch.float32,
        device=device,
    )

    model = UtilityClassifier(
        input_dim=train_z.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0e-3,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed
    )

    batch_size = min(
        512,
        len(train_z),
    )

    for _ in range(
        epochs
    ):

        permutation = torch.randperm(
            len(train_z),
            generator=generator,
        )

        model.train()

        for start in range(
            0,
            len(train_z),
            batch_size,
        ):

            idx = permutation[
                start:
                start + batch_size
            ].to(
                device
            )

            logits = model(
                x_train[idx]
            )

            loss = criterion(
                logits,
                y_train[idx],
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

    model.eval()

    with torch.no_grad():

        val_prob = (
            torch.sigmoid(
                model(
                    x_val
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

    auc = roc_auc_score_simple(
        val_y,
        val_prob,
    )

    accuracy = float(
        np.mean(
            (
                val_prob >= 0.5
            ).astype(
                np.int64
            )
            ==
            val_y
        )
    )

    return {
        "model":
            model,

        "mean":
            torch.tensor(
                mean,
                dtype=torch.float32,
                device=device,
            ),

        "std":
            torch.tensor(
                std,
                dtype=torch.float32,
                device=device,
            ),

        "auc":
            auc,

        "accuracy":
            accuracy,

        "num_features":
            len(columns),
    }


# ============================================================
# Runtime feature construction
# ============================================================

def spatial_rms(x):

    return torch.sqrt(
        torch.mean(
            x * x,
            dim=(-2, -1),
        )
        +
        1.0e-12
    )


def build_legacy_features(
    context_5d,
    param,
    comp,
):

    # Exact M10 state-summary:
    # 8 state features
    #
    # plus:
    # [logRa, logPr]
    return torch.cat(
        [
            comp[
                "state_summary"
            ],
            param,
        ],
        dim=-1,
    )


def build_temporal_alpha_features(
    context_5d,
    param,
    comp,
):

    legacy = (
        build_legacy_features(
            context_5d,
            param,
            comp,
        )
    )

    cap_scale = comp[
        "path_b_scale"
    ].reshape(
        -1,
        1,
    )

    latest = context_5d[
        :,
        -1,
    ]

    prev = context_5d[
        :,
        -2,
    ]

    prev2 = context_5d[
        :,
        -3,
    ]

    d1 = latest - prev
    d0 = prev - prev2
    d2 = d1 - d0

    d1_rms = torch.stack(
        [
            spatial_rms(
                d1[
                    :,
                    B_IDX,
                ]
            ),
            spatial_rms(
                d1[
                    :,
                    UX_IDX,
                ]
            ),
            spatial_rms(
                d1[
                    :,
                    UY_IDX,
                ]
            ),
        ],
        dim=-1,
    )

    d2_rms = torch.stack(
        [
            spatial_rms(
                d2[
                    :,
                    B_IDX,
                ]
            ),
            spatial_rms(
                d2[
                    :,
                    UX_IDX,
                ]
            ),
            spatial_rms(
                d2[
                    :,
                    UY_IDX,
                ]
            ),
        ],
        dim=-1,
    )

    base_delta = comp[
        "base_delta_norm"
    ]

    delta_rms = torch.stack(
        [
            spatial_rms(
                base_delta[
                    :,
                    B_IDX,
                ]
            ),
            spatial_rms(
                base_delta[
                    :,
                    UY_IDX,
                ]
            ),
        ],
        dim=-1,
    )

    alpha_b = comp[
        "alpha_b"
    ].reshape(
        -1,
        1,
    )

    return torch.cat(
        [
            legacy,
            cap_scale,
            d1_rms,
            d2_rms,
            delta_rms,
            alpha_b,
        ],
        dim=-1,
    )


@torch.no_grad()
def predict_probability(
    predictor,
    features,
):

    z = (
        features
        -
        predictor[
            "mean"
        ]
    ) / predictor[
        "std"
    ]

    logits = predictor[
        "model"
    ](
        z
    )

    return torch.sigmoid(
        logits
    )


# ============================================================
# Path-B-only residual builder
# ============================================================

def make_path_b_delta(
    comp,
    trust,
):

    out = comp[
        "base_delta_norm"
    ].clone()

    out[
        :,
        B_IDX,
        :,
        :,
    ] = (
        out[
            :,
            B_IDX,
            :,
            :,
        ]
        +
        trust[
            :,
            None,
            None,
        ]
        *
        comp[
            "alpha_b"
        ][
            :,
            None,
            None,
        ]
        *
        comp[
            "path_b_norm_safe"
        ]
    )

    return out


# ============================================================
# Gate statistics
# ============================================================

def update_gate_stats(
    stats,
    model_name,
    step,
    q,
):

    key = (
        model_name,
        step,
    )

    if key not in stats:

        stats[key] = {
            "n": 0,
            "sum": 0.0,
            "sq_sum": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "above_0p5": 0,
        }

    bucket = stats[key]

    q = q.detach().double()

    bucket[
        "n"
    ] += q.numel()

    bucket[
        "sum"
    ] += float(
        q.sum().item()
    )

    bucket[
        "sq_sum"
    ] += float(
        (
            q * q
        ).sum().item()
    )

    bucket[
        "min"
    ] = min(
        bucket["min"],
        float(
            q.min().item()
        ),
    )

    bucket[
        "max"
    ] = max(
        bucket["max"],
        float(
            q.max().item()
        ),
    )

    bucket[
        "above_0p5"
    ] += int(
        (
            q >= 0.5
        ).sum().item()
    )


def finalize_gate_stats(
    stats,
):

    rows = []

    for (
        model_name,
        step,
    ), bucket in sorted(
        stats.items(),
        key=lambda x: (
            MODEL_ORDER.index(
                x[0][0]
            ),
            x[0][1],
        ),
    ):

        n = bucket[
            "n"
        ]

        mean = (
            bucket["sum"]
            / n
        )

        variance = max(
            0.0,
            bucket["sq_sum"]
            / n
            -
            mean * mean,
        )

        rows.append(
            {
                "model":
                    model_name,

                "horizon":
                    step,

                "q_mean":
                    mean,

                "q_std":
                    variance ** 0.5,

                "q_min":
                    bucket["min"],

                "q_max":
                    bucket["max"],

                "q_ge_0p5_fraction":
                    bucket[
                        "above_0p5"
                    ]
                    / n,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Closed-loop validation
# ============================================================

@torch.no_grad()
def evaluate_closed_loop(
    loader,
    normalizer,
    m6,
    m10,
    legacy_predictor,
    temporal_predictor,
    device,
    max_horizon,
):
    """
    Exact fair-rebase H16 closed loop plus Physics Audit v2.

    IMPORTANT:
      * prediction trajectories are unchanged;
      * utility gate is unchanged;
      * no GT enters the gate;
      * GT is used only for offline evaluation metrics;
      * TEST is not accessed;
      * Physics FD settings do not modify the model.
    """

    error_stats = {}
    gate_stats = {}

    # Physics Audit v2 accumulators.
    physics_stats = {}

    # Predeclared coupling-region accumulators.
    region_stats = {}

    embedded_m6_max_abs = 0.0

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

        x0_phys = x0_phys.to(
            device
        )

        future_phys = (
            future_phys.to(
                device
            )
        )

        param = param.to(
            device
        )

        initial_norm = (
            normalizer.normalize_x(
                x0_phys
            )
        )

        contexts = {
            name:
                initial_norm.clone()
            for name in MODEL_ORDER
        }

        for step in range(
            1,
            max_horizon + 1,
        ):

            true_phys = (
                future_phys[
                    :,
                    step - 1,
                ]
            )

            # GT previous state for transition residual.
            if step == 1:
                gt_prev_phys = (
                    x0_phys[
                        :,
                        -4:,
                    ]
                )
            else:
                gt_prev_phys = (
                    future_phys[
                        :,
                        step - 2,
                    ]
                )

            # =================================================
            # Predeclared coupling-active mask
            #
            # Defined ONLY from GT previous state.
            # Same mask is reused for every model branch.
            # =================================================

            high_adv_mask = (
                build_gt_high_adv_mask(
                    gt_prev_phys
                )
            )

            # =================================================
            # M6
            # =================================================

            context = contexts[
                "M6"
            ]

            current_norm = context[
                :,
                -4:,
            ]

            # Previous M6 predicted physical state.
            pred_prev_phys = (
                normalizer.denormalize_y(
                    current_norm
                )
            )

            delta = m6(
                context
            )

            next_norm = (
                current_norm
                +
                delta
            )

            next_phys = (
                normalizer.denormalize_y(
                    next_norm
                )
            )

            # Existing formal prediction metrics.
            audit.base.update_error_stats(
                error_stats,
                "M6",
                step,
                next_phys,
                true_phys,
            )

            # Physics Audit v2 on identical prediction.
            pkey = (
                "M6",
                step,
            )

            if pkey not in physics_stats:
                physics_stats[
                    pkey
                ] = audit.new_bucket()

            audit.update_physics_metrics(
                physics_stats[
                    pkey
                ],
                pred_prev=pred_prev_phys,
                pred_next=next_phys,
                gt_prev=gt_prev_phys,
                gt_next=true_phys,
                param=param,
                dx=PHYSICS_DX,
                dy=PHYSICS_DY,
                dt=PHYSICS_DT,
            )

            # ---------------------------------------------
            # Coupling-region audit:
            # GT-prev top-20% |u.grad b|
            # ---------------------------------------------

            rkey = (
                "M6",
                step,
            )

            if rkey not in region_stats:
                region_stats[
                    rkey
                ] = {}

            update_high_adv_region_metrics(
                bucket=(
                    region_stats[
                        rkey
                    ]
                ),
                pred_prev=(
                    pred_prev_phys
                ),
                pred_next=(
                    next_phys
                ),
                gt_prev=(
                    gt_prev_phys
                ),
                gt_next=(
                    true_phys
                ),
                param=param,
                mask=(
                    high_adv_mask
                ),
            )

            contexts[
                "M6"
            ] = torch.cat(
                [
                    context[
                        :,
                        4:,
                    ],
                    next_norm,
                ],
                dim=1,
            )

            # =================================================
            # M10-2 branches
            # =================================================

            for model_name in [
                "B-only",
                "UtilityGate-Legacy10",
                "UtilityGate-TemporalAlpha20",
            ]:

                context = contexts[
                    model_name
                ]

                current_norm = context[
                    :,
                    -4:,
                ]

                # Previous physical prediction for this branch.
                pred_prev_phys = (
                    normalizer.denormalize_y(
                        current_norm
                    )
                )

                (
                    _,
                    comp,
                ) = m10(
                    context,
                    params=param,
                    return_components=True,
                )

                # ---------------------------------------------
                # Exact embedded-M6 audit
                # ---------------------------------------------

                ref_base = m6(
                    context
                )

                diff = (
                    comp[
                        "base_delta_norm"
                    ]
                    -
                    ref_base
                ).abs().max().item()

                embedded_m6_max_abs = max(
                    embedded_m6_max_abs,
                    diff,
                )

                # ---------------------------------------------
                # Trust
                # ---------------------------------------------

                if model_name == "B-only":

                    trust = torch.ones(
                        context.shape[0],
                        dtype=context.dtype,
                        device=context.device,
                    )

                elif step == 1:

                    # Predictor was trained only on steps 2-4.
                    # Step 1 is pure-GT context by construction.
                    trust = torch.ones(
                        context.shape[0],
                        dtype=context.dtype,
                        device=context.device,
                    )

                else:

                    batch_size = (
                        context.shape[0]
                    )

                    X = context.shape[-2]
                    Y = context.shape[-1]

                    context_5d = (
                        context.reshape(
                            batch_size,
                            4,
                            4,
                            X,
                            Y,
                        )
                    )

                    if (
                        model_name
                        ==
                        "UtilityGate-Legacy10"
                    ):

                        features = (
                            build_legacy_features(
                                context_5d,
                                param,
                                comp,
                            )
                        )

                        trust = (
                            predict_probability(
                                legacy_predictor,
                                features,
                            )
                        )

                    else:

                        features = (
                            build_temporal_alpha_features(
                                context_5d,
                                param,
                                comp,
                            )
                        )

                        trust = (
                            predict_probability(
                                temporal_predictor,
                                features,
                            )
                        )

                # ---------------------------------------------
                # Exact Path-B residual interface
                # ---------------------------------------------

                delta = make_path_b_delta(
                    comp,
                    trust,
                )

                next_norm = (
                    current_norm
                    +
                    delta
                )

                next_phys = (
                    normalizer.denormalize_y(
                        next_norm
                    )
                )

                # Existing formal prediction metrics.
                audit.base.update_error_stats(
                    error_stats,
                    model_name,
                    step,
                    next_phys,
                    true_phys,
                )

                # ---------------------------------------------
                # Physics metrics only for formal primary set:
                #
                # M6
                # B-only
                # UtilityGate-Legacy10
                #
                # TemporalAlpha20 remains a secondary accuracy
                # comparator but is not the formal physics model.
                # ---------------------------------------------

                if (
                    model_name
                    in
                    PRIMARY_PHYSICS_MODELS
                ):

                    pkey = (
                        model_name,
                        step,
                    )

                    if (
                        pkey
                        not in
                        physics_stats
                    ):
                        physics_stats[
                            pkey
                        ] = audit.new_bucket()

                    audit.update_physics_metrics(
                        physics_stats[
                            pkey
                        ],
                        pred_prev=(
                            pred_prev_phys
                        ),
                        pred_next=(
                            next_phys
                        ),
                        gt_prev=(
                            gt_prev_phys
                        ),
                        gt_next=(
                            true_phys
                        ),
                        param=param,
                        dx=PHYSICS_DX,
                        dy=PHYSICS_DY,
                        dt=PHYSICS_DT,
                    )

                # -----------------------------------------
                # Same GT-defined high-advection region.
                #
                # Formal regional models:
                #   B-only
                #   UtilityGate-Legacy10
                #
                # TemporalAlpha20 remains secondary only.
                # -----------------------------------------

                if (
                    model_name
                    in
                    PRIMARY_PHYSICS_MODELS
                ):

                    rkey = (
                        model_name,
                        step,
                    )

                    if (
                        rkey
                        not in
                        region_stats
                    ):
                        region_stats[
                            rkey
                        ] = {}

                    update_high_adv_region_metrics(
                        bucket=(
                            region_stats[
                                rkey
                            ]
                        ),
                        pred_prev=(
                            pred_prev_phys
                        ),
                        pred_next=(
                            next_phys
                        ),
                        gt_prev=(
                            gt_prev_phys
                        ),
                        gt_next=(
                            true_phys
                        ),
                        param=param,
                        mask=(
                            high_adv_mask
                        ),
                    )

                if model_name.startswith(
                    "UtilityGate"
                ):

                    update_gate_stats(
                        gate_stats,
                        model_name,
                        step,
                        trust,
                    )

                contexts[
                    model_name
                ] = torch.cat(
                    [
                        context[
                            :,
                            4:,
                        ],
                        next_norm,
                    ],
                    dim=1,
                )

        if (
            batch_idx % 25 == 0
            or batch_idx
            == len(loader)
        ):

            print(
                f"  processed batch "
                f"{batch_idx}/"
                f"{len(loader)}"
            )

    return (
        error_stats,
        gate_stats,
        embedded_m6_max_abs,
        physics_stats,
        region_stats,
    )



# ============================================================
# Coupling-region audit
#
# Primary region:
#   GT previous-state top-20% |u . grad b|
#
# Tensor convention for mask:
#   [B, X, Y-2]
#
# Primary regional metrics:
#
#   1) next-state buoyancy Rel-L2
#   2) next-state adv_b Rel-L2
#   3) R_b FD normalized mismatch
#
# Lower is better for all three.
# ============================================================


@torch.no_grad()
def build_gt_high_adv_mask(
    gt_prev_phys,
):
    """
    Exactly top ceil(20%) spatial locations PER SNAPSHOT.

    Region definition is based only on GT previous state.
    """

    gt_state_phys = (
        audit.compute_state_physics(
            gt_prev_phys,
            PHYSICS_DX,
            PHYSICS_DY,
        )
    )

    score = torch.abs(
        gt_state_phys[
            "adv_b"
        ]
    )

    batch_size = score.shape[0]

    flat = score.reshape(
        batch_size,
        -1,
    )

    num_points = flat.shape[1]

    k = max(
        1,
        int(
            np.ceil(
                HIGH_ADV_FRACTION
                *
                num_points
            )
        ),
    )

    top_idx = torch.topk(
        flat,
        k=k,
        dim=1,
        largest=True,
        sorted=False,
    ).indices

    mask_flat = torch.zeros_like(
        flat,
        dtype=torch.bool,
    )

    mask_flat.scatter_(
        1,
        top_idx,
        True,
    )

    return mask_flat.reshape_as(
        score
    )


def region_add(
    bucket,
    key,
    value,
):
    bucket[key] = (
        bucket.get(
            key,
            0.0,
        )
        +
        float(value)
    )


def update_masked_rel(
    bucket,
    prefix,
    pred,
    gt,
    mask,
):
    pred64 = pred.double()
    gt64 = gt.double()

    diff = (
        pred64
        -
        gt64
    )

    mask_bool = mask.bool()

    region_add(
        bucket,
        f"{prefix}_err_sq",
        torch.sum(
            diff[
                mask_bool
            ]
            *
            diff[
                mask_bool
            ]
        ).item(),
    )

    region_add(
        bucket,
        f"{prefix}_gt_sq",
        torch.sum(
            gt64[
                mask_bool
            ]
            *
            gt64[
                mask_bool
            ]
        ).item(),
    )


def update_high_adv_region_metrics(
    bucket,
    pred_prev,
    pred_next,
    gt_prev,
    gt_next,
    param,
    mask,
):
    """
    Compute all primary metrics on the SAME predeclared mask.
    """

    # --------------------------------------------------------
    # Mask accounting
    # --------------------------------------------------------

    region_add(
        bucket,
        "mask_selected",
        mask.sum().item(),
    )

    region_add(
        bucket,
        "mask_total",
        mask.numel(),
    )

    # --------------------------------------------------------
    # 1. Next buoyancy prediction in coupling-active region
    #
    # mask lives on y-interior because adv_b uses derivatives.
    # --------------------------------------------------------

    pred_b = pred_next[
        :,
        B_IDX,
        :,
        1:-1,
    ]

    gt_b = gt_next[
        :,
        B_IDX,
        :,
        1:-1,
    ]

    update_masked_rel(
        bucket,
        "buoyancy",
        pred_b,
        gt_b,
        mask,
    )

    # --------------------------------------------------------
    # 2. Next-state advection diagnostic
    # --------------------------------------------------------

    pred_state_phys = (
        audit.compute_state_physics(
            pred_next,
            PHYSICS_DX,
            PHYSICS_DY,
        )
    )

    gt_state_phys = (
        audit.compute_state_physics(
            gt_next,
            PHYSICS_DX,
            PHYSICS_DY,
        )
    )

    update_masked_rel(
        bucket,
        "adv_b",
        pred_state_phys[
            "adv_b"
        ],
        gt_state_phys[
            "adv_b"
        ],
        mask,
    )

    # --------------------------------------------------------
    # 3. R_b FD-based comparative proxy
    #
    # numerator:
    #   ||R_pred^FD - R_GT^FD|| on region
    #
    # denominator:
    #   same-equation GT term energy on SAME region
    # --------------------------------------------------------

    pred_res = (
        audit.compute_pde_residuals(
            pred_prev,
            pred_next,
            param,
            PHYSICS_DX,
            PHYSICS_DY,
            PHYSICS_DT,
        )
    )

    gt_res = (
        audit.compute_pde_residuals(
            gt_prev,
            gt_next,
            param,
            PHYSICS_DX,
            PHYSICS_DY,
            PHYSICS_DT,
        )
    )

    r_diff = (
        pred_res[
            "r_b"
        ].double()
        -
        gt_res[
            "r_b"
        ].double()
    )

    mask_bool = mask.bool()

    region_add(
        bucket,
        "r_b_err_sq",
        torch.sum(
            r_diff[
                mask_bool
            ]
            *
            r_diff[
                mask_bool
            ]
        ).item(),
    )

    gt_term_energy = (
        gt_res[
            "term_energy_b"
        ].double()
    )

    region_add(
        bucket,
        "r_b_gt_term_energy",
        torch.sum(
            gt_term_energy[
                mask_bool
            ]
        ).item(),
    )


def finalize_high_adv_bucket(
    model_name,
    horizon,
    bucket,
    seed,
):
    buoy_rel = (
        np.sqrt(
            bucket[
                "buoyancy_err_sq"
            ]
            /
            (
                bucket[
                    "buoyancy_gt_sq"
                ]
                +
                1.0e-30
            )
        )
        *
        100.0
    )

    adv_rel = (
        np.sqrt(
            bucket[
                "adv_b_err_sq"
            ]
            /
            (
                bucket[
                    "adv_b_gt_sq"
                ]
                +
                1.0e-30
            )
        )
        *
        100.0
    )

    rb_mismatch = (
        np.sqrt(
            bucket[
                "r_b_err_sq"
            ]
            /
            (
                bucket[
                    "r_b_gt_term_energy"
                ]
                +
                1.0e-30
            )
        )
    )

    mask_fraction = (
        bucket[
            "mask_selected"
        ]
        /
        max(
            bucket[
                "mask_total"
            ],
            1.0,
        )
    )

    return {
        "split":
            "unseen_ra",

        "seed":
            int(seed),

        "model":
            model_name,

        "horizon":
            int(horizon),

        "region_definition":
            (
                "GT-prev top20pct "
                "abs(u_dot_grad_b)"
            ),

        "mask_fraction":
            float(
                mask_fraction
            ),

        "high_adv_buoyancy_rel_l2_percent":
            float(
                buoy_rel
            ),

        "high_adv_adv_b_rel_l2_percent":
            float(
                adv_rel
            ),

        "high_adv_r_b_normalized_mismatch":
            float(
                rb_mismatch
            ),
    }


# ============================================================
# Difference tables
# ============================================================

def make_differences(
    summary,
):

    pairs = [
        (
            "B-only",
            "M6",
        ),
        (
            "UtilityGate-Legacy10",
            "M6",
        ),
        (
            "UtilityGate-TemporalAlpha20",
            "M6",
        ),
        (
            "UtilityGate-Legacy10",
            "B-only",
        ),
        (
            "UtilityGate-TemporalAlpha20",
            "B-only",
        ),
    ]

    rows = []

    for (
        model_a,
        model_b,
    ) in pairs:

        a = summary[
            summary["model"]
            ==
            model_a
        ]

        b = summary[
            summary["model"]
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
                        f"{model_a} - "
                        f"{model_b}",

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

                    "rel_l2_diff_pp":
                        float(
                            row[
                                "rel_l2_percent_a"
                            ]
                            -
                            row[
                                "rel_l2_percent_b"
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

    if args.max_horizon != 16:
        raise ValueError(
            "M10-2 candidate long-rollout validation "
            "is deliberately locked to H16."
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
        "=" * 115
    )

    print(
        "M10-2 CANDIDATE LONG-ROLLOUT VALIDATION"
    )

    print(
        "=" * 115
    )

    print(
        "📌 Stage: formal-candidate long-rollout validation"
    )

    print(
        "📌 Neural-operator training: NO"
    )

    print(
        "📌 Utility classifiers: TRAIN-only fit, then frozen"
    )

    print(
        "📌 Closed-loop evaluation split: VAL only"
    )

    print(
        "📌 TEST SPLIT IS NOT ACCESSED"
    )

    print(
        "📌 Path A: OFF"
    )

    print(
        "📌 Path B: M10-1a RMSCap-safe signal"
    )

    print(
        "📌 step1 trust = 1 by design"
    )

    print(
        "📌 step2-16 trust = frozen predicted P(Path B helps)"
    )

    print(
        "📌 No threshold tuning"
    )

    print(
        "📌 H16 stress test: utility gate trained only on H4 states; NO refit"
    )

    print(
        f"📌 Device: {device}"
    )

    # ========================================================
    # Provenance
    # ========================================================

    print()
    print(
        "========== PROVENANCE =========="
    )

    require_sha(
        args.split,
        EXPECTED_SPLIT_SHA,
        "SPLIT",
    )

    require_sha(
        args.stats,
        EXPECTED_STATS_SHA,
        "STATS",
    )

    require_sha(
        args.m6_checkpoint,
        EXPECTED_M6_SHA,
        "M6",
    )

    require_sha(
        args.m10_1a_checkpoint,
        EXPECTED_M10_SHA,
        "M10_1A",
    )

    for label, path in [
        (
            "TRAIN_UTILITY_CSV",
            args.train_utility_csv,
        ),
        (
            "VAL_UTILITY_CSV",
            args.val_utility_csv,
        ),
    ]:

        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                path
            )

        print(
            f"{label}_SHA256:",
            sha256_file(
                path
            ),
        )

    # ========================================================
    # Fit utility predictors from TRAIN only
    # ========================================================

    train_df = pd.read_csv(
        args.train_utility_csv
    )

    val_df = pd.read_csv(
        args.val_utility_csv
    )

    # Classifier contract matches M10-2b:
    # rollout-contaminated states only.
    train_probe = train_df[
        train_df["step"] >= 2
    ].copy()

    val_probe = val_df[
        val_df["step"] >= 2
    ].copy()

    print()
    print(
        "========== UTILITY CLASSIFIER FIT =========="
    )

    print(
        "TRAIN_ROWS:",
        len(train_probe),
    )

    print(
        "VAL_SANITY_ROWS:",
        len(val_probe),
    )

    legacy_predictor = fit_classifier(
        train_probe,
        val_probe,
        LEGACY_COLUMNS,
        seed=args.seed,
        epochs=args.classifier_epochs,
    )

    temporal_predictor = fit_classifier(
        train_probe,
        val_probe,
        TEMPORAL_ALPHA_COLUMNS,
        seed=args.seed,
        epochs=args.classifier_epochs,
    )

    predictor_sanity = pd.DataFrame(
        [
            {
                "predictor":
                    "Legacy10",

                "num_features":
                    legacy_predictor[
                        "num_features"
                    ],

                "val_auc_on_bonly_states":
                    legacy_predictor[
                        "auc"
                    ],

                "val_accuracy_on_bonly_states":
                    legacy_predictor[
                        "accuracy"
                    ],
            },
            {
                "predictor":
                    "TemporalAlpha20",

                "num_features":
                    temporal_predictor[
                        "num_features"
                    ],

                "val_auc_on_bonly_states":
                    temporal_predictor[
                        "auc"
                    ],

                "val_accuracy_on_bonly_states":
                    temporal_predictor[
                        "accuracy"
                    ],
            },
        ]
    )

    print(
        predictor_sanity.to_string(
            index=False
        )
    )

    # These should approximately reproduce M10-2b.
    # ====================================================
    # FAIR-REBASE utility-classifier reproduction lock
    #
    # These values come from the independently completed
    # M10-2 fair-B-only utility-predictability audit:
    #
    #   Legacy10        AUC = 0.826224
    #   TemporalAlpha20 AUC = 0.884938
    #
    # The check remains active. We are NOT disabling the
    # classifier sanity criterion.
    # ====================================================

    expected_auc = {
        "Legacy10":
            0.826224,

        "TemporalAlpha20":
            0.884938,
    }

    for _, row in (
        predictor_sanity.iterrows()
    ):

        expected = expected_auc[
            row["predictor"]
        ]

        actual = float(
            row[
                "val_auc_on_bonly_states"
            ]
        )

        if abs(
            actual
            -
            expected
        ) > 2.0e-3:

            raise RuntimeError(
                "Utility classifier sanity "
                "does not reproduce M10-2b: "
                f"{row['predictor']} "
                f"actual={actual:.6f} "
                f"expected≈{expected:.6f}"
            )

    print(
        "✅ Utility classifier reproduction PASS"
    )

    # ========================================================
    # VAL rollout dataset
    # ========================================================

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as f:

        split_config = json.load(f)

    dataset = audit.base.RolloutDataset(
        split_config=(
            split_config["val"]
        ),
        max_horizon=(
            args.max_horizon
        ),
        max_samples=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print()
    print(
        "VAL_ROLLOUT_SAMPLES:",
        len(dataset),
    )

    # Expected same 558 windows as
    # the earlier TRAIN/VAL reliability study.
    if len(dataset) != 486:

        raise RuntimeError(
            f"Expected 486 VAL H16 windows, "
            f"got {len(dataset)}"
        )

    # ========================================================
    # Models / normalizer
    # ========================================================

    normalizer = (
        audit.base.FieldWiseNormalizer(
            args.stats
        ).to(device)
    )

    (
        field_mean,
        field_std,
    ) = audit.base.build_field_stats(
        args.stats
    )

    m6, _ = (
        audit.base.build_m6(
            args.m6_checkpoint,
            device,
        )
    )

    model_args = SimpleNamespace(
        seed=args.seed,
        dx=args.dx,
        dy=args.dy,
        alpha_max=args.alpha_max,
        conditioner_hidden=(
            args.conditioner_hidden
        ),
        path_b_rms_cap=(
            args.path_b_rms_cap
        ),
        path_b_rms_eps=(
            args.path_b_rms_eps
        ),
    )

    (
        m10,
        payload,
    ) = (
        audit.rmscap_eval.
        build_rmscap_stateparam(
            checkpoint_path=(
                args.m10_1a_checkpoint
            ),
            field_mean=field_mean,
            field_std=field_std,
            args=model_args,
            device=device,
        )
    )

    audit.base.assert_same_m6(
        m6,
        m10,
        "M10-1a-StateParam-RMSCap",
    )

    print(
        "M10_1A_BEST_VAL:",
        payload.get(
            "best_val_loss",
            payload.get(
                "val_loss",
                "NA",
            ),
        )
        if isinstance(
            payload,
            dict,
        )
        else "NA",
    )

    # ========================================================
    # Closed-loop H4
    # ========================================================

    print()
    print(
        "🔥 Starting closed-loop VAL H16 + Physics Audit v2..."
    )

    (
        error_stats,
        gate_stats,
        embedded_m6_max_abs,
        physics_stats,
        region_stats,
    ) = evaluate_closed_loop(
        loader=loader,
        normalizer=normalizer,
        m6=m6,
        m10=m10,
        legacy_predictor=(
            legacy_predictor
        ),
        temporal_predictor=(
            temporal_predictor
        ),
        device=device,
        max_horizon=(
            args.max_horizon
        ),
    )

    print()
    print(
        "========== EMBEDDED M6 CHECK =========="
    )

    print(
        "max_abs embedded vs independent M6 =",
        f"{embedded_m6_max_abs:.12e}",
    )

    if (
        embedded_m6_max_abs
        >
        1.0e-6
    ):
        raise RuntimeError(
            "Embedded M6 mismatch."
        )

    print(
        "✅ Embedded M6 reconstruction PASS"
    )

    # ========================================================
    # Finalize standard physical errors
    # ========================================================

    summary = pd.DataFrame(
        audit.base.error_rows(
            error_stats,
            MODEL_ORDER,
            args.max_horizon,
        )
    )

    differences = (
        make_differences(
            summary
        )
    )

    gate_df = (
        finalize_gate_stats(
            gate_stats
        )
    )

    # ========================================================
    # Finalize Physics Audit v2
    # ========================================================

    physics_rows = []

    for model_name in (
        PRIMARY_PHYSICS_MODELS
    ):

        for horizon in range(
            1,
            args.max_horizon + 1,
        ):

            key = (
                model_name,
                horizon,
            )

            if key not in physics_stats:
                raise RuntimeError(
                    "Missing physics bucket: "
                    f"{key}"
                )

            physics_rows.append(
                audit.finalize_bucket(
                    split_label=(
                        "unseen_ra"
                    ),
                    seed=args.seed,
                    model_name=(
                        model_name
                    ),
                    horizon=horizon,
                    bucket=(
                        physics_stats[
                            key
                        ]
                    ),
                )
            )

    physics_curve = pd.DataFrame(
        physics_rows
    )

    physics_summary = (
        physics_curve[
            physics_curve[
                "horizon"
            ].isin(
                [
                    1,
                    4,
                    8,
                    16,
                ]
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Physics differences
    #
    # Negative = first model is better.
    # --------------------------------------------------------

    physics_pairs = [
        (
            "B-only",
            "M6",
        ),
        (
            "UtilityGate-Legacy10",
            "M6",
        ),
        (
            "UtilityGate-Legacy10",
            "B-only",
        ),
    ]

    physics_diff_rows = []

    for (
        model_a,
        model_b,
    ) in physics_pairs:

        a = physics_summary[
            physics_summary[
                "model"
            ]
            ==
            model_a
        ]

        b = physics_summary[
            physics_summary[
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
                    row["split"],

                "seed":
                    int(
                        row["seed"]
                    ),

                "comparison":
                    f"{model_a} - "
                    f"{model_b}",

                "horizon":
                    int(
                        row[
                            "horizon"
                        ]
                    ),
            }

            for metric in (
                audit.LOWER_IS_BETTER
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

            physics_diff_rows.append(
                out
            )

    physics_differences = (
        pd.DataFrame(
            physics_diff_rows
        )
    )


    # ========================================================
    # Finalize coupling-region audit
    # ========================================================

    region_rows = []

    for model_name in (
        PRIMARY_PHYSICS_MODELS
    ):

        for horizon in range(
            1,
            args.max_horizon + 1,
        ):

            key = (
                model_name,
                horizon,
            )

            if key not in region_stats:
                raise RuntimeError(
                    "Missing coupling-region bucket: "
                    f"{key}"
                )

            region_rows.append(
                finalize_high_adv_bucket(
                    model_name=(
                        model_name
                    ),
                    horizon=(
                        horizon
                    ),
                    bucket=(
                        region_stats[
                            key
                        ]
                    ),
                    seed=args.seed,
                )
            )

    region_curve = pd.DataFrame(
        region_rows
    )

    region_summary = (
        region_curve[
            region_curve[
                "horizon"
            ].isin(
                [
                    1,
                    4,
                    8,
                    16,
                ]
            )
        ]
        .copy()
    )

    region_pairs = [
        (
            "B-only",
            "M6",
        ),
        (
            "UtilityGate-Legacy10",
            "M6",
        ),
        (
            "UtilityGate-Legacy10",
            "B-only",
        ),
    ]

    region_metric_names = [
        "high_adv_buoyancy_rel_l2_percent",
        "high_adv_adv_b_rel_l2_percent",
        "high_adv_r_b_normalized_mismatch",
    ]

    region_diff_rows = []

    for (
        model_a,
        model_b,
    ) in region_pairs:

        a = region_summary[
            region_summary[
                "model"
            ]
            ==
            model_a
        ]

        b = region_summary[
            region_summary[
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
                        f"{model_a} - "
                        f"{model_b}"
                    ),

                "horizon":
                    int(
                        row[
                            "horizon"
                        ]
                    ),
            }

            for metric in (
                region_metric_names
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

            region_diff_rows.append(
                out
            )

    region_differences = (
        pd.DataFrame(
            region_diff_rows
        )
    )

    # ========================================================
    # Compact output
    # ========================================================

    print()
    print(
        "=" * 115
    )

    print(
        "GLOBAL REL-L2 (%)"
    )

    print(
        "=" * 115
    )

    global_table = (
        summary[
            summary[
                "field"
            ]
            ==
            "global"
        ]
        .pivot(
            index="horizon",
            columns="model",
            values="rel_l2_percent",
        )
        .reindex(
            columns=MODEL_ORDER
        )
    )

    print(
        global_table.to_string()
    )

    print()
    print(
        "=" * 115
    )

    print(
        "BUOYANCY REL-L2 (%)"
    )

    print(
        "=" * 115
    )

    buoy_table = (
        summary[
            summary[
                "field"
            ]
            ==
            "buoyancy"
        ]
        .pivot(
            index="horizon",
            columns="model",
            values="rel_l2_percent",
        )
        .reindex(
            columns=MODEL_ORDER
        )
    )

    print(
        buoy_table.to_string()
    )

    print()
    print(
        "=" * 115
    )

    print(
        "UTILITY TRUST DURING CLOSED LOOP"
    )

    print(
        "=" * 115
    )

    print(
        gate_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 115
    )

    print(
        "H4 / H8 / H16 DECISION TABLE"
    )

    print(
        "Negative difference = first model better"
    )

    print(
        "=" * 115
    )

    h4_diff = differences[
        (
            differences[
                "horizon"
            ].isin(
                [
                    4,
                    8,
                    16,
                ]
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

    print(
        h4_diff.to_string(
            index=False
        )
    )


    print()
    print(
        "=" * 150
    )
    print(
        "M10-2 FINAL PHYSICS AUDIT v2 / "
        "UNSEEN-RA / VAL / SEED42"
    )
    print(
        "=" * 150
    )

    print(
        "FD proxy only; "
        "NOT solver-level PDE violation."
    )

    print(
        f"physics_dx={PHYSICS_DX} | "
        f"physics_dy={PHYSICS_DY} | "
        f"dt={PHYSICS_DT}"
    )

    physics_compact_cols = [
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
        physics_summary[
            physics_compact_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "========== PHYSICS KEY DIFFERENCES =========="
    )
    print(
        "Negative = first model better"
    )

    physics_diff_cols = [
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

    print(
        physics_differences[
            physics_diff_cols
        ].to_string(
            index=False
        )
    )


    print()
    print(
        "=" * 140
    )

    print(
        "M10-2 PRIMARY COUPLING-REGION AUDIT"
    )

    print(
        "GT previous-state top-20% |u.grad b|"
    )

    print(
        "UNSEEN-RA / VAL / H16 / SEED42"
    )

    print(
        "=" * 140
    )

    print(
        "Region is model-independent and "
        "predeclared before inspecting results."
    )

    region_cols = [
        "model",
        "horizon",
        "mask_fraction",
        "high_adv_buoyancy_rel_l2_percent",
        "high_adv_adv_b_rel_l2_percent",
        "high_adv_r_b_normalized_mismatch",
    ]

    print()
    print(
        region_summary[
            region_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "========== COUPLING-REGION DIFFERENCES =========="
    )

    print(
        "Negative = first model better"
    )

    region_diff_cols = [
        "comparison",
        "horizon",
        "high_adv_buoyancy_rel_l2_percent_diff",
        "high_adv_adv_b_rel_l2_percent_diff",
        "high_adv_r_b_normalized_mismatch_diff",
    ]

    print(
        region_differences[
            region_diff_cols
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # Save
    # ========================================================

    output_dir = os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            args.output_dir,
        )
        if not os.path.isabs(
            args.output_dir
        )
        else args.output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    prefix = os.path.join(
        output_dir,
        (
            "m10_2_final_physics_"
            "unseen_ra_"
            f"seed{args.seed}_"
            "val_h16"
        ),
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


    physics_curve_path = (
        prefix
        +
        "_physics_curve_h1_h16.csv"
    )

    physics_summary_path = (
        prefix
        +
        "_physics_summary_h1_h4_h8_h16.csv"
    )

    physics_diff_path = (
        prefix
        +
        "_physics_differences.csv"
    )


    region_curve_path = (
        prefix
        +
        "_high_adv_region_curve_h1_h16.csv"
    )

    region_summary_path = (
        prefix
        +
        "_high_adv_region_summary_h1_h4_h8_h16.csv"
    )

    region_diff_path = (
        prefix
        +
        "_high_adv_region_differences.csv"
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


    physics_curve.to_csv(
        physics_curve_path,
        index=False,
    )

    physics_summary.to_csv(
        physics_summary_path,
        index=False,
    )

    physics_differences.to_csv(
        physics_diff_path,
        index=False,
    )


    region_curve.to_csv(
        region_curve_path,
        index=False,
    )

    region_summary.to_csv(
        region_summary_path,
        index=False,
    )

    region_differences.to_csv(
        region_diff_path,
        index=False,
    )

    metadata = {
        "experiment":
            "M10-2 Utility-Aware Candidate Long-Rollout Validation",

        "stage":
            "formal_candidate_long_rollout_validation",

        "neural_operator_training":
            False,

        "utility_classifier_training":
            "TRAIN-only",

        "closed_loop_evaluation":
            "VAL-only",

        "test_split_accessed":
            False,

        "path_a_enabled":
            False,

        "path_b":
            "M10-1a RMSCap-safe",

        "step1_trust":
            1.0,

        "step2_to_16_trust":
            "P(Path B helps | inference-visible state)",

        "thresholding":
            False,

        "max_horizon":
            args.max_horizon,

        "models":
            MODEL_ORDER,

        "split_sha256":
            sha256_file(
                args.split
            ),

        "stats_sha256":
            sha256_file(
                args.stats
            ),

        "m6_sha256":
            sha256_file(
                args.m6_checkpoint
            ),

        "m10_1a_sha256":
            sha256_file(
                args.m10_1a_checkpoint
            ),

        "train_utility_csv_sha256":
            sha256_file(
                args.train_utility_csv
            ),

        "val_utility_csv_sha256":
            sha256_file(
                args.val_utility_csv
            ),

        "classifier_epochs":
            args.classifier_epochs,

        "seed":
            args.seed,
    }


    metadata[
        "physics_audit"
    ] = {
        "version":
            "M10-Physics-Audit-v2",

        "evaluation_split":
            "VAL-only",

        "test_split_accessed":
            False,

        "primary_models":
            PRIMARY_PHYSICS_MODELS,

        "pde_semantics":
            (
                "FD-based comparative proxy; "
                "not solver-level PDE violation"
            ),

        "primary_pde_metric":
            (
                "R_pred^FD minus R_GT^FD, "
                "normalized by same-unit "
                "GT PDE-term scale"
            ),

        "physics_dx":
            PHYSICS_DX,

        "physics_dy":
            PHYSICS_DY,

        "dt":
            PHYSICS_DT,
    }


    metadata[
        "coupling_region_audit"
    ] = {
        "status":
            "predeclared_primary_region",

        "evaluation_split":
            "VAL-only",

        "test_split_accessed":
            False,

        "region_source":
            "GT previous state",

        "region_quantity":
            "abs(u dot grad b)",

        "region_fraction":
            HIGH_ADV_FRACTION,

        "threshold_rule":
            (
                "per-snapshot exact top-k; "
                "k=ceil(0.20*N)"
            ),

        "region_tuning":
            False,

        "primary_models":
            PRIMARY_PHYSICS_MODELS,

        "metrics":
            [
                "next buoyancy Rel-L2",
                "next adv_b Rel-L2",
                "R_b FD normalized mismatch",
            ],
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
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
        physics_curve_path,
        physics_summary_path,
        physics_diff_path,
        region_curve_path,
        region_summary_path,
        region_diff_path,
        metadata_path,
    ]:
        print(
            " ",
            path,
        )

    print()
    print(
        "========== DECISION KEY =========="
    )

    print(
        "PASS pattern:"
    )

    print(
        "  UtilityGate improves over B-only "
        "at h8/h16,"
    )

    print(
        "  and preferably improves or at least "
        "matches M6 on h16 global/buoyancy "
        "without material u_y degradation."
    )

    print()

    print(
        "Partial pattern:"
    )

    print(
        "  UtilityGate improves over B-only "
        "but remains worse than M6."
    )

    print(
        "  -> utility prediction is useful as "
        "damage control, but direct residual "
        "injection is still not a winning interface."
    )

    print()

    print(
        "FAIL pattern:"
    )

    print(
        "  UtilityGate does not improve over B-only."
    )

    print(
        "  -> offline utility predictability does "
        "not survive closed-loop distribution shift."
    )


if __name__ == "__main__":
    main()
