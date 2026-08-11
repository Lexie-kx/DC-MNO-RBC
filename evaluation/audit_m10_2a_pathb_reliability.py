import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
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

from datasets.rbc_dataset import RBCDataset

from scripts.train_m10_1_rmscap_h4 import (
    M10MultiStepParamDataset,
)


# ============================================================
# Reuse audited M10 build logic
# ============================================================

AUDIT_V2_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluate_m10_physics_audit_v2.py",
)

spec = importlib.util.spec_from_file_location(
    "m10_physics_audit_v2",
    AUDIT_V2_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Cannot import {AUDIT_V2_PATH}"
    )

audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


# ============================================================
# Fixed experiment contract
# ============================================================

EXPECTED_SPLIT_SHA = (
    "475d3092bb9d0ad16f023088446419b"
    "5651b1186d369ccfe0019c841f1fd8e36"
)

EXPECTED_STATS_SHA = (
    "a96b1a01cf25d7b9910e01abd4de567"
    "2078e6ec3f6a6dda8a96a19e1a022d5af"
)

EXPECTED_M10_SHA = (
    "4c515ec3eb5c2e6985b68c20e112085"
    "443c58758599ad2bc41ab4c4e26ef3927"
)

FEATURE_NAMES_LEGACY = [
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

FEATURE_NAMES_CAP = (
    FEATURE_NAMES_LEGACY
    +
    [
        "path_b_scale",
    ]
)

FEATURE_NAMES_TEMPORAL = (
    FEATURE_NAMES_CAP
    +
    [
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


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "M10-2a train/val-only Path-B reliability "
            "feasibility audit. "
            "Does NOT train or modify the neural operator."
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
        "--m10_1a_checkpoint",
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
        default=8,
    )

    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=4,
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
        "--probe_epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--run_name",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/tables/m10_2a_reliability",
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
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


# ============================================================
# Tensor helpers
# ============================================================

def spatial_rms(x):

    return torch.sqrt(
        torch.mean(
            x * x,
            dim=(-2, -1),
        )
        + 1.0e-12
    )


def make_b_only_delta(comp):

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
        comp["alpha_b"][
            :,
            None,
            None,
        ]
        *
        comp["path_b_norm_safe"]
    )

    return out


def build_inference_features(
    context,
    param,
    comp,
):

    # --------------------------------------------------------
    # Legacy-10:
    # exact state summary already available to M10-StateParam
    # + [logRa, logPr].
    # --------------------------------------------------------

    state_summary = comp[
        "state_summary"
    ]

    if state_summary is None:

        raise RuntimeError(
            "Expected StateParam state_summary."
        )

    legacy = torch.cat(
        [
            state_summary,
            param,
        ],
        dim=-1,
    )

    # --------------------------------------------------------
    # Existing RMSCap magnitude cue.
    # --------------------------------------------------------

    cap_scale = comp[
        "path_b_scale"
    ].reshape(
        -1,
        1,
    )

    legacy_cap = torch.cat(
        [
            legacy,
            cap_scale,
        ],
        dim=-1,
    )

    # --------------------------------------------------------
    # New temporal drift cues.
    #
    # context:
    # [B,4 time,4 field,X,Y]
    #
    # All are computed only from rollout-visible states.
    # NO GT is used here.
    # --------------------------------------------------------

    latest = context[
        :,
        -1,
    ]

    prev = context[
        :,
        -2,
    ]

    prev2 = context[
        :,
        -3,
    ]

    d1 = latest - prev

    d0 = prev - prev2

    d2 = d1 - d0

    d1_rms = torch.stack(
        [
            spatial_rms(
                d1[:, B_IDX]
            ),
            spatial_rms(
                d1[:, UX_IDX]
            ),
            spatial_rms(
                d1[:, UY_IDX]
            ),
        ],
        dim=-1,
    )

    d2_rms = torch.stack(
        [
            spatial_rms(
                d2[:, B_IDX]
            ),
            spatial_rms(
                d2[:, UX_IDX]
            ),
            spatial_rms(
                d2[:, UY_IDX]
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

    temporal = torch.cat(
        [
            legacy_cap,
            d1_rms,
            d2_rms,
            delta_rms,
        ],
        dim=-1,
    )

    return (
        legacy,
        legacy_cap,
        temporal,
    )


# ============================================================
# Reliability target
# ============================================================

def path_b_reliability_target(
    pred_raw,
    gt_raw,
):
    """
    Dimensionless Path-B reliability target.

    IMPORTANT:
    Do NOT add an RMS floor to the numerator.

    If pred_raw == gt_raw exactly, then:
        error_rms = 0
        relative_error = 0
        q_target = 1

    Numerical epsilon is used only in the denominator.
    """

    difference = (
        pred_raw
        -
        gt_raw
    )

    error_rms = torch.sqrt(
        torch.mean(
            difference
            * difference,
            dim=(-2, -1),
        )
    )

    gt_rms = torch.sqrt(
        torch.mean(
            gt_raw
            * gt_raw,
            dim=(-2, -1),
        )
    )

    relative_error = (
        error_rms
        /
        (
            gt_rms
            +
            1.0e-6
        )
    )

    q_target = (
        1.0
        /
        (
            1.0
            +
            relative_error
        )
    )

    return (
        q_target,
        relative_error,
        gt_rms,
    )


# ============================================================
# Generate reliability rows
# ============================================================

@torch.no_grad()
def generate_rows(
    split_name,
    loader,
    model,
    device,
    rollout_steps,
    max_batches,
):

    model.eval()

    output = []

    for batch_idx, batch in enumerate(
        loader
    ):

        if (
            max_batches is not None
            and batch_idx >= max_batches
        ):
            break

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

        # --------------------------------------------
        # Prediction context:
        # B-only M10-1a rollout.
        #
        # Path A is intentionally OFF because
        # M10-2 candidate direction is Path-B only.
        # --------------------------------------------

        pred_context = (
            context_norm.clone()
        )

        # --------------------------------------------
        # GT context at the same temporal location.
        # --------------------------------------------

        gt_context = (
            context_norm.clone()
        )

        batch_size = (
            context_norm.shape[0]
        )

        for step in range(
            rollout_steps
        ):

            (
                bsz,
                context_len,
                channels,
                h,
                w,
            ) = pred_context.shape

            pred_input = (
                pred_context.reshape(
                    bsz,
                    context_len
                    * channels,
                    h,
                    w,
                )
            )

            gt_input = (
                gt_context.reshape(
                    bsz,
                    context_len
                    * channels,
                    h,
                    w,
                )
            )

            (
                _,
                comp_pred,
            ) = model(
                pred_input,
                params=param,
                return_components=True,
            )

            (
                _,
                comp_gt,
            ) = model(
                gt_input,
                params=param,
                return_components=True,
            )

            # ----------------------------------------
            # Target:
            # fidelity of Path-B computed from
            # rollout state against Path-B from
            # temporally aligned GT state.
            # ----------------------------------------

            (
                q_target,
                path_b_relative_error,
                path_b_gt_rms,
            ) = path_b_reliability_target(
                comp_pred[
                    "path_b_norm_raw"
                ],
                comp_gt[
                    "path_b_norm_raw"
                ],
            )

            # ----------------------------------------
            # Inference-visible features only.
            # ----------------------------------------

            (
                feat_legacy,
                feat_cap,
                feat_temporal,
            ) = build_inference_features(
                context=pred_context,
                param=param,
                comp=comp_pred,
            )

            # ----------------------------------------
            # Does Path B actually help the next-step
            # buoyancy prediction from THIS state?
            #
            # Positive utility_gain:
            #     B-only better than M6.
            #
            # This uses GT only as an offline audit
            # target, NEVER as an inference feature.
            # ----------------------------------------

            current_norm = (
                pred_context[
                    :,
                    -1,
                ]
            )

            gt_next_norm = (
                y_seq_norm[
                    :,
                    step,
                ]
            )

            base_next = (
                current_norm
                +
                comp_pred[
                    "base_delta_norm"
                ]
            )

            b_only_delta = (
                make_b_only_delta(
                    comp_pred
                )
            )

            b_only_next = (
                current_norm
                +
                b_only_delta
            )

            base_b_error = torch.sqrt(
                torch.mean(
                    (
                        base_next[
                            :,
                            B_IDX,
                        ]
                        -
                        gt_next_norm[
                            :,
                            B_IDX,
                        ]
                    )
                    ** 2,
                    dim=(-2, -1),
                )
                + 1.0e-12
            )

            b_only_b_error = torch.sqrt(
                torch.mean(
                    (
                        b_only_next[
                            :,
                            B_IDX,
                        ]
                        -
                        gt_next_norm[
                            :,
                            B_IDX,
                        ]
                    )
                    ** 2,
                    dim=(-2, -1),
                )
                + 1.0e-12
            )

            utility_gain = (
                base_b_error
                -
                b_only_b_error
            )

            # ----------------------------------------
            # Diagnostics for current old mechanisms.
            # ----------------------------------------

            alpha_b = (
                comp_pred[
                    "alpha_b"
                ]
            )

            cap_scale = (
                comp_pred[
                    "path_b_scale"
                ]
            )

            raw_rms = (
                comp_pred[
                    "path_b_rms_raw"
                ]
            )

            cap_active = (
                comp_pred[
                    "path_b_cap_active"
                ].float()
            )

            # ----------------------------------------
            # Move batch rows to CPU.
            # ----------------------------------------

            q_np = (
                q_target.detach()
                .cpu()
                .numpy()
            )

            rel_np = (
                path_b_relative_error
                .detach()
                .cpu()
                .numpy()
            )

            gt_rms_np = (
                path_b_gt_rms.detach()
                .cpu()
                .numpy()
            )

            utility_np = (
                utility_gain.detach()
                .cpu()
                .numpy()
            )

            base_error_np = (
                base_b_error.detach()
                .cpu()
                .numpy()
            )

            b_error_np = (
                b_only_b_error.detach()
                .cpu()
                .numpy()
            )

            alpha_np = (
                alpha_b.detach()
                .cpu()
                .numpy()
            )

            scale_np = (
                cap_scale.detach()
                .cpu()
                .numpy()
            )

            raw_rms_np = (
                raw_rms.detach()
                .cpu()
                .numpy()
            )

            cap_active_np = (
                cap_active.detach()
                .cpu()
                .numpy()
            )

            legacy_np = (
                feat_legacy.detach()
                .cpu()
                .numpy()
            )

            cap_np = (
                feat_cap.detach()
                .cpu()
                .numpy()
            )

            temporal_np = (
                feat_temporal.detach()
                .cpu()
                .numpy()
            )

            for i in range(
                batch_size
            ):

                row = {
                    "split":
                        split_name,

                    "batch_idx":
                        int(batch_idx),

                    "step":
                        int(step + 1),

                    "q_target":
                        float(
                            q_np[i]
                        ),

                    "path_b_relative_error":
                        float(
                            rel_np[i]
                        ),

                    "path_b_gt_rms":
                        float(
                            gt_rms_np[i]
                        ),

                    "utility_gain":
                        float(
                            utility_np[i]
                        ),

                    "path_b_helps":
                        int(
                            utility_np[i]
                            > 0.0
                        ),

                    "m6_b_error":
                        float(
                            base_error_np[i]
                        ),

                    "b_only_b_error":
                        float(
                            b_error_np[i]
                        ),

                    "alpha_b":
                        float(
                            alpha_np[i]
                        ),

                    "path_b_scale":
                        float(
                            scale_np[i]
                        ),

                    "path_b_raw_rms":
                        float(
                            raw_rms_np[i]
                        ),

                    "cap_active":
                        int(
                            cap_active_np[i]
                            > 0.5
                        ),
                }

                for (
                    name,
                    value,
                ) in zip(
                    FEATURE_NAMES_LEGACY,
                    legacy_np[i],
                ):
                    row[
                        f"legacy__{name}"
                    ] = float(
                        value
                    )

                for (
                    name,
                    value,
                ) in zip(
                    FEATURE_NAMES_CAP,
                    cap_np[i],
                ):
                    row[
                        f"cap__{name}"
                    ] = float(
                        value
                    )

                for (
                    name,
                    value,
                ) in zip(
                    FEATURE_NAMES_TEMPORAL,
                    temporal_np[i],
                ):
                    row[
                        f"temporal__{name}"
                    ] = float(
                        value
                    )

                output.append(
                    row
                )

            # ----------------------------------------
            # Advance:
            #
            # prediction = B-only free rollout
            # GT         = true next frame
            # ----------------------------------------

            pred_next_norm = (
                current_norm
                +
                b_only_delta
            )

            pred_context = torch.cat(
                [
                    pred_context[
                        :,
                        1:,
                    ],
                    pred_next_norm.unsqueeze(1),
                ],
                dim=1,
            )

            gt_context = torch.cat(
                [
                    gt_context[
                        :,
                        1:,
                    ],
                    gt_next_norm.unsqueeze(1),
                ],
                dim=1,
            )

        if (
            (batch_idx + 1) % 25 == 0
        ):
            print(
                f"  {split_name}: "
                f"processed batch "
                f"{batch_idx + 1}/"
                f"{len(loader)}"
            )

    return pd.DataFrame(
        output
    )


# ============================================================
# Metrics
# ============================================================

def spearman_corr(
    x,
    y,
):

    x_rank = pd.Series(
        x
    ).rank(
        method="average"
    ).to_numpy()

    y_rank = pd.Series(
        y
    ).rank(
        method="average"
    ).to_numpy()

    if (
        np.std(x_rank) < 1.0e-12
        or np.std(y_rank) < 1.0e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(
            x_rank,
            y_rank,
        )[0, 1]
    )


def regression_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    mae = float(
        np.mean(
            np.abs(
                y_true
                -
                y_pred
            )
        )
    )

    mse = float(
        np.mean(
            (
                y_true
                -
                y_pred
            )
            ** 2
        )
    )

    denominator = float(
        np.sum(
            (
                y_true
                -
                np.mean(
                    y_true
                )
            )
            ** 2
        )
    )

    r2 = (
        float(
            1.0
            -
            np.sum(
                (
                    y_true
                    -
                    y_pred
                )
                ** 2
            )
            /
            (
                denominator
                +
                1.0e-30
            )
        )
    )

    rho = spearman_corr(
        y_true,
        y_pred,
    )

    return {
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "spearman": rho,
    }


# ============================================================
# Tiny auxiliary reliability probe
#
# This does NOT modify M10.
# ============================================================

class ReliabilityProbe(nn.Module):

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
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
    ):
        return (
            self.net(x)
            .squeeze(-1)
        )


def fit_probe(
    train_x,
    train_y,
    val_x,
    *,
    seed,
    epochs,
):

    set_seed(seed)

    train_x = np.asarray(
        train_x,
        dtype=np.float32,
    )

    train_y = np.asarray(
        train_y,
        dtype=np.float32,
    )

    val_x = np.asarray(
        val_x,
        dtype=np.float32,
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

    x_tensor = torch.tensor(
        train_z,
        dtype=torch.float32,
        device=device,
    )

    y_tensor = torch.tensor(
        train_y,
        dtype=torch.float32,
        device=device,
    )

    val_tensor = torch.tensor(
        val_z,
        dtype=torch.float32,
        device=device,
    )

    model = ReliabilityProbe(
        input_dim=train_z.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0e-3,
    )

    criterion = nn.MSELoss()

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

    model.train()

    for epoch in range(
        epochs
    ):

        permutation = torch.randperm(
            len(train_z),
            generator=generator,
        )

        for start in range(
            0,
            len(train_z),
            batch_size,
        ):

            idx_cpu = permutation[
                start:
                start + batch_size
            ]

            idx = idx_cpu.to(
                device
            )

            pred = model(
                x_tensor[
                    idx
                ]
            )

            loss = criterion(
                pred,
                y_tensor[
                    idx
                ],
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

    model.eval()

    with torch.no_grad():

        train_pred = (
            model(
                x_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )

        val_pred = (
            model(
                val_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )

    return (
        train_pred,
        val_pred,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.rollout_steps != 4:
        raise ValueError(
            "M10-2a is locked to H4 "
            "for the first feasibility audit."
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
        "=" * 110
    )

    print(
        "M10-2a PATH-B RELIABILITY FEASIBILITY AUDIT"
    )

    print(
        "=" * 110
    )

    print(
        "📌 Stage: pre-candidate control diagnostic"
    )

    print(
        "📌 Main neural operator training: NO"
    )

    print(
        "📌 Tiny offline reliability probe: YES"
    )

    print(
        "📌 Data used: TRAIN + VAL only"
    )

    print(
        "📌 TEST SPLIT IS NOT ACCESSED"
    )

    print(
        "📌 Rollout generator: frozen M10-1a, "
        "B-only, Path A OFF"
    )

    print(
        "📌 Reliability target uses GT only "
        "during offline supervision"
    )

    print(
        "📌 Reliability features use ONLY "
        "inference-visible predicted states"
    )

    print(
        f"📌 Device: {device}"
    )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

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
        args.m10_1a_checkpoint,
        EXPECTED_M10_SHA,
        "M10_1A",
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    # --------------------------------------------------------
    # TRAIN / VAL only
    # --------------------------------------------------------

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
        len(train_dataset),
    )

    print(
        "VAL_SAMPLES:",
        len(val_dataset),
    )

    # --------------------------------------------------------
    # Frozen M10-1a
    # --------------------------------------------------------

    (
        field_mean,
        field_std,
    ) = audit.base.build_field_stats(
        args.stats
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
        model,
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

    model.eval()

    print()
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

    # --------------------------------------------------------
    # Generate reliability dataset
    # --------------------------------------------------------

    print()
    print(
        "🔥 Generating TRAIN reliability rows..."
    )

    train_df = generate_rows(
        split_name="train",
        loader=train_loader,
        model=model,
        device=device,
        rollout_steps=args.rollout_steps,
        max_batches=(
            args.max_train_batches
        ),
    )

    print(
        "🔥 Generating VAL reliability rows..."
    )

    val_df = generate_rows(
        split_name="val",
        loader=val_loader,
        model=model,
        device=device,
        rollout_steps=args.rollout_steps,
        max_batches=(
            args.max_val_batches
        ),
    )

    if (
        len(train_df) == 0
        or len(val_df) == 0
    ):
        raise RuntimeError(
            "Reliability dataset is empty."
        )

    # --------------------------------------------------------
    # Basic reliability target behavior
    # --------------------------------------------------------

    target_summary = (
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
                "q_target",
                "size",
            ),
            q_mean=(
                "q_target",
                "mean",
            ),
            q_std=(
                "q_target",
                "std",
            ),
            path_b_rel_error_mean=(
                "path_b_relative_error",
                "mean",
            ),
            help_fraction=(
                "path_b_helps",
                "mean",
            ),
            utility_gain_mean=(
                "utility_gain",
                "mean",
            ),
            alpha_b_mean=(
                "alpha_b",
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
        "=" * 110
    )

    print(
        "RELIABILITY TARGET BY STEP"
    )

    print(
        "=" * 110
    )

    print(
        target_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Correlation of target with actual Path-B utility.
    # Focus on rollout-contaminated steps 2..4.
    # --------------------------------------------------------

    utility_rows = []

    for split_name, frame in [
        ("train", train_df),
        ("val", val_df),
    ]:

        sub = frame[
            frame["step"] >= 2
        ]

        utility_rows.append(
            {
                "split":
                    split_name,

                "rows":
                    len(sub),

                "spearman_q_vs_utility":
                    spearman_corr(
                        sub[
                            "q_target"
                        ].to_numpy(),
                        sub[
                            "utility_gain"
                        ].to_numpy(),
                    ),

                "spearman_alpha_vs_q":
                    spearman_corr(
                        sub[
                            "alpha_b"
                        ].to_numpy(),
                        sub[
                            "q_target"
                        ].to_numpy(),
                    ),

                "spearman_capscale_vs_q":
                    spearman_corr(
                        sub[
                            "path_b_scale"
                        ].to_numpy(),
                        sub[
                            "q_target"
                        ].to_numpy(),
                    ),

                "help_fraction":
                    float(
                        sub[
                            "path_b_helps"
                        ].mean()
                    ),
            }
        )

    utility_corr_df = pd.DataFrame(
        utility_rows
    )

    print()
    print(
        "=" * 110
    )

    print(
        "TARGET / EXISTING-GATE DIAGNOSTICS"
    )

    print(
        "Steps 2-4 only"
    )

    print(
        "=" * 110
    )

    print(
        utility_corr_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Reliability target stratification.
    #
    # VAL bins use TRAIN q thresholds.
    # No validation-derived threshold tuning.
    # --------------------------------------------------------

    train_roll = train_df[
        train_df["step"] >= 2
    ]

    val_roll = val_df[
        val_df["step"] >= 2
    ].copy()

    q_edges = np.quantile(
        train_roll[
            "q_target"
        ].to_numpy(),
        [
            0.25,
            0.50,
            0.75,
        ],
    )

    val_roll[
        "q_bin"
    ] = np.digitize(
        val_roll[
            "q_target"
        ].to_numpy(),
        q_edges,
        right=False,
    )

    bin_names = {
        0: "Q1-low",
        1: "Q2",
        2: "Q3",
        3: "Q4-high",
    }

    val_roll[
        "q_bin"
    ] = val_roll[
        "q_bin"
    ].map(
        bin_names
    )

    utility_bins = (
        val_roll.groupby(
            "q_bin",
            observed=False,
            as_index=False,
        )
        .agg(
            count=(
                "q_target",
                "size",
            ),
            q_mean=(
                "q_target",
                "mean",
            ),
            help_fraction=(
                "path_b_helps",
                "mean",
            ),
            utility_gain_mean=(
                "utility_gain",
                "mean",
            ),
        )
    )

    print()
    print(
        "=" * 110
    )

    print(
        "VAL UTILITY STRATIFIED BY TRUE RELIABILITY"
    )

    print(
        "Bins fixed from TRAIN q quartiles"
    )

    print(
        "=" * 110
    )

    print(
        utility_bins.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Fit tiny probes.
    #
    # IMPORTANT:
    # Step1 is excluded because pred state == GT state,
    # q≈1 trivially. We care about rollout-state reliability.
    # --------------------------------------------------------

    train_probe = train_df[
        train_df["step"] >= 2
    ].copy()

    val_probe = val_df[
        val_df["step"] >= 2
    ].copy()

    feature_sets = {
        "Legacy10":
            [
                f"legacy__{x}"
                for x in FEATURE_NAMES_LEGACY
            ],

        "LegacyCap11":
            [
                f"cap__{x}"
                for x in FEATURE_NAMES_CAP
            ],

        "Temporal19":
            [
                f"temporal__{x}"
                for x in FEATURE_NAMES_TEMPORAL
            ],
    }

    y_train = train_probe[
        "q_target"
    ].to_numpy()

    y_val = val_probe[
        "q_target"
    ].to_numpy()

    probe_rows = []

    prediction_store = {}

    # Constant train-mean baseline.
    constant_pred = np.full(
        len(y_val),
        float(
            np.mean(
                y_train
            )
        ),
        dtype=np.float64,
    )

    constant_metrics = (
        regression_metrics(
            y_val,
            constant_pred,
        )
    )

    probe_rows.append(
        {
            "probe":
                "ConstantMean",
            "num_features":
                0,
            **constant_metrics,
        }
    )

    for (
        probe_name,
        columns,
    ) in feature_sets.items():

        train_x = (
            train_probe[
                columns
            ].to_numpy()
        )

        val_x = (
            val_probe[
                columns
            ].to_numpy()
        )

        (
            train_pred,
            val_pred,
        ) = fit_probe(
            train_x=train_x,
            train_y=y_train,
            val_x=val_x,
            seed=args.seed,
            epochs=args.probe_epochs,
        )

        metrics = regression_metrics(
            y_val,
            val_pred,
        )

        probe_rows.append(
            {
                "probe":
                    probe_name,

                "num_features":
                    len(
                        columns
                    ),

                **metrics,
            }
        )

        prediction_store[
            probe_name
        ] = val_pred

    probe_df = pd.DataFrame(
        probe_rows
    )

    print()
    print(
        "=" * 110
    )

    print(
        "VAL RELIABILITY PROBE RESULTS"
    )

    print(
        "Steps 2-4 only"
    )

    print(
        "=" * 110
    )

    print(
        probe_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Does predicted reliability rank actual utility?
    # --------------------------------------------------------

    predicted_utility_rows = []

    for (
        probe_name,
        pred_q,
    ) in prediction_store.items():

        predicted_utility_rows.append(
            {
                "probe":
                    probe_name,

                "spearman_predq_vs_trueq":
                    spearman_corr(
                        pred_q,
                        y_val,
                    ),

                "spearman_predq_vs_utility":
                    spearman_corr(
                        pred_q,
                        val_probe[
                            "utility_gain"
                        ].to_numpy(),
                    ),
            }
        )

    predicted_utility_df = (
        pd.DataFrame(
            predicted_utility_rows
        )
    )

    print()
    print(
        "=" * 110
    )

    print(
        "PREDICTED RELIABILITY vs ACTUAL PATH-B UTILITY"
    )

    print(
        "=" * 110
    )

    print(
        predicted_utility_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    target_path = (
        prefix
        +
        "_target_summary.csv"
    )

    utility_corr_path = (
        prefix
        +
        "_target_diagnostics.csv"
    )

    bins_path = (
        prefix
        +
        "_utility_bins.csv"
    )

    probe_path = (
        prefix
        +
        "_probe_results.csv"
    )

    pred_util_path = (
        prefix
        +
        "_predicted_utility.csv"
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

    target_summary.to_csv(
        target_path,
        index=False,
    )

    utility_corr_df.to_csv(
        utility_corr_path,
        index=False,
    )

    utility_bins.to_csv(
        bins_path,
        index=False,
    )

    probe_df.to_csv(
        probe_path,
        index=False,
    )

    predicted_utility_df.to_csv(
        pred_util_path,
        index=False,
    )

    metadata = {
        "experiment":
            "M10-2a Path-B Reliability Feasibility Audit",

        "stage":
            "pre_candidate_control_diagnostic",

        "main_operator_training":
            False,

        "auxiliary_probe_training":
            True,

        "test_split_accessed":
            False,

        "rollout_generator":
            "frozen M10-1a StateParam RMSCap, B-only",

        "path_a_enabled":
            False,

        "reliability_target":
            "q=1/(1+RMS(PB_pred-PB_GT)/(RMS(PB_GT)+eps))",

        "feature_sets": {
            "Legacy10":
                FEATURE_NAMES_LEGACY,

            "LegacyCap11":
                FEATURE_NAMES_CAP,

            "Temporal19":
                FEATURE_NAMES_TEMPORAL,
        },

        "split_sha256":
            sha256_file(
                args.split
            ),

        "stats_sha256":
            sha256_file(
                args.stats
            ),

        "m10_1a_sha256":
            sha256_file(
                args.m10_1a_checkpoint
            ),

        "seed":
            args.seed,

        "rollout_steps":
            args.rollout_steps,

        "path_b_rms_cap":
            args.path_b_rms_cap,

        "probe_epochs":
            args.probe_epochs,

        "max_train_batches":
            args.max_train_batches,

        "max_val_batches":
            args.max_val_batches,
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
        train_path,
        val_path,
        target_path,
        utility_corr_path,
        bins_path,
        probe_path,
        pred_util_path,
        metadata_path,
    ]:
        print(
            " ",
            path,
        )


if __name__ == "__main__":
    main()
