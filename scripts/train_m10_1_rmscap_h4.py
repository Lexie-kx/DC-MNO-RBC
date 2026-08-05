import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from constants import FIELD_ORDER
from datasets.rbc_dataset import RBCDataset
from training.metrics import FieldWiseRelativeL2Loss

from models.operators.fno2d_m10_1_rmscap import (
    M10RMSCapTwoPathFNO2d,
)

from scripts.train_m6_fieldwise_encoder_h4 import (
    make_rollout_weights,
)


# ============================================================
# Dataset wrapper
# ============================================================

class M10MultiStepParamDataset(Dataset):
    """
    Unified M10-0 dataset wrapper.

    Every factorial arm receives the same dataset object:

        context_norm: [4, 4, H, W]
        y_seq_norm:   [S, 4, H, W]
        param:        [2]

    where:

        param = [log10(Ra), log10(Pr)]

    Static and State ignore param inside the model.
    Param and StateParam use it.

    Keeping return_params=True for ALL arms ensures the
    data pipeline itself is identical across the 2x2 study.
    """

    def __init__(
        self,
        base_dataset,
        context_length=4,
    ):
        self.base_dataset = base_dataset
        self.context_length = int(context_length)

    def __len__(self):
        return len(self.base_dataset)

    def _parse_sample(self, sample):
        x_norm = None
        y_seq_norm = None
        param = None

        if isinstance(sample, dict):

            for key in [
                "x_norm",
                "x",
                "context",
                "context_norm",
            ]:
                if key in sample:
                    x_norm = sample[key]
                    break

            for key in [
                "y_seq_norm",
                "y_seq",
                "target_sequence",
                "sequence",
            ]:
                if key in sample:
                    y_seq_norm = sample[key]
                    break

            for key in [
                "param",
                "params",
                "parameter",
                "parameters",
            ]:
                if key in sample:
                    param = sample[key]
                    break

        elif isinstance(sample, (tuple, list)):

            for obj in sample:

                if torch.is_tensor(obj):

                    if (
                        obj.ndim == 3
                        and obj.shape[0]
                        == self.context_length * 4
                    ):
                        x_norm = obj

                    elif (
                        obj.ndim == 4
                        and obj.shape[1] == 4
                    ):
                        y_seq_norm = obj

                    elif (
                        obj.ndim == 1
                        and obj.numel() == 2
                    ):
                        param = obj

                elif (
                    isinstance(obj, (tuple, list))
                    and len(obj) == 2
                ):
                    try:
                        maybe_param = torch.tensor(
                            obj,
                            dtype=torch.float32,
                        )

                        if (
                            maybe_param.ndim == 1
                            and maybe_param.numel() == 2
                        ):
                            param = maybe_param

                    except Exception:
                        pass

        else:
            raise TypeError(
                "Unsupported sample type from RBCDataset: "
                f"{type(sample)}"
            )

        if (
            x_norm is None
            or y_seq_norm is None
            or param is None
        ):
            raise RuntimeError(
                "Could not parse x_norm / y_seq_norm / param "
                "from RBCDataset output.\n"
                "M10-0 requires return_sequence=True and "
                "return_params=True.\n"
                f"sample type: {type(sample)}\n"
                f"sample repr: {repr(sample)[:500]}"
            )

        if not torch.is_tensor(param):
            param = torch.tensor(
                param,
                dtype=torch.float32,
            )

        return (
            x_norm,
            y_seq_norm,
            param.float(),
        )

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        (
            x_norm,
            y_seq_norm,
            param,
        ) = self._parse_sample(sample)

        c_total, h, w = x_norm.shape

        expected_channels = (
            self.context_length * 4
        )

        if c_total != expected_channels:
            raise RuntimeError(
                "Unexpected context channels: "
                f"expected {expected_channels}, "
                f"got {c_total}"
            )

        context_norm = x_norm.view(
            self.context_length,
            4,
            h,
            w,
        )

        return (
            context_norm,
            y_seq_norm,
            param,
        )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "M10-1a lightweight residual-stabilization control: "
            "M10-0 2Path + train-support Path-B RMS cap."
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--stats",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--m6_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "static",
            "param",
            "state",
            "stateparam",
        ],
    )

    parser.add_argument(
        "--run_name",
        type=str,
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

    # --------------------------------------------------------
    # Explicit screening grid convention.
    #
    # Keep exactly aligned with M10-Pre for M10-0 mechanism
    # screening. These values are deliberately passed through
    # CLI and stored in the checkpoint.
    # --------------------------------------------------------

    parser.add_argument(
        "--dx",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--dy",
        type=float,
        required=True,
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
        help=(
            "TRAIN-ONLY ground-truth Path-B RMS support cap. "
            "Use the split-specific value from M10-1-Pre audit."
        ),
    )

    parser.add_argument(
        "--path_b_rms_eps",
        type=float,
        default=1e-12,
        help="Numerical epsilon used by the RMS cap.",
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
        type=str,
        default="checkpoints/m10_0",
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

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ============================================================
# Audit helpers
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def build_field_stats(stats_path):
    with open(
        stats_path,
        "r",
        encoding="utf-8",
    ) as f:
        stats = json.load(f)

    field_mean = [
        float(stats[field]["mean"])
        for field in FIELD_ORDER
    ]

    field_std = [
        float(stats[field]["std"])
        for field in FIELD_ORDER
    ]

    return field_mean, field_std


def extract_m6_state(checkpoint):
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint["model_state_dict"]

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        return checkpoint["state_dict"]

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
        and isinstance(
            checkpoint["model"],
            dict,
        )
    ):
        return checkpoint["model"]

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
        "Could not locate M6 state_dict "
        "inside checkpoint."
    )


# ============================================================
# M10 H4 autoregressive loss
# ============================================================

def autoregressive_multistep_loss_m10(
    model,
    context_norm,
    y_seq_norm,
    param,
    criterion,
    rollout_weights,
):
    """
    context_norm:
        [B, T=4, C=4, H, W]

    y_seq_norm:
        [B, S=4, C=4, H, W]

    param:
        [B, 2]
        [log10(Ra), log10(Pr)]

    Free-autoregressive H4:

        model input
            ↓
        normalized delta
            ↓
        pred_next_norm
            =
        current_state_norm + pred_delta_norm
            ↓
        append prediction back into context
            ↓
        next rollout step

    No teacher forcing after the initial context.
    """

    (
        batch_size,
        context_len,
        channels,
        h,
        w,
    ) = context_norm.shape

    rollout_steps = y_seq_norm.shape[1]

    if context_len != 4:
        raise RuntimeError(
            f"Expected context_len=4, got {context_len}"
        )

    if channels != 4:
        raise RuntimeError(
            f"Expected 4 fields, got {channels}"
        )

    if rollout_steps != len(rollout_weights):
        raise RuntimeError(
            "rollout_steps and rollout_weights mismatch."
        )

    if (
        param.ndim != 2
        or param.shape[1] != 2
    ):
        raise RuntimeError(
            "param must be [B,2] = "
            "[log10(Ra), log10(Pr)], "
            f"got {tuple(param.shape)}"
        )

    context = context_norm

    loss_total = 0.0
    step_losses = []

    for step in range(rollout_steps):

        model_input = context.reshape(
            batch_size,
            context_len * channels,
            h,
            w,
        )

        pred_delta_norm = model(
            model_input,
            params=param,
        )

        current_state_norm = context[
            :,
            -1,
            :,
            :,
            :,
        ]

        pred_next_norm = (
            current_state_norm
            + pred_delta_norm
        )

        gt_next_norm = y_seq_norm[
            :,
            step,
            :,
            :,
            :,
        ]

        loss_step = criterion(
            pred_next_norm,
            gt_next_norm,
        )

        loss_total = (
            loss_total
            + rollout_weights[step]
            * loss_step
        )

        step_losses.append(
            loss_step.detach()
        )

        context = torch.cat(
            [
                context[
                    :,
                    1:,
                    :,
                    :,
                    :,
                ],
                pred_next_norm.unsqueeze(1),
            ],
            dim=1,
        )

    loss_total = (
        loss_total
        / rollout_weights.sum()
    )

    return (
        loss_total,
        step_losses,
    )


# ============================================================
# Validation
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    rollout_weights,
    device,
    max_batches=None,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

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

            context_norm = context_norm.to(
                device
            )

            y_seq_norm = y_seq_norm.to(
                device
            )

            param = param.to(
                device
            )

            (
                loss,
                _,
            ) = autoregressive_multistep_loss_m10(
                model=model,
                context_norm=context_norm,
                y_seq_norm=y_seq_norm,
                param=param,
                criterion=criterion,
                rollout_weights=rollout_weights,
            )

            bs = context_norm.shape[0]

            total_loss += (
                float(loss.item())
                * bs
            )

            total_samples += bs

    if total_samples == 0:
        raise RuntimeError(
            "Validation loader produced zero samples."
        )

    return (
        total_loss
        / total_samples
    )


# ============================================================
# Fixed cross-condition gate probe
# ============================================================

def build_probe_batch(
    dataset,
    probe_size=12,
):
    """
    Build a deterministic validation probe spread across
    the whole validation dataset rather than taking only
    the first condition group.
    """

    n = len(dataset)

    if n <= 0:
        raise RuntimeError(
            "Cannot build probe from empty dataset."
        )

    probe_size = min(
        int(probe_size),
        n,
    )

    indices = np.linspace(
        0,
        n - 1,
        num=probe_size,
        dtype=int,
    )

    contexts = []
    y_seqs = []
    params = []

    for idx in indices:
        (
            context_norm,
            y_seq_norm,
            param,
        ) = dataset[int(idx)]

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


def summarize_gate_probe(
    model,
    probe_batch,
    device,
):
    """
    Roll the fixed probe autoregressively for H4 and summarize
    the actual per-sample effective alpha values.

    Useful for detecting whether:

        Param      varies across parameter conditions,
        State      varies across states,
        StateParam varies across both.

    This is a diagnostic only, not a training loss.
    """

    model.eval()

    (
        context_norm,
        _,
        param,
    ) = probe_batch

    context = context_norm.to(
        device
    )

    param = param.to(
        device
    )

    alpha_a_all = []
    alpha_b_all = []
    dynamic_abs_all = []

    path_b_rms_raw_all = []
    path_b_rms_safe_all = []
    path_b_scale_all = []
    path_b_cap_active_all = []

    with torch.no_grad():

        for _ in range(4):

            (
                batch_size,
                context_len,
                channels,
                h,
                w,
            ) = context.shape

            model_input = context.reshape(
                batch_size,
                context_len * channels,
                h,
                w,
            )

            (
                pred_delta_norm,
                comp,
            ) = model(
                model_input,
                params=param,
                return_components=True,
            )

            alpha_a_all.append(
                comp["alpha_a"].detach().cpu()
            )

            alpha_b_all.append(
                comp["alpha_b"].detach().cpu()
            )

            dynamic_abs_all.append(
                comp["dynamic_raw"]
                .detach()
                .abs()
                .cpu()
                .reshape(-1)
            )

            path_b_rms_raw_all.append(
                comp["path_b_rms_raw"]
                .detach()
                .cpu()
                .reshape(-1)
            )

            path_b_rms_safe_all.append(
                comp["path_b_rms_safe"]
                .detach()
                .cpu()
                .reshape(-1)
            )

            path_b_scale_all.append(
                comp["path_b_scale"]
                .detach()
                .cpu()
                .reshape(-1)
            )

            path_b_cap_active_all.append(
                comp["path_b_cap_active"]
                .detach()
                .cpu()
                .reshape(-1)
            )

            current_state_norm = context[
                :,
                -1,
                :,
                :,
                :,
            ]

            pred_next_norm = (
                current_state_norm
                + pred_delta_norm
            )

            context = torch.cat(
                [
                    context[
                        :,
                        1:,
                        :,
                        :,
                        :,
                    ],
                    pred_next_norm.unsqueeze(1),
                ],
                dim=1,
            )

    alpha_a = torch.cat(
        alpha_a_all,
        dim=0,
    )

    alpha_b = torch.cat(
        alpha_b_all,
        dim=0,
    )

    dynamic_abs = torch.cat(
        dynamic_abs_all,
        dim=0,
    )

    path_b_rms_raw = torch.cat(
        path_b_rms_raw_all,
        dim=0,
    )

    path_b_rms_safe = torch.cat(
        path_b_rms_safe_all,
        dim=0,
    )

    path_b_scale = torch.cat(
        path_b_scale_all,
        dim=0,
    )

    path_b_cap_active = torch.cat(
        path_b_cap_active_all,
        dim=0,
    ).float()

    return {
        "alpha_a_mean": float(
            alpha_a.mean()
        ),
        "alpha_a_std": float(
            alpha_a.std(
                unbiased=False
            )
        ),
        "alpha_a_min": float(
            alpha_a.min()
        ),
        "alpha_a_max": float(
            alpha_a.max()
        ),

        "alpha_b_mean": float(
            alpha_b.mean()
        ),
        "alpha_b_std": float(
            alpha_b.std(
                unbiased=False
            )
        ),
        "alpha_b_min": float(
            alpha_b.min()
        ),
        "alpha_b_max": float(
            alpha_b.max()
        ),

        "dynamic_raw_abs_mean": float(
            dynamic_abs.mean()
        ),

        "path_b_rms_raw_mean": float(
            path_b_rms_raw.mean()
        ),

        "path_b_rms_raw_max": float(
            path_b_rms_raw.max()
        ),

        "path_b_rms_safe_mean": float(
            path_b_rms_safe.mean()
        ),

        "path_b_rms_safe_max": float(
            path_b_rms_safe.max()
        ),

        "path_b_scale_min": float(
            path_b_scale.min()
        ),

        "path_b_cap_active_fraction": float(
            path_b_cap_active.mean()
        ),

        "base_raw_alpha_a": float(
            model.base_raw_alpha_a
            .detach()
            .cpu()
        ),

        "base_raw_alpha_b": float(
            model.base_raw_alpha_b
            .detach()
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
    m6_sha256,
    split_sha256,
    stats_sha256,
    trainable_names,
    trainable_numel,
    gate_summary,
):
    payload = {
        "experiment": (
            "M10-1a-PathB-RMSCap-2Path-H4"
        ),

        "stage": (
            "lightweight_residual_stabilization_control"
        ),

        "parent_experiment": (
            "M10-0-Structured-2Path-H4"
        ),

        "factorial_mode": args.mode,

        "epoch": int(epoch),

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            None
            if optimizer is None
            else optimizer.state_dict()
        ),

        "val_loss": float(
            val_loss
        ),

        "best_val_loss": float(
            best_val_loss
        ),

        "best_epoch": int(
            best_epoch
        ),

        # ----------------------------------------------------
        # Fixed model contract
        # ----------------------------------------------------

        "backbone": (
            "audited_frozen_M6_FieldWiseEncoder_H4"
        ),

        "backbone_frozen": True,

        "physics_topology": {
            "path_a": (
                "horizontal buoyancy anomaly "
                "b-mean_x(b) -> u_y normalized delta"
            ),
            "path_b": (
                "-u_dot_grad_b -> buoyancy normalized delta"
            ),
            "direct_ux_path": False,
            "direct_pressure_path": False,
        },

        "conditioner": {
            "mode": args.mode,
            "hidden_dim": int(
                args.conditioner_hidden
            ),
            "parameter_format": (
                "[log10(Ra), log10(Pr)]"
            ),
            "dynamic_output_zero_initialized": True,
        },

        "gate_parameterization": (
            "alpha_max * tanh("
            "base_raw + tanh(conditioner(condition)))"
        ),

        "alpha_max": float(
            args.alpha_max
        ),

        "stabilization": {
            "target": "Path-B residual injection only",
            "type": "per_sample_spatial_rms_cap",
            "support_source": (
                "split-specific TRAIN-ONLY ground-truth audit"
            ),
            "path_b_rms_cap": float(
                args.path_b_rms_cap
            ),
            "path_b_rms_eps": float(
                args.path_b_rms_eps
            ),
            "state_conditioner_uses_raw_path_b": True,
            "path_a_capped": False,
        },

        # ----------------------------------------------------
        # Grid convention
        # ----------------------------------------------------

        "dx": float(
            args.dx
        ),

        "dy": float(
            args.dy
        ),

        "grid_spacing_status": (
            "M10-0 screening convention; "
            "kept identical to M10-Pre; "
            "exact preprocessing provenance not yet source-proven"
        ),

        # ----------------------------------------------------
        # Training protocol
        # ----------------------------------------------------

        "prediction_type": (
            "normalized_delta"
        ),

        "delta_definition": (
            "pred_next_norm = "
            "current_state_norm + pred_delta_norm"
        ),

        "training_type": (
            "H4 free-autoregressive"
        ),

        "loss": (
            "weighted H4 free-autoregressive "
            "FieldWiseRelativeL2Loss"
        ),

        "pde_loss": False,

        "rollout_steps": int(
            args.rollout_steps
        ),

        "rollout_weights": [
            float(x)
            for x in make_rollout_weights(
                args.rollout_steps
            ).tolist()
        ],

        "epochs": int(
            args.epochs
        ),

        "batch_size": int(
            args.batch_size
        ),

        "learning_rate": float(
            args.lr
        ),

        "optimizer": "Adam",

        "weight_decay": 0.0,

        "seed": int(
            args.seed
        ),

        # ----------------------------------------------------
        # Data / provenance
        # ----------------------------------------------------

        "split": args.split,
        "split_sha256": (
            split_sha256
        ),

        "stats": args.stats,
        "stats_sha256": (
            stats_sha256
        ),

        "m6_checkpoint": (
            args.m6_checkpoint
        ),

        "m6_checkpoint_sha256": (
            m6_sha256
        ),

        # ----------------------------------------------------
        # Trainable audit
        # ----------------------------------------------------

        "trainable_parameter_names": (
            trainable_names
        ),

        "trainable_parameter_count": int(
            trainable_numel
        ),

        "frozen_m6_parameter_tensors": 38,

        # ----------------------------------------------------
        # Mechanism diagnostic
        # ----------------------------------------------------

        "gate_probe_summary": (
            gate_summary
        ),
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

    if args.rollout_steps != 4:
        raise ValueError(
            "M10-1a protocol is locked "
            "to rollout_steps=4."
        )

    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if args.lr <= 0:
        raise ValueError(
            "lr must be positive."
        )

    if args.path_b_rms_cap <= 0:
        raise ValueError(
            "path_b_rms_cap must be positive."
        )

    if args.path_b_rms_eps <= 0:
        raise ValueError(
            "path_b_rms_eps must be positive."
        )

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # Experiment identity
    # ========================================================

    experiment_labels = {
        "static": (
            "M10-1a Static-PathB-RMSCap-2Path-H4"
        ),
        "param": (
            "M10-1a Param-PathB-RMSCap-2Path-H4"
        ),
        "state": (
            "M10-1a State-PathB-RMSCap-2Path-H4"
        ),
        "stateparam": (
            "M10-1a StateParam-PathB-RMSCap-2Path-H4"
        ),
    }

    experiment_name = (
        experiment_labels[
            args.mode
        ]
    )

    print(
        f"🚀 [{experiment_name}]"
    )

    print(
        "📌 Stage: M10-1a lightweight "
        "Path-B residual stabilization control"
    )

    print(
        "📌 Device:",
        device,
    )

    print(
        "📌 Mode:",
        args.mode,
    )

    print(
        "📌 Seed:",
        args.seed,
    )

    print(
        "📌 Epochs:",
        args.epochs,
    )

    print(
        "📌 Batch size:",
        args.batch_size,
    )

    print(
        "📌 LR:",
        args.lr,
    )

    print(
        "📌 H4 rollout weights: "
        "[1.0, 0.8, 0.6, 0.4]"
    )

    print(
        "📌 Frozen audited M6 backbone"
    )

    print(
        "📌 Fixed Path A: "
        "b-mean_x(b) -> u_y"
    )

    print(
        "📌 Fixed Path B: "
        "-u·grad(b) -> buoyancy"
    )

    print(
        "📌 No direct pressure residual"
    )

    print(
        "📌 No PDE loss"
    )

    print(
        "📌 dx:",
        args.dx,
    )

    print(
        "📌 dy:",
        args.dy,
    )

    print(
        "📌 alpha_max:",
        args.alpha_max,
    )

    print(
        "📌 conditioner_hidden:",
        args.conditioner_hidden,
    )

    print(
        "📌 Path-B RMS cap:",
        args.path_b_rms_cap,
    )

    print(
        "📌 Path-B RMS eps:",
        args.path_b_rms_eps,
    )

    print(
        "📌 Cap source: split-specific TRAIN-ONLY "
        "ground-truth support audit"
    )

    print(
        "📌 State conditioner uses RAW Path-B; "
        "cap applies only to residual injection"
    )

    # ========================================================
    # Provenance
    # ========================================================

    for path in [
        args.split,
        args.stats,
        args.m6_checkpoint,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                path
            )

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

    # ========================================================
    # Data
    # ========================================================

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    (
        field_mean,
        field_std,
    ) = build_field_stats(
        args.stats
    )

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

    # --------------------------------------------------------
    # Explicit independent DataLoader RNG.
    #
    # Same seed + same split means all four factorial arms see
    # the same shuffled sample order.
    # --------------------------------------------------------

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

    print(
        "PARAM_EXAMPLE:",
        sample_param.tolist(),
    )

    # ========================================================
    # Fixed cross-condition gate probe
    # ========================================================

    probe_batch = build_probe_batch(
        val_dataset,
        probe_size=12,
    )

    # ========================================================
    # Model
    # ========================================================

    set_seed(
        args.seed
    )

    model = M10RMSCapTwoPathFNO2d(
        field_mean=field_mean,
        field_std=field_std,
        mode=args.mode,
        dx=args.dx,
        dy=args.dy,
        path_b_rms_cap=(
            args.path_b_rms_cap
        ),
        path_b_rms_eps=(
            args.path_b_rms_eps
        ),
        alpha_max=args.alpha_max,
        conditioner_hidden=(
            args.conditioner_hidden
        ),
        freeze_m6=True,
    ).to(device)

    m6_checkpoint = torch.load(
        args.m6_checkpoint,
        map_location=device,
    )

    m6_state = extract_m6_state(
        m6_checkpoint
    )

    model.load_m6_state_dict(
        m6_state
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

    expected_numel = {
        "static": 2,
        "param": 164,
        "state": 356,
        "stateparam": 420,
    }[
        args.mode
    ]

    if (
        trainable_numel
        != expected_numel
    ):
        raise RuntimeError(
            "Unexpected trainable parameter count.\n"
            f"Mode: {args.mode}\n"
            f"Expected: {expected_numel}\n"
            f"Actual: {trainable_numel}"
        )

    frozen_m6_tensors = (
        model.frozen_m6_parameter_tensor_count()
    )

    m6_trainable = [
        name
        for name, parameter
        in model.m6.named_parameters()
        if parameter.requires_grad
    ]

    if frozen_m6_tensors != 38:
        raise RuntimeError(
            "Expected 38 frozen M6 "
            "parameter tensors, got "
            f"{frozen_m6_tensors}"
        )

    if m6_trainable:
        raise RuntimeError(
            "M6 backbone is not fully frozen: "
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
    # Training objects
    # ========================================================

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    criterion = (
        FieldWiseRelativeL2Loss()
    )

    rollout_weights = (
        make_rollout_weights(
            args.rollout_steps
        ).to(device)
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
    # Epoch 0
    #
    # All four modes MUST equal pure M6 here.
    # Epoch 0 participates in model selection.
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
        m6_sha256=m6_sha,
        split_sha256=split_sha,
        stats_sha256=stats_sha,
        trainable_names=trainable_names,
        trainable_numel=trainable_numel,
        gate_summary=gate_summary,
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

        # Frozen backbone stays explicitly deterministic.
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
            ) = autoregressive_multistep_loss_m10(
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
                    "Non-finite training loss "
                    f"at epoch={epoch}, "
                    f"batch={batch_idx}: {loss}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()

            bs = context_norm.shape[0]

            train_loss_sum += (
                float(loss.item())
                * bs
            )

            train_samples += bs

        if train_samples == 0:
            raise RuntimeError(
                "Training loader produced zero samples."
            )

        train_loss = (
            train_loss_sum
            / train_samples
        )

        # ----------------------------------------------------
        # Full validation
        # ----------------------------------------------------

        val_loss = evaluate(
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

        improved = (
            val_loss
            < best_val_loss
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
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                args=args,
                m6_sha256=m6_sha,
                split_sha256=split_sha,
                stats_sha256=stats_sha,
                trainable_names=(
                    trainable_names
                ),
                trainable_numel=(
                    trainable_numel
                ),
                gate_summary=(
                    gate_summary
                ),
            )

        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            args=args,
            m6_sha256=m6_sha,
            split_sha256=split_sha,
            stats_sha256=stats_sha,
            trainable_names=(
                trainable_names
            ),
            trainable_numel=(
                trainable_numel
            ),
            gate_summary=(
                gate_summary
            ),
        )

        print(
            f"[Epoch "
            f"{epoch:02d}/"
            f"{args.epochs:02d}] "
            f"train={train_loss:.12f} "
            f"val={val_loss:.12f} "
            f"A={gate_summary['alpha_a_mean']:+.6e}"
            f"±{gate_summary['alpha_a_std']:.2e} "
            f"B={gate_summary['alpha_b_mean']:+.6e}"
            f"±{gate_summary['alpha_b_std']:.2e} "
            f"dyn={gate_summary['dynamic_raw_abs_mean']:.3e} "
            f"capFrac={gate_summary['path_b_cap_active_fraction']:.3f} "
            f"rawBmax={gate_summary['path_b_rms_raw_max']:.3e} "
            f"safeBmax={gate_summary['path_b_rms_safe_max']:.3e} "
            f"minScale={gate_summary['path_b_scale_min']:.3e} "
            f"best_epoch={best_epoch}"
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
        "========== M10-1a COMPLETE =========="
    )

    print(
        "EXPERIMENT:",
        experiment_name,
    )

    print(
        "MODE:",
        args.mode,
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
