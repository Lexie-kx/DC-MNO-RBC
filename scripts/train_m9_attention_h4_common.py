"""
Shared training implementation for M9-0a and M9-0b.

This file is shared infrastructure, not a third experiment.

M9-0a:
    Static Softmax field-axis attention.
    Controlled experiment.

M9-0b:
    State-dependent Q/K field-axis attention.
    Formal attention baseline candidate.

Shared protocol:
- M6-FieldWiseEncoder-H4 warm start
- normalized delta prediction
- H4 autoregressive training
- rollout weights [1.0, 0.8, 0.6, 0.4]
- FieldWiseRelativeL2Loss on predicted normalized state
- no ParameterToken
- no parameter conditioning
- no physics prior
- no PDE loss
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from datasets.rbc_dataset import RBCDataset
from models.operators.fno2d_m9_static_attention import (
    M9StaticAttentionFNO2d,
)
from models.operators.fno2d_m9_state_attention import (
    M9StateAttentionFNO2d,
)
from training.metrics import FieldWiseRelativeL2Loss


FIELD_NAMES = [
    "buoyancy",
    "u_x",
    "u_y",
    "pressure",
]

EPS = 1e-12


class MultiStepDeltaDataset(Dataset):
    """
    Convert RBCDataset output into H4 autoregressive format.

    Base dataset:
        x_norm:     [16, H, W]
        y_norm:     [4, H, W], unused
        y_seq_norm: [S, 4, H, W]

    Returned:
        context_norm: [4, 4, H, W]
        y_seq_norm:   [S, 4, H, W]
    """

    def __init__(
        self,
        base_dataset: Dataset,
        context_length: int = 4,
    ) -> None:
        self.base_dataset = base_dataset
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm, _y_norm, y_seq_norm = (
            self.base_dataset[idx]
        )

        channels, height, width = x_norm.shape
        expected_channels = self.context_length * 4

        if channels != expected_channels:
            raise ValueError(
                f"x_norm channels={channels}, "
                f"expected {expected_channels}"
            )

        context_norm = x_norm.reshape(
            self.context_length,
            4,
            height,
            width,
        )

        return context_norm, y_seq_norm


def build_parser(
    variant: str,
) -> argparse.ArgumentParser:
    if variant == "static":
        experiment_name = (
            "M9-0a-StaticSoftmaxFieldAttention-H4"
        )
    elif variant == "state":
        experiment_name = (
            "M9-0b-StateDependentFieldAttention-H4"
        )
    else:
        raise ValueError(
            f"Unknown variant: {variant}"
        )

    parser = argparse.ArgumentParser(
        description=f"Train {experiment_name}."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="data/splits/unseen_pr_split.json",
    )

    parser.add_argument(
        "--stats",
        type=str,
        default=(
            "data/stats/"
            "rbc_field_stats_unseen_pr.json"
        ),
    )

    parser.add_argument(
        "--init_ckpt",
        type=str,
        required=True,
        help=(
            "Corresponding M6-FieldWiseEncoder-H4 "
            "checkpoint. Required for fair warm start."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "configs/"
            "m9_0_attention_frozen.json"
        ),
    )

    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints/cross_param",
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
        default=3e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--eta_min",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260717,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Lightweight trial only.",
    )

    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Lightweight trial only.",
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )

    return parser


def resolve_path(
    path_value: str,
) -> Path:
    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_frozen_config(
    config: dict[str, Any],
) -> None:
    required_values = {
        "score_scale_mode": "sqrt_token_dim",
        "score_temperature": 0.03,
        "attention_output_scale": 0.05675224,
        "qk_init_std": 0.02,
        "shared_scale_for_static_and_dynamic": True,
        "attention_output_scale_trainable": False,
        "mask_self_attention": True,
        "parameter_token": False,
        "parameter_condition_token": False,
        "physics_prior": False,
        "pde_loss": False,
    }

    for key, expected in required_values.items():
        if key not in config:
            raise KeyError(
                f"Frozen config missing key: {key}"
            )

        actual = config[key]

        if isinstance(expected, float):
            if not math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Frozen config mismatch: "
                    f"{key}={actual}, "
                    f"expected {expected}"
                )
        elif actual != expected:
            raise ValueError(
                f"Frozen config mismatch: "
                f"{key}={actual!r}, "
                f"expected {expected!r}"
            )

    if config["field_order"] != FIELD_NAMES:
        raise ValueError(
            "Field order mismatch in frozen config."
        )


def make_rollout_weights(
    rollout_steps: int,
) -> torch.Tensor:
    if rollout_steps != 4:
        raise ValueError(
            "M9-0 H4 formal protocol requires "
            "rollout_steps=4."
        )

    return torch.tensor(
        [1.0, 0.8, 0.6, 0.4],
        dtype=torch.float32,
    )


def build_model(
    variant: str,
    config: dict[str, Any],
) -> torch.nn.Module:
    common_kwargs = {
        "in_channels": 16,
        "out_channels": 4,
        "modes1": 16,
        "modes2": 16,
        "width": 32,
        "context_length": 4,
        "num_fields": 4,
        "field_width": int(
            config["field_width"]
        ),
        "attention_hidden_channels": int(
            config["hidden_channels"]
        ),
        "attention_dropout": float(
            config["coupling_dropout"]
        ),
        "attention_init_gate": float(
            config["residual_gate_init_logit"]
        ),
        "attention_use_norm": bool(
            config["use_norm"]
        ),
        "attention_output_scale": float(
            config["attention_output_scale"]
        ),
        "score_temperature": float(
            config["score_temperature"]
        ),
    }

    if variant == "static":
        return M9StaticAttentionFNO2d(
            **common_kwargs
        )

    if variant == "state":
        return M9StateAttentionFNO2d(
            **common_kwargs,
            qk_channels=int(
                config["qk_channels"]
            ),
            score_scale_mode=str(
                config["score_scale_mode"]
            ),
            qk_init_std=float(
                config["qk_init_std"]
            ),
        )

    raise ValueError(
        f"Unknown variant: {variant}"
    )


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint["model_state_dict"]

    return checkpoint


def load_m6_warm_start(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    m6_state = extract_state_dict(
        checkpoint
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            m6_state,
            strict=False,
        )
    )

    print(
        "✅ M6 warm-start load completed "
        "with strict=False"
    )
    print(
        "   missing_keys:",
        missing_keys,
    )
    print(
        "   unexpected_keys:",
        unexpected_keys,
    )

    if unexpected_keys:
        raise RuntimeError(
            "Unexpected M6 keys are not allowed: "
            f"{unexpected_keys}"
        )

    bad_missing = [
        key
        for key in missing_keys
        if not key.startswith(
            "field_attention."
        )
    ]

    if bad_missing:
        raise RuntimeError(
            "Only field_attention.* may be "
            "missing during M6 -> M9 loading. "
            f"Bad keys: {bad_missing}"
        )

    model_state = model.state_dict()
    mismatched = []

    for key, old_value in m6_state.items():
        if key not in model_state:
            mismatched.append(
                f"old key absent: {key}"
            )
            continue

        if not torch.equal(
            old_value.detach().cpu(),
            model_state[key].detach().cpu(),
        ):
            mismatched.append(
                f"value mismatch: {key}"
            )

    if mismatched:
        raise RuntimeError(
            "M6 warm-start tensor mismatch:\n"
            + "\n".join(
                mismatched[:20]
            )
        )

    print(
        "✅ Every M6 tensor was copied exactly."
    )


def autoregressive_multistep_loss(
    model: torch.nn.Module,
    context_norm: torch.Tensor,
    y_seq_norm: torch.Tensor,
    criterion: torch.nn.Module,
    rollout_weights: torch.Tensor,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
]:
    """
    H4 autoregressive normalized-delta training.

    At each step:
        pred_delta_norm = model(context)
        pred_next_norm = current + pred_delta_norm
        loss(pred_next_norm, gt_next_norm)
        append pred_next_norm into context
    """
    (
        batch_size,
        context_length,
        channels,
        height,
        width,
    ) = context_norm.shape

    rollout_steps = y_seq_norm.shape[1]

    if context_length != 4:
        raise ValueError(
            f"context_length={context_length}, "
            "expected 4"
        )

    if channels != 4:
        raise ValueError(
            f"field channels={channels}, expected 4"
        )

    if rollout_steps != len(
        rollout_weights
    ):
        raise ValueError(
            "Rollout sequence and weights mismatch."
        )

    context = context_norm
    total_loss = torch.zeros(
        (),
        device=context.device,
        dtype=context.dtype,
    )
    step_losses = []

    for step in range(rollout_steps):
        model_input = context.reshape(
            batch_size,
            context_length * channels,
            height,
            width,
        )

        pred_delta_norm = model(
            model_input
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

        step_loss = criterion(
            pred_next_norm,
            gt_next_norm,
        )

        total_loss = (
            total_loss
            + rollout_weights[step]
            * step_loss
        )

        step_losses.append(
            step_loss.detach()
        )

        context = torch.cat(
            [
                context[:, 1:, :, :, :],
                pred_next_norm.unsqueeze(1),
            ],
            dim=1,
        )

    total_loss = (
        total_loss
        / rollout_weights.sum()
    )

    return total_loss, step_losses


def subset_grad_norm(
    model: torch.nn.Module,
    predicate,
) -> float:
    total = 0.0

    for name, parameter in (
        model.named_parameters()
    ):
        if (
            predicate(name)
            and parameter.grad is not None
        ):
            grad = (
                parameter.grad
                .detach()
                .float()
            )

            total += float(
                grad.square()
                .sum()
                .item()
            )

    return math.sqrt(total)


def attention_grad_norm(
    model: torch.nn.Module,
) -> float:
    return subset_grad_norm(
        model,
        lambda name: name.startswith(
            "field_attention."
        ),
    )


def score_parameter_grad_norm(
    model: torch.nn.Module,
    variant: str,
) -> float:
    if variant == "static":
        return subset_grad_norm(
            model,
            lambda name: (
                name
                == (
                    "field_attention."
                    "static_attention_logits"
                )
            ),
        )

    return subset_grad_norm(
        model,
        lambda name: (
            name.startswith(
                "field_attention.q_proj."
            )
            or name.startswith(
                "field_attention.k_proj."
            )
        ),
    )


@torch.no_grad()
def summarize_attention(
    model: torch.nn.Module,
    probe_context: torch.Tensor,
) -> dict[str, Any]:
    model.eval()

    (
        batch_size,
        context_length,
        channels,
        height,
        width,
    ) = probe_context.shape

    model_input = probe_context.reshape(
        batch_size,
        context_length * channels,
        height,
        width,
    )

    (
        _fused,
        field_features,
        _attended_features,
        diagnostics,
    ) = model.encode_fields(
        model_input,
        return_attention=True,
    )

    if diagnostics is None:
        raise RuntimeError(
            "Attention diagnostics were not returned."
        )

    attention = (
        diagnostics["attention"]
        .detach()
        .float()
    )

    scores = (
        diagnostics["scores"]
        .detach()
        .float()
    )

    branch_update = (
        diagnostics["branch_update"]
        .detach()
        .float()
    )

    gate = (
        model.field_attention
        .gate_values()
        .detach()
        .float()
    )

    field_stack = torch.stack(
        field_features,
        dim=1,
    ).float()

    safe_attention = attention.clamp_min(
        1e-12
    )

    entropy = -(
        safe_attention
        * safe_attention.log()
    ).sum(dim=-1)

    normalized_entropy = (
        entropy
        / math.log(
            attention.shape[-1] - 1
        )
    )

    source_max = attention.max(
        dim=-1
    ).values

    branch_rms = (
        branch_update
        .square()
        .mean()
        .sqrt()
    )

    feature_rms = (
        field_stack
        .square()
        .mean()
        .sqrt()
    )

    mean_attention = attention.mean(
        dim=0
    )

    return {
        "gate_mean": float(
            gate.mean().item()
        ),
        "gate_min": float(
            gate.min().item()
        ),
        "gate_max": float(
            gate.max().item()
        ),
        "score_std": float(
            scores.std(
                unbiased=False
            ).item()
        ),
        "attention_sample_std": float(
            attention.std(
                dim=0,
                unbiased=False,
            ).mean().item()
        ),
        "attention_entropy": float(
            normalized_entropy.mean().item()
        ),
        "attention_max_mean": float(
            source_max.mean().item()
        ),
        "branch_relative_rms": float(
            (
                branch_rms
                / (feature_rms + EPS)
            ).item()
        ),
        "mean_attention": (
            mean_attention
            .cpu()
            .tolist()
        ),
    }


def print_attention_matrix(
    matrix: list[list[float]],
) -> None:
    print(
        "      source -> "
        "buoyancy      u_x      u_y  pressure"
    )

    for target_idx, row in enumerate(
        matrix
    ):
        values = " ".join(
            f"{value:8.5f}"
            for value in row
        )

        print(
            f"target {FIELD_NAMES[target_idx]:10s} "
            f"{values}"
        )


def variant_metadata(
    variant: str,
) -> dict[str, str]:
    if variant == "static":
        return {
            "experiment": (
                "M9-0a-StaticSoftmax"
                "FieldAttention-H4"
            ),
            "experiment_type": (
                "controlled_experiment"
            ),
            "architecture": (
                "FieldWiseEncoder + "
                "static Softmax field attention"
            ),
            "upgrade": (
                "globally shared trainable "
                "off-diagonal 4x4 attention"
            ),
        }

    return {
        "experiment": (
            "M9-0b-StateDependent"
            "FieldAttention-H4"
        ),
        "experiment_type": (
            "formal_attention_baseline_candidate"
        ),
        "architecture": (
            "FieldWiseEncoder + "
            "state-dependent Q/K field attention"
        ),
        "upgrade": (
            "sample-dependent off-diagonal "
            "4x4 attention generated from state"
        ),
    }


def main_for_variant(
    variant: str,
) -> None:
    parser = build_parser(variant)
    args = parser.parse_args()

    if args.rollout_steps != 4:
        raise ValueError(
            "This is an H4 script. "
            "--rollout_steps must be 4."
        )

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be positive."
        )

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    split_path = resolve_path(
        args.split
    )
    stats_path = resolve_path(
        args.stats
    )
    init_ckpt_path = resolve_path(
        args.init_ckpt
    )
    config_path = resolve_path(
        args.config
    )
    ckpt_dir = resolve_path(
        args.ckpt_dir
    )

    for path in (
        split_path,
        stats_path,
        init_ckpt_path,
        config_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_save_path = (
        ckpt_dir
        / f"{args.run_name}_best.pth"
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        frozen_config = json.load(file)

    validate_frozen_config(
        frozen_config
    )

    metadata = variant_metadata(
        variant
    )

    lightweight_trial = (
        args.max_train_batches is not None
        or args.max_val_batches is not None
        or args.epochs == 1
    )

    print(
        "========== M9 ATTENTION H4 TRAINING =========="
    )
    print(
        "Experiment:",
        metadata["experiment"],
    )
    print(
        "Type:",
        metadata["experiment_type"],
    )
    print(
        "Mode:",
        (
            "LIGHTWEIGHT TRAINING-SCRIPT TRIAL"
            if lightweight_trial
            else "FORMAL TRAINING"
        ),
    )
    print("Device:", device)
    print("Seed:", args.seed)
    print("Split:", split_path)
    print("Stats:", stats_path)
    print(
        "M6 warm start:",
        init_ckpt_path,
    )
    print(
        "Frozen config:",
        config_path,
    )
    print(
        "Run name:",
        args.run_name,
    )
    print(
        "Best checkpoint:",
        best_save_path,
    )
    print(
        "Temperature:",
        frozen_config[
            "score_temperature"
        ],
    )
    print(
        "Score scaling:",
        frozen_config[
            "score_scale_mode"
        ],
    )
    print(
        "Shared lambda:",
        frozen_config[
            "attention_output_scale"
        ],
    )
    print(
        "Rollout weights:",
        [1.0, 0.8, 0.6, 0.4],
    )
    print("No ParameterToken")
    print("No parameter conditioning")
    print("No physics prior")
    print("No PDE loss")

    if args.no_wandb:
        os.environ["WANDB_MODE"] = (
            "disabled"
        )

    wandb.init(
        project="DC-MNO",
        name=args.run_name,
        config={
            **metadata,
            "base_model": (
                "M6-FieldWiseEncoder-H4"
            ),
            "variant": variant,
            "prediction_type": (
                "normalized_delta"
            ),
            "task": (
                "autoregressive_multi_step_"
                "delta_prediction"
            ),
            "rollout_steps": 4,
            "rollout_weights": [
                1.0,
                0.8,
                0.6,
                0.4,
            ],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": (
                args.weight_decay
            ),
            "eta_min": args.eta_min,
            "grad_clip": 1.0,
            "seed": args.seed,
            "split": str(split_path),
            "stats": str(stats_path),
            "init_ckpt": str(
                init_ckpt_path
            ),
            "frozen_config": (
                frozen_config
            ),
            "max_train_batches": (
                args.max_train_batches
            ),
            "max_val_batches": (
                args.max_val_batches
            ),
        },
    )

    with split_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        split_config = json.load(file)

    print(
        "📦 Loading datasets: "
        "return_sequence=True, target_steps=4"
    )

    train_base_dataset = RBCDataset(
        split_config=split_config["train"],
        normalize=True,
        stats_path=str(stats_path),
        return_sequence=True,
        target_steps=4,
    )

    val_base_dataset = RBCDataset(
        split_config=split_config["val"],
        normalize=True,
        stats_path=str(stats_path),
        return_sequence=True,
        target_steps=4,
    )

    train_dataset = MultiStepDeltaDataset(
        train_base_dataset,
        context_length=4,
    )

    val_dataset = MultiStepDeltaDataset(
        val_base_dataset,
        context_length=4,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    print(
        "Train samples:",
        len(train_dataset),
    )
    print(
        "Val samples:",
        len(val_dataset),
    )

    sample_context, sample_y_seq = next(
        iter(train_loader)
    )

    print(
        "Context shape:",
        tuple(sample_context.shape),
    )
    print(
        "Y_seq shape:",
        tuple(sample_y_seq.shape),
    )

    if sample_context.shape[1] != 4:
        raise RuntimeError(
            "Context time length is not 4."
        )

    if sample_y_seq.shape[1] != 4:
        raise RuntimeError(
            "Target rollout length is not 4."
        )

    probe_context, _probe_y_seq = next(
        iter(val_loader)
    )

    probe_context = probe_context[
        : min(4, probe_context.shape[0])
    ].to(device)

    set_seed(args.seed)

    model = build_model(
        variant=variant,
        config=frozen_config,
    ).to(device)

    load_m6_warm_start(
        model=model,
        checkpoint_path=init_ckpt_path,
        device=device,
    )

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    attention_params = sum(
        parameter.numel()
        for name, parameter
        in model.named_parameters()
        if name.startswith(
            "field_attention."
        )
    )

    print(
        "Total parameters:",
        f"{total_params:,}",
    )
    print(
        "New attention parameters:",
        f"{attention_params:,}",
    )

    initial_attention = (
        summarize_attention(
            model,
            probe_context,
        )
    )

    print(
        "Initial attention summary:"
    )
    print(
        "   Gate mean:",
        f"{initial_attention['gate_mean']:.8f}",
    )
    print(
        "   Score std:",
        f"{initial_attention['score_std']:.8e}",
    )
    print(
        "   Sample std:",
        (
            f"{initial_attention['attention_sample_std']:.8e}"
        ),
    )
    print(
        "   Normalized entropy:",
        f"{initial_attention['attention_entropy']:.8f}",
    )
    print(
        "   Attention max mean:",
        f"{initial_attention['attention_max_mean']:.8f}",
    )
    print(
        "   Branch relative RMS:",
        f"{initial_attention['branch_relative_rms']:.8e}",
    )

    criterion = FieldWiseRelativeL2Loss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.eta_min,
        )
    )

    rollout_weights = (
        make_rollout_weights(4)
        .to(device)
    )

    best_val_loss = float("inf")
    start_time = time.time()

    print(
        "\n🔥 Starting H4 autoregressive training..."
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        train_loss_sum = 0.0
        train_count = 0
        train_batch_count = 0

        total_grad_norm_sum = 0.0
        attention_grad_norm_sum = 0.0
        score_grad_norm_sum = 0.0

        train_step_loss_sum = (
            torch.zeros(
                4,
                dtype=torch.float64,
            )
        )

        for batch_idx, (
            batch_context,
            batch_y_seq,
        ) in enumerate(
            train_loader,
            start=1,
        ):
            batch_context = (
                batch_context.to(device)
            )
            batch_y_seq = (
                batch_y_seq.to(device)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss, step_losses = (
                autoregressive_multistep_loss(
                    model=model,
                    context_norm=batch_context,
                    y_seq_norm=batch_y_seq,
                    criterion=criterion,
                    rollout_weights=(
                        rollout_weights
                    ),
                )
            )

            if not torch.isfinite(loss):
                print(
                    "⚠️ NaN/Inf loss detected; "
                    "skipping batch."
                )
                continue

            loss.backward()

            current_attention_grad = (
                attention_grad_norm(model)
            )

            current_score_grad = (
                score_parameter_grad_norm(
                    model,
                    variant,
                )
            )

            total_grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
            )

            optimizer.step()

            current_batch_size = (
                batch_context.shape[0]
            )

            train_loss_sum += (
                float(loss.item())
                * current_batch_size
            )

            train_count += (
                current_batch_size
            )
            train_batch_count += 1

            total_grad_norm_sum += float(
                total_grad_norm.item()
            )

            attention_grad_norm_sum += (
                current_attention_grad
            )

            score_grad_norm_sum += (
                current_score_grad
            )

            for step_idx, step_loss in enumerate(
                step_losses
            ):
                train_step_loss_sum[
                    step_idx
                ] += (
                    float(step_loss.item())
                    * current_batch_size
                )

            if (
                args.max_train_batches
                is not None
                and batch_idx
                >= args.max_train_batches
            ):
                break

        if train_batch_count == 0:
            raise RuntimeError(
                "No valid training batch completed."
            )

        train_loss = (
            train_loss_sum
            / train_count
        )

        train_step_losses = (
            train_step_loss_sum
            / train_count
        )

        average_total_grad = (
            total_grad_norm_sum
            / train_batch_count
        )

        average_attention_grad = (
            attention_grad_norm_sum
            / train_batch_count
        )

        average_score_grad = (
            score_grad_norm_sum
            / train_batch_count
        )

        model.eval()

        val_loss_sum = 0.0
        val_count = 0
        val_batch_count = 0

        val_step_loss_sum = torch.zeros(
            4,
            dtype=torch.float64,
        )

        with torch.no_grad():
            for batch_idx, (
                batch_context,
                batch_y_seq,
            ) in enumerate(
                val_loader,
                start=1,
            ):
                batch_context = (
                    batch_context.to(device)
                )
                batch_y_seq = (
                    batch_y_seq.to(device)
                )

                loss, step_losses = (
                    autoregressive_multistep_loss(
                        model=model,
                        context_norm=(
                            batch_context
                        ),
                        y_seq_norm=(
                            batch_y_seq
                        ),
                        criterion=criterion,
                        rollout_weights=(
                            rollout_weights
                        ),
                    )
                )

                if not torch.isfinite(loss):
                    continue

                current_batch_size = (
                    batch_context.shape[0]
                )

                val_loss_sum += (
                    float(loss.item())
                    * current_batch_size
                )

                val_count += (
                    current_batch_size
                )
                val_batch_count += 1

                for step_idx, step_loss in enumerate(
                    step_losses
                ):
                    val_step_loss_sum[
                        step_idx
                    ] += (
                        float(step_loss.item())
                        * current_batch_size
                    )

                if (
                    args.max_val_batches
                    is not None
                    and batch_idx
                    >= args.max_val_batches
                ):
                    break

        if val_batch_count == 0:
            raise RuntimeError(
                "No valid validation batch completed."
            )

        val_loss = (
            val_loss_sum
            / val_count
        )

        val_step_losses = (
            val_step_loss_sum
            / val_count
        )

        attention_summary = (
            summarize_attention(
                model,
                probe_context,
            )
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        score_grad_label = (
            "LogitGrad"
            if variant == "static"
            else "QKGrad"
        )

        step_message = " | ".join(
            (
                f"Val t+{idx + 1}: "
                f"{val_step_losses[idx]:.6f}"
            )
            for idx in range(4)
        )

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"{step_message} | "
            f"Grad: {average_total_grad:.4f} | "
            f"AttnGrad: "
            f"{average_attention_grad:.3e} | "
            f"{score_grad_label}: "
            f"{average_score_grad:.3e} | "
            f"Gate: "
            f"{attention_summary['gate_mean']:.5f} | "
            f"Entropy: "
            f"{attention_summary['attention_entropy']:.5f} | "
            f"Amax: "
            f"{attention_summary['attention_max_mean']:.5f} | "
            f"Astd: "
            f"{attention_summary['attention_sample_std']:.3e} | "
            f"LR: {current_lr:.2e}"
        )

        if average_score_grad == 0.0:
            print(
                "⚠️ Score-producing parameters "
                "received zero gradient."
            )

        current_best = min(
            best_val_loss,
            val_loss,
        )

        log_data = {
            "epoch": epoch,
            "Train MultiStep Rel-L2": (
                train_loss
            ),
            "Val MultiStep Rel-L2": (
                val_loss
            ),
            "Total Grad Norm": (
                average_total_grad
            ),
            "Attention Grad Norm": (
                average_attention_grad
            ),
            (
                "Static Logit Grad Norm"
                if variant == "static"
                else "QK Grad Norm"
            ): average_score_grad,
            "Learning Rate": current_lr,
            "Best Val MultiStep Rel-L2": (
                current_best
            ),
            "Attention gate mean": (
                attention_summary[
                    "gate_mean"
                ]
            ),
            "Attention gate min": (
                attention_summary[
                    "gate_min"
                ]
            ),
            "Attention gate max": (
                attention_summary[
                    "gate_max"
                ]
            ),
            "Attention score std": (
                attention_summary[
                    "score_std"
                ]
            ),
            "Attention sample std": (
                attention_summary[
                    "attention_sample_std"
                ]
            ),
            "Attention normalized entropy": (
                attention_summary[
                    "attention_entropy"
                ]
            ),
            "Attention max mean": (
                attention_summary[
                    "attention_max_mean"
                ]
            ),
            "Attention branch relative RMS": (
                attention_summary[
                    "branch_relative_rms"
                ]
            ),
        }

        for step_idx in range(4):
            log_data[
                f"Train step t+{step_idx + 1} Rel-L2"
            ] = float(
                train_step_losses[step_idx]
            )

            log_data[
                f"Val step t+{step_idx + 1} Rel-L2"
            ] = float(
                val_step_losses[step_idx]
            )

        mean_matrix = attention_summary[
            "mean_attention"
        ]

        for target_idx, target_name in enumerate(
            FIELD_NAMES
        ):
            for source_idx, source_name in enumerate(
                FIELD_NAMES
            ):
                log_data[
                    (
                        "Attention/"
                        f"{target_name}_from_"
                        f"{source_name}"
                    )
                ] = mean_matrix[
                    target_idx
                ][source_idx]

        wandb.log(log_data)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "val_loss": val_loss,
            "best_val_loss": (
                current_best
            ),
            "experiment": (
                metadata["experiment"]
            ),
            "experiment_type": (
                metadata["experiment_type"]
            ),
            "variant": variant,
            "base_model": (
                "M6-FieldWiseEncoder-H4"
            ),
            "prediction_type": (
                "multi_step_delta_prediction"
            ),
            "normalization": (
                "field-wise mean/std"
            ),
            "task": (
                "autoregressive multi-step "
                "delta prediction"
            ),
            "delta_definition": (
                "pred_next_norm = "
                "current_state_norm + "
                "pred_delta_norm"
            ),
            "loss_function": (
                "Weighted H4 "
                "FieldWiseRelativeL2Loss "
                "on normalized state"
            ),
            "parameter_token": False,
            "parameter_conditioning": False,
            "physics_prior": False,
            "pde_loss": False,
            "rollout_steps": 4,
            "rollout_weights": (
                rollout_weights
                .detach()
                .cpu()
                .tolist()
            ),
            "split_path": str(
                split_path
            ),
            "stats_path": str(
                stats_path
            ),
            "run_name": args.run_name,
            "init_ckpt": str(
                init_ckpt_path
            ),
            "frozen_attention_config": (
                frozen_config
            ),
            "attention_summary": (
                attention_summary
            ),
            "training_protocol": {
                "epochs": args.epochs,
                "batch_size": (
                    args.batch_size
                ),
                "learning_rate": args.lr,
                "weight_decay": (
                    args.weight_decay
                ),
                "scheduler": (
                    "CosineAnnealingLR"
                ),
                "eta_min": args.eta_min,
                "grad_clip": 1.0,
                "seed": args.seed,
                "max_train_batches": (
                    args.max_train_batches
                ),
                "max_val_batches": (
                    args.max_val_batches
                ),
            },
        }

        if (
            args.save_every > 0
            and epoch % args.save_every == 0
        ):
            epoch_path = (
                ckpt_dir
                / (
                    f"{args.run_name}_"
                    f"epoch_{epoch:02d}.pth"
                )
            )

            torch.save(
                checkpoint_payload,
                epoch_path,
            )

            print(
                "   [*] Saved periodic checkpoint:",
                epoch_path,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint_payload[
                "best_val_loss"
            ] = best_val_loss

            torch.save(
                checkpoint_payload,
                best_save_path,
            )

            print(
                "  🌟 [New Best] Saved:",
                best_save_path,
            )

        scheduler.step()

    total_minutes = (
        time.time() - start_time
    ) / 60.0

    final_attention = summarize_attention(
        model,
        probe_context,
    )

    print(
        "\n✅ Training process completed."
    )
    print(
        "Best checkpoint:",
        best_save_path,
    )
    print(
        "Best validation loss:",
        f"{best_val_loss:.8f}",
    )
    print(
        "Elapsed minutes:",
        f"{total_minutes:.2f}",
    )
    print(
        "Final mean attention matrix:"
    )

    print_attention_matrix(
        final_attention[
            "mean_attention"
        ]
    )

    if lightweight_trial:
        print(
            "⚠️ This was only a lightweight "
            "training-script trial."
        )
        print(
            "⚠️ It is not a formal M9 result."
        )

    wandb.finish()
