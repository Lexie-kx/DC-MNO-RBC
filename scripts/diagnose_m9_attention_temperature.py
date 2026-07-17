"""
M9-0b score-temperature diagnostic on real M6 field features.

Purpose:
- Use fixed training batches only.
- Load the trained M6-FieldWiseEncoder checkpoint.
- Extract real field features [B,4,8,H,W].
- Compare score temperatures under sqrt_token_dim scaling.
- Keep all block parameters identical across temperatures.
- Do not train the model.
- Do not select settings from validation/test performance.

This script does NOT freeze the final temperature or lambda.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from datasets.rbc_dataset import RBCDataset
from models.blocks.field_attention import (
    StateDependentFieldAttentionBlock,
)
from models.blocks.field_coupling import FieldCouplingBlock
from models.operators.fno2d_fieldwise import FieldWiseFNO2d


EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose M9 state-dependent field-attention "
            "temperature using real M6 field features."
        )
    )

    parser.add_argument("--split", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--m6_ckpt", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--temperatures",
        type=str,
        default="1,0.3,0.1,0.03,0.01,0.003,0.001",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260717,
    )

    parser.add_argument(
        "--attention_output_scale",
        type=float,
        default=0.06,
        help=(
            "Diagnostic candidate only. This script does not "
            "freeze the final shared lambda."
        ),
    )

    parser.add_argument(
        "--qk_init_std",
        type=float,
        default=0.02,
    )

    return parser.parse_args()


def resolve_path(project_root: Path, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_state_dict(ckpt):
    if (
        isinstance(ckpt, dict)
        and "model_state_dict" in ckpt
    ):
        return ckpt["model_state_dict"]
    return ckpt


def normalized_attention_entropy(
    attention: torch.Tensor,
) -> torch.Tensor:
    """
    attention: [N, target, source]

    The diagonal is zero, so each row has F-1 valid sources.
    """
    tiny = torch.finfo(attention.dtype).tiny

    terms = torch.where(
        attention > 0,
        attention
        * torch.log(attention.clamp_min(tiny)),
        torch.zeros_like(attention),
    )

    entropy = -terms.sum(dim=-1)

    return entropy / math.log(
        attention.shape[-1] - 1
    )


def parameter_grad_norm(
    parameter: torch.nn.Parameter,
) -> float:
    if parameter.grad is None:
        return 0.0

    return float(
        parameter.grad.detach().norm().cpu().item()
    )


def collect_real_field_features(
    model,
    loader,
    device,
    num_batches,
):
    batches = []

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            loader,
            start=1,
        ):
            x_norm = batch[0].to(
                device=device,
                dtype=torch.float32,
            )

            _, field_features = model.encode_fields(
                x_norm
            )

            field_stack = torch.stack(
                field_features,
                dim=1,
            )

            batches.append(
                field_stack.detach().cpu()
            )

            print(
                f"   collected batch "
                f"{batch_idx}/{num_batches}: "
                f"{tuple(field_stack.shape)}"
            )

            if batch_idx >= num_batches:
                break

    if not batches:
        raise RuntimeError(
            "No training batches were collected."
        )

    return batches


def make_reference_state(
    seed,
    channels,
    num_fields,
    attention_output_scale,
    qk_init_std,
):
    """
    Create one deterministic M9 state.

    A temporary M7 block supplies paired initialization for:
    - norm
    - pre_proj
    - post_proj
    - residual gate

    Q/K are initialized once and then reused by every temperature.
    """
    set_seed(seed)

    m7_reference = FieldCouplingBlock(
        channels=channels,
        num_fields=num_fields,
        hidden_channels=channels,
        dropout=0.0,
        init_gate=-4.0,
        use_norm=True,
    )

    reference = StateDependentFieldAttentionBlock(
        channels=channels,
        num_fields=num_fields,
        hidden_channels=channels,
        qk_channels=channels,
        dropout=0.0,
        init_gate=-4.0,
        use_norm=True,
        attention_output_scale=attention_output_scale,
        score_scale_mode="sqrt_token_dim",
        score_temperature=1.0,
        qk_init_std=qk_init_std,
    )

    reference.copy_shared_branch_from(
        m7_reference
    )

    return {
        key: value.detach().cpu().clone()
        for key, value
        in reference.state_dict().items()
    }


def build_block(
    reference_state,
    temperature,
    channels,
    num_fields,
    attention_output_scale,
    qk_init_std,
    device,
):
    block = StateDependentFieldAttentionBlock(
        channels=channels,
        num_fields=num_fields,
        hidden_channels=channels,
        qk_channels=channels,
        dropout=0.0,
        init_gate=-4.0,
        use_norm=True,
        attention_output_scale=attention_output_scale,
        score_scale_mode="sqrt_token_dim",
        score_temperature=temperature,
        qk_init_std=qk_init_std,
    ).to(device)

    block.load_state_dict(
        reference_state,
        strict=True,
    )

    return block


def diagnose_temperature(
    temperature,
    reference_state,
    feature_batches,
    channels,
    num_fields,
    attention_output_scale,
    qk_init_std,
    device,
    seed,
):
    block = build_block(
        reference_state=reference_state,
        temperature=temperature,
        channels=channels,
        num_fields=num_fields,
        attention_output_scale=(
            attention_output_scale
        ),
        qk_init_std=qk_init_std,
        device=device,
    )

    block.eval()

    all_scores = []
    all_attention = []
    all_branch_rel = []

    with torch.no_grad():
        for features_cpu in feature_batches:
            features = features_cpu.to(
                device=device,
                dtype=torch.float32,
            )

            info = block(
                features,
                return_diagnostics=True,
            )

            all_scores.append(
                info["scores"].detach().cpu()
            )
            all_attention.append(
                info["attention"].detach().cpu()
            )

            branch_rel = (
                info["branch_update"]
                .flatten(1)
                .norm(dim=1)
                /
                (
                    features
                    .flatten(1)
                    .norm(dim=1)
                    + EPS
                )
            )

            all_branch_rel.append(
                branch_rel.detach().cpu()
            )

    scores = torch.cat(
        all_scores,
        dim=0,
    )

    attention = torch.cat(
        all_attention,
        dim=0,
    )

    branch_rel = torch.cat(
        all_branch_rel,
        dim=0,
    )

    effective_logits = scores / temperature

    entropy = normalized_attention_entropy(
        attention
    )

    sample_std = attention.std(
        dim=0,
        unbiased=False,
    ).mean()

    shifted_diff = (
        attention
        - torch.roll(
            attention,
            shifts=1,
            dims=0,
        )
    ).abs()

    offdiag_mask = (
        ~torch.eye(
            num_fields,
            dtype=torch.bool,
        )
    ).view(
        1,
        num_fields,
        num_fields,
    )

    offdiag_values = attention[
        offdiag_mask.expand_as(attention)
    ]

    uniform_value = 1.0 / (
        num_fields - 1
    )

    uniform_deviation = (
        offdiag_values - uniform_value
    ).abs()

    diagonal = attention.diagonal(
        dim1=-2,
        dim2=-1,
    )

    row_error = (
        attention.sum(dim=-1) - 1.0
    ).abs()

    # Gradient diagnostic on the first real feature batch.
    block.train()
    block.zero_grad(set_to_none=True)

    first_features = feature_batches[0].to(
        device=device,
        dtype=torch.float32,
    )

    info = block(
        first_features,
        return_diagnostics=True,
    )

    set_seed(seed + 999)

    probe = torch.randn_like(
        info["out"]
    )

    loss = (
        info["out"] * probe
    ).mean()

    loss.backward()

    q_grad = parameter_grad_norm(
        block.q_proj.weight
    )
    k_grad = parameter_grad_norm(
        block.k_proj.weight
    )
    post_grad = parameter_grad_norm(
        block.post_proj.weight
    )
    gate_grad = parameter_grad_norm(
        block.residual_gate
    )

    return {
        "temperature": temperature,
        "num_samples": int(
            attention.shape[0]
        ),
        "raw_scores_mean": float(
            scores.mean().item()
        ),
        "raw_scores_std": float(
            scores.std().item()
        ),
        "raw_scores_max_abs": float(
            scores.abs().max().item()
        ),
        "effective_logits_std": float(
            effective_logits.std().item()
        ),
        "effective_logits_max_abs": float(
            effective_logits.abs().max().item()
        ),
        "attention_sample_std": float(
            sample_std.item()
        ),
        "attention_shift_diff_mean": float(
            shifted_diff.mean().item()
        ),
        "attention_shift_diff_max": float(
            shifted_diff.max().item()
        ),
        "attention_offdiag_max": float(
            offdiag_values.max().item()
        ),
        "attention_offdiag_min": float(
            offdiag_values.min().item()
        ),
        "uniform_deviation_mean": float(
            uniform_deviation.mean().item()
        ),
        "normalized_entropy_mean": float(
            entropy.mean().item()
        ),
        "normalized_entropy_min": float(
            entropy.min().item()
        ),
        "branch_relative_norm_mean": float(
            branch_rel.mean().item()
        ),
        "branch_relative_norm_max": float(
            branch_rel.max().item()
        ),
        "diagonal_max_abs": float(
            diagonal.abs().max().item()
        ),
        "row_sum_max_error": float(
            row_error.max().item()
        ),
        "q_grad_norm": q_grad,
        "k_grad_norm": k_grad,
        "post_grad_norm": post_grad,
        "gate_grad_norm": gate_grad,
        "q_to_post_grad_ratio": (
            q_grad / (post_grad + EPS)
        ),
        "k_to_post_grad_ratio": (
            k_grad / (post_grad + EPS)
        ),
    }


def main():
    args = parse_args()

    project_root = Path(
        __file__
    ).resolve().parent.parent

    split_path = resolve_path(
        project_root,
        args.split,
    )
    stats_path = resolve_path(
        project_root,
        args.stats,
    )
    ckpt_path = resolve_path(
        project_root,
        args.m6_ckpt,
    )
    output_path = resolve_path(
        project_root,
        args.output,
    )

    for path in (
        split_path,
        stats_path,
        ckpt_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temperatures = [
        float(value.strip())
        for value
        in args.temperatures.split(",")
        if value.strip()
    ]

    if not temperatures:
        raise ValueError(
            "No temperatures were provided."
        )

    if any(
        temperature <= 0.0
        for temperature in temperatures
    ):
        raise ValueError(
            "Every temperature must be positive."
        )

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "========== M9 TEMPERATURE DIAGNOSTIC =========="
    )
    print("Device:", device)
    print("Split:", split_path)
    print("Stats:", stats_path)
    print("M6 checkpoint:", ckpt_path)
    print("Output:", output_path)
    print("Temperatures:", temperatures)
    print("Fixed train batches:", args.num_batches)
    print("Seed:", args.seed)
    print(
        "Attention scale candidate:",
        args.attention_output_scale,
    )
    print(
        "Score scaling: sqrt_token_dim"
    )
    print(
        "⚠️ This diagnostic uses training data only."
    )
    print(
        "⚠️ It does not freeze temperature or lambda."
    )

    with split_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        split_config = json.load(f)

    train_dataset = RBCDataset(
        split_config=split_config["train"],
        normalize=True,
        stats_path=str(stats_path),
        return_sequence=True,
        target_steps=4,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )

    print(
        "Train samples:",
        len(train_dataset),
    )

    m6 = FieldWiseFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        context_length=4,
        num_fields=4,
    ).to(device)

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    m6.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "✅ M6 checkpoint loaded with strict=True"
    )
    print(
        "📦 Collecting fixed real M6 field features..."
    )

    feature_batches = collect_real_field_features(
        model=m6,
        loader=train_loader,
        device=device,
        num_batches=args.num_batches,
    )

    channels = int(
        feature_batches[0].shape[2]
    )
    num_fields = int(
        feature_batches[0].shape[1]
    )

    reference_state = make_reference_state(
        seed=args.seed + 100,
        channels=channels,
        num_fields=num_fields,
        attention_output_scale=(
            args.attention_output_scale
        ),
        qk_init_std=args.qk_init_std,
    )

    rows = []

    print(
        "\n========== TEMPERATURE SWEEP =========="
    )
    print(
        "temp | logit_std | attn_std | "
        "entropy | attn_max | q_grad | "
        "k_grad | q/post"
    )

    for temperature in temperatures:
        row = diagnose_temperature(
            temperature=temperature,
            reference_state=reference_state,
            feature_batches=feature_batches,
            channels=channels,
            num_fields=num_fields,
            attention_output_scale=(
                args.attention_output_scale
            ),
            qk_init_std=args.qk_init_std,
            device=device,
            seed=args.seed,
        )

        rows.append(row)

        print(
            f"{temperature:>6g} | "
            f"{row['effective_logits_std']:.3e} | "
            f"{row['attention_sample_std']:.3e} | "
            f"{row['normalized_entropy_mean']:.6f} | "
            f"{row['attention_offdiag_max']:.6f} | "
            f"{row['q_grad_norm']:.3e} | "
            f"{row['k_grad_norm']:.3e} | "
            f"{row['q_to_post_grad_ratio']:.3e}"
        )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        "\n✅ Temperature diagnostic complete."
    )
    print(
        "📌 CSV:",
        output_path,
    )
    print(
        "⚠️ Do not start formal M9 training yet."
    )


if __name__ == "__main__":
    main()
