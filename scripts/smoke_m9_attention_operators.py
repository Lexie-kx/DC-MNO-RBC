"""
M9-0a/0b operator-level smoke test.

Checks:
- frozen configuration
- M6 checkpoint strict loading
- missing/unexpected key rules
- every M6 parameter copied exactly
- paired Static/Dynamic shared branch initialization
- output and Attention shapes
- scale=0 strict equivalence to M6
- real initialization perturbation
- per-field initialization perturbation
- Attention invariants
- parameter counts

This is not a training script.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from models.operators.fno2d_fieldwise import (
    FieldWiseFNO2d,
)
from models.operators.fno2d_m9_static_attention import (
    M9StaticAttentionFNO2d,
)
from models.operators.fno2d_m9_state_attention import (
    M9StateAttentionFNO2d,
)


SEED = 20260717
EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test M9 attention operators "
            "against one M6 checkpoint."
        )
    )

    parser.add_argument(
        "--m6_ckpt",
        required=True,
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/m9_0_attention_frozen.json"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=64,
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(
    root: Path,
    value: str,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = root / path

    return path.resolve()


def extract_state_dict(checkpoint):
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint["model_state_dict"]

    return checkpoint


def parameter_count(model) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def trainable_parameter_count(model) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def new_parameter_count(model) -> int:
    return sum(
        parameter.numel()
        for name, parameter
        in model.named_parameters()
        if name.startswith("field_attention.")
    )


def load_m6_into_m9(
    model,
    m6_state,
    model_name: str,
) -> None:
    missing_keys, unexpected_keys = (
        model.load_state_dict(
            m6_state,
            strict=False,
        )
    )

    print(f"\n[{model_name}: checkpoint loading]")
    print("missing keys:", missing_keys)
    print("unexpected keys:", unexpected_keys)

    if unexpected_keys:
        raise RuntimeError(
            f"{model_name}: unexpected keys: "
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
            f"{model_name}: only field_attention.* "
            f"may be missing. Bad keys: {bad_missing}"
        )

    model_state = model.state_dict()

    mismatched = []

    for key, value in m6_state.items():
        if key not in model_state:
            mismatched.append(
                f"missing old key: {key}"
            )
            continue

        if not torch.equal(
            value.detach().cpu(),
            model_state[key].detach().cpu(),
        ):
            mismatched.append(
                f"value mismatch: {key}"
            )

    if mismatched:
        raise RuntimeError(
            f"{model_name}: M6 parameter mismatch:\n"
            + "\n".join(mismatched[:20])
        )

    print(
        f"✅ {model_name}: all M6 tensors "
        "loaded exactly."
    )


def compare_shared_attention_branch(
    static_model,
    dynamic_model,
) -> None:
    names = [
        "norm",
        "pre_proj",
        "post_proj",
        "residual_gate",
    ]

    mismatches = []

    for name in names:
        if name == "residual_gate":
            static_state = {
                name: (
                    static_model
                    .field_attention
                    .residual_gate
                    .detach()
                )
            }
            dynamic_state = {
                name: (
                    dynamic_model
                    .field_attention
                    .residual_gate
                    .detach()
                )
            }
        else:
            static_state = getattr(
                static_model.field_attention,
                name,
            ).state_dict()

            dynamic_state = getattr(
                dynamic_model.field_attention,
                name,
            ).state_dict()

        for key in static_state:
            if not torch.equal(
                static_state[key].detach().cpu(),
                dynamic_state[key].detach().cpu(),
            ):
                mismatches.append(
                    f"{name}.{key}"
                )

    if mismatches:
        raise RuntimeError(
            "Static/Dynamic shared attention "
            f"initialization differs: {mismatches}"
        )

    print(
        "✅ Static/Dynamic attention shared branch "
        "uses paired initialization."
    )


def attention_checks(
    model_name: str,
    attention: torch.Tensor,
) -> None:
    diagonal = attention.diagonal(
        dim1=-2,
        dim2=-1,
    )

    row_error = (
        attention.sum(dim=-1) - 1.0
    ).abs().max()

    print(f"\n[{model_name}: attention]")
    print("shape:", tuple(attention.shape))
    print(
        "diagonal max:",
        float(diagonal.abs().max().item()),
    )
    print(
        "row-sum max error:",
        float(row_error.item()),
    )
    print(
        "sample std:",
        float(
            attention.std(
                dim=0,
                unbiased=False,
            ).mean().item()
        ),
    )

    assert torch.equal(
        diagonal,
        torch.zeros_like(diagonal),
    )

    assert torch.allclose(
        attention.sum(dim=-1),
        torch.ones_like(
            attention.sum(dim=-1)
        ),
        atol=1e-6,
        rtol=1e-6,
    )

    assert torch.isfinite(attention).all()


def relative_difference(
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    return float(
        (
            prediction - reference
        ).norm().item()
        /
        (
            reference.norm().item()
            + EPS
        )
    )


def per_field_relative_difference(
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> list[float]:
    values = []

    for field_idx in range(
        prediction.shape[1]
    ):
        pred_field = prediction[
            :,
            field_idx,
        ]
        ref_field = reference[
            :,
            field_idx,
        ]

        values.append(
            float(
                (
                    pred_field - ref_field
                ).norm().item()
                /
                (
                    ref_field.norm().item()
                    + EPS
                )
            )
        )

    return values


def main():
    args = parse_args()

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    checkpoint_path = resolve_path(
        project_root,
        args.m6_ckpt,
    )

    config_path = resolve_path(
        project_root,
        args.config,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    if not config_path.exists():
        raise FileNotFoundError(
            config_path
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    assert (
        config["score_scale_mode"]
        == "sqrt_token_dim"
    )
    assert (
        config["score_temperature"]
        == 0.03
    )
    assert (
        config["attention_output_scale"]
        == 0.05675224
    )
    assert (
        config[
            "shared_scale_for_static_and_dynamic"
        ]
        is True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "========== M9 OPERATOR SMOKE =========="
    )
    print("device:", device)
    print("checkpoint:", checkpoint_path)
    print("config:", config_path)
    print(
        "temperature:",
        config["score_temperature"],
    )
    print(
        "score scaling:",
        config["score_scale_mode"],
    )
    print(
        "shared lambda:",
        config["attention_output_scale"],
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
        checkpoint_path,
        map_location=device,
    )

    m6_state = extract_state_dict(
        checkpoint
    )

    m6.load_state_dict(
        m6_state,
        strict=True,
    )

    print(
        "✅ M6 checkpoint loaded with strict=True"
    )

    # Reset the same seed before constructing each
    # Attention model, so their shared new branch
    # receives paired initialization.
    set_seed(SEED)

    static_model = M9StaticAttentionFNO2d(
        attention_output_scale=(
            config["attention_output_scale"]
        ),
        score_temperature=(
            config["score_temperature"]
        ),
    ).to(device)

    set_seed(SEED)

    dynamic_model = M9StateAttentionFNO2d(
        attention_output_scale=(
            config["attention_output_scale"]
        ),
        score_scale_mode=(
            config["score_scale_mode"]
        ),
        score_temperature=(
            config["score_temperature"]
        ),
        qk_init_std=config["qk_init_std"],
    ).to(device)

    load_m6_into_m9(
        static_model,
        m6_state,
        "M9-0a Static",
    )

    load_m6_into_m9(
        dynamic_model,
        m6_state,
        "M9-0b Dynamic",
    )

    compare_shared_attention_branch(
        static_model,
        dynamic_model,
    )

    print("\n========== PARAMETER COUNTS ==========")

    for name, model in (
        ("M6", m6),
        ("M9-0a Static", static_model),
        ("M9-0b Dynamic", dynamic_model),
    ):
        print(
            f"{name:16s} | "
            f"total={parameter_count(model):,} | "
            f"trainable="
            f"{trainable_parameter_count(model):,} | "
            f"new_attention="
            f"{new_parameter_count(model):,}"
        )

    set_seed(SEED + 1)

    x = torch.randn(
        args.batch_size,
        16,
        args.height,
        args.width,
        device=device,
        dtype=torch.float32,
    )

    m6.eval()
    static_model.eval()
    dynamic_model.eval()

    with torch.no_grad():
        y_m6 = m6(x)

    static_scale = float(
        static_model
        .field_attention
        .attention_output_scale
        .item()
    )

    dynamic_scale = float(
        dynamic_model
        .field_attention
        .attention_output_scale
        .item()
    )

    static_model.field_attention.set_attention_output_scale(
        0.0
    )
    dynamic_model.field_attention.set_attention_output_scale(
        0.0
    )

    with torch.no_grad():
        y_static_zero = static_model(x)
        y_dynamic_zero = dynamic_model(x)

    static_zero_diff = (
        y_static_zero - y_m6
    ).abs().max()

    dynamic_zero_diff = (
        y_dynamic_zero - y_m6
    ).abs().max()

    print("\n========== SCALE=0 EQUIVALENCE ==========")
    print(
        "Static max abs diff:",
        float(static_zero_diff.item()),
    )
    print(
        "Dynamic max abs diff:",
        float(dynamic_zero_diff.item()),
    )

    assert torch.equal(
        y_static_zero,
        y_m6,
    )

    assert torch.equal(
        y_dynamic_zero,
        y_m6,
    )

    static_model.field_attention.set_attention_output_scale(
        static_scale
    )
    dynamic_model.field_attention.set_attention_output_scale(
        dynamic_scale
    )

    with torch.no_grad():
        static_info = static_model(
            x,
            return_features=True,
        )

        dynamic_info = dynamic_model(
            x,
            return_features=True,
        )

    y_static = static_info["out"]
    y_dynamic = dynamic_info["out"]

    attention_checks(
        "M9-0a Static",
        static_info[
            "field_attention"
        ]["attention"],
    )

    attention_checks(
        "M9-0b Dynamic",
        dynamic_info[
            "field_attention"
        ]["attention"],
    )

    field_names = [
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
    ]

    print(
        "\n========== REAL INIT PERTURBATION =========="
    )

    print(
        "Static relative init diff:",
        relative_difference(
            y_static,
            y_m6,
        ),
    )

    print(
        "Dynamic relative init diff:",
        relative_difference(
            y_dynamic,
            y_m6,
        ),
    )

    static_per_field = (
        per_field_relative_difference(
            y_static,
            y_m6,
        )
    )

    dynamic_per_field = (
        per_field_relative_difference(
            y_dynamic,
            y_m6,
        )
    )

    for field_idx, field_name in enumerate(
        field_names
    ):
        print(
            f"{field_name:10s} | "
            f"Static={static_per_field[field_idx]:.6e} | "
            f"Dynamic={dynamic_per_field[field_idx]:.6e}"
        )

    assert y_static.shape == (
        args.batch_size,
        4,
        args.height,
        args.width,
    )

    assert y_dynamic.shape == (
        args.batch_size,
        4,
        args.height,
        args.width,
    )

    assert torch.isfinite(y_static).all()
    assert torch.isfinite(y_dynamic).all()

    print(
        "\n✅ M9 operator smoke passed."
    )
    print(
        "⚠️ This is an initialization/loading test, "
        "not a formal training result."
    )


if __name__ == "__main__":
    main()
