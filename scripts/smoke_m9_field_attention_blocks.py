"""
Comprehensive block-level smoke tests for M9-0 field attention.

This script tests:
- shapes
- off-diagonal masking
- source-axis row normalization
- target/source direction
- scale=0 residual identity
- input-dependent score/attention differences
- paired M7/M9 branch initialization
- gradients
- FP32 numerical stability
- CUDA AMP numerical stability
- state_dict save/reload consistency

This is not a training or evaluation script.
"""

from __future__ import annotations

import io
import math
import os
import sys
from typing import Callable

import torch
import torch.nn as nn

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from models.blocks.field_attention import (
    StaticSoftmaxFieldAttentionBlock,
    StateDependentFieldAttentionBlock,
)
from models.blocks.field_coupling import FieldCouplingBlock


SEED = 20260717
FIELDS = 4
CHANNELS = 8
HEIGHT = 16
WIDTH = 64
BATCH_SIZE = 4
DEFAULT_SCALE = 0.06
EPS = 1e-12


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.square().mean().sqrt()


def grad_norm(parameter: nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0

    return float(
        parameter.grad.detach().norm().cpu().item()
    )


def normalized_attention_entropy(
    attention: torch.Tensor,
) -> torch.Tensor:
    """
    attention: [B, target, source]

    Since self edges are masked, there are F-1 valid sources.
    Maximum entropy is log(F-1).
    """
    terms = torch.where(
        attention > 0,
        attention * torch.log(
            attention.clamp_min(
                torch.finfo(attention.dtype).tiny
            )
        ),
        torch.zeros_like(attention),
    )

    entropy = -terms.sum(dim=-1)

    return entropy / math.log(
        attention.shape[-1] - 1
    )


def check_attention_invariants(
    name: str,
    attention: torch.Tensor,
    atol: float = 1e-6,
) -> None:
    diagonal = attention.diagonal(
        dim1=-2,
        dim2=-1,
    )

    row_sum = attention.sum(dim=-1)
    row_error = (
        row_sum - 1.0
    ).abs().max()

    print(f"\n[{name}: invariants]")
    print(
        "attention shape:       ",
        tuple(attention.shape),
    )
    print(
        "diagonal max abs:      ",
        float(diagonal.abs().max().item()),
    )
    print(
        "row-sum max error:     ",
        float(row_error.item()),
    )
    print(
        "attention min/max:     ",
        float(attention.min().item()),
        float(attention.max().item()),
    )

    assert attention.ndim == 3
    assert attention.shape[-2:] == (
        FIELDS,
        FIELDS,
    )
    assert torch.isfinite(attention).all()

    assert torch.equal(
        diagonal,
        torch.zeros_like(diagonal),
    )

    assert torch.allclose(
        row_sum,
        torch.ones_like(row_sum),
        atol=atol,
        rtol=atol,
    )

    assert attention.min().item() >= 0.0


def print_dynamic_statistics(
    name: str,
    scores: torch.Tensor,
    attention: torch.Tensor,
) -> None:
    entropy = normalized_attention_entropy(
        attention
    )

    sample_std = attention.std(
        dim=0,
        unbiased=False,
    ).mean()

    offdiag_mask = (
        ~torch.eye(
            FIELDS,
            dtype=torch.bool,
            device=attention.device,
        )
    ).view(1, FIELDS, FIELDS)

    offdiag_values = attention[
        offdiag_mask.expand_as(attention)
    ]

    print(f"\n[{name}: initialization statistics]")
    print(
        "scores mean/std:       ",
        float(scores.mean().item()),
        float(scores.std().item()),
    )
    print(
        "scores max abs:        ",
        float(scores.abs().max().item()),
    )
    print(
        "attention sample std:  ",
        float(sample_std.item()),
    )
    print(
        "attention offdiag max: ",
        float(offdiag_values.max().item()),
    )
    print(
        "normalized entropy:    ",
        float(entropy.mean().item()),
    )


def direction_test(
    device: torch.device,
) -> None:
    """
    Verify:
        attention[target, source]

    Target 0 receives source 3.
    Target 1 receives source 2.
    Target 2 receives source 1.
    Target 3 receives source 0.
    """
    h = torch.tensor(
        [1.0, 2.0, 4.0, 8.0],
        device=device,
    ).view(1, 4, 1, 1, 1)

    attention = torch.zeros(
        1,
        4,
        4,
        device=device,
    )

    attention[0, 0, 3] = 1.0
    attention[0, 1, 2] = 1.0
    attention[0, 2, 1] = 1.0
    attention[0, 3, 0] = 1.0

    mixed = (
        StaticSoftmaxFieldAttentionBlock
        .mix_values(h, attention)
    )

    expected = torch.tensor(
        [8.0, 4.0, 2.0, 1.0],
        device=device,
    ).view(1, 4, 1, 1, 1)

    max_diff = (
        mixed - expected
    ).abs().max()

    print("\n[target/source direction test]")
    print(
        "mixed values:          ",
        mixed.flatten().detach().cpu().tolist(),
    )
    print(
        "expected values:       ",
        expected.flatten().detach().cpu().tolist(),
    )
    print(
        "max abs diff:          ",
        float(max_diff.item()),
    )

    assert torch.equal(mixed, expected)


def scale_zero_identity_test(
    name: str,
    block: nn.Module,
    x: torch.Tensor,
) -> None:
    old_scale = float(
        block.attention_output_scale.item()
    )

    block.set_attention_output_scale(0.0)

    with torch.no_grad():
        out = block(x)

    max_diff = (
        out - x
    ).abs().max()

    block.set_attention_output_scale(
        old_scale
    )

    print(f"\n[{name}: scale=0 identity]")
    print(
        "max abs diff:          ",
        float(max_diff.item()),
    )

    assert torch.equal(out, x)


def input_dependence_test(
    name: str,
    block: StateDependentFieldAttentionBlock,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
) -> None:
    block.eval()

    with torch.no_grad():
        info_a = block(
            x_a,
            return_diagnostics=True,
        )
        info_b = block(
            x_b,
            return_diagnostics=True,
        )

    score_diff = (
        info_a["scores"] - info_b["scores"]
    ).abs()

    attention_diff = (
        info_a["attention"]
        - info_b["attention"]
    ).abs()

    print(f"\n[{name}: input dependence]")
    print(
        "score diff mean/max:   ",
        float(score_diff.mean().item()),
        float(score_diff.max().item()),
    )
    print(
        "attn diff mean/max:    ",
        float(attention_diff.mean().item()),
        float(attention_diff.max().item()),
    )

    assert score_diff.max().item() > 1e-12
    assert attention_diff.max().item() > 1e-12


def m7_branch_diagnostics(
    block: FieldCouplingBlock,
    x: torch.Tensor,
) -> dict[str, torch.Tensor]:
    bsz, fields, channels, height, width = x.shape

    h = x.reshape(
        bsz * fields,
        channels,
        height,
        width,
    )
    h = block.norm(h)
    h = block.pre_proj(h)
    h = block.act(h)
    h = h.reshape(
        bsz,
        fields,
        block.hidden_channels,
        height,
        width,
    )

    weight = (
        block.effective_coupling_matrix()
        .to(dtype=h.dtype, device=h.device)
    )

    mixed = torch.einsum(
        "ij,bjchw->bichw",
        weight,
        h,
    )

    projected = mixed.reshape(
        bsz * fields,
        block.hidden_channels,
        height,
        width,
    )
    projected = block.post_proj(projected)
    projected = block.dropout(projected)
    projected = projected.reshape(
        bsz,
        fields,
        channels,
        height,
        width,
    )

    gate = block.gate_values().to(
        dtype=x.dtype,
        device=x.device,
    ).view(
        1,
        fields,
        1,
        1,
        1,
    )

    branch_update = gate * projected

    return {
        "h": h,
        "mixed": mixed,
        "projected_update": projected,
        "branch_update": branch_update,
        "gate": gate,
    }


def paired_branch_test(
    m7: FieldCouplingBlock,
    static: StaticSoftmaxFieldAttentionBlock,
    dynamic_mean: StateDependentFieldAttentionBlock,
    dynamic_sqrt: StateDependentFieldAttentionBlock,
    x: torch.Tensor,
) -> None:
    """
    All blocks share the exact same:
    - norm
    - pre_proj
    - post_proj
    - residual gate

    The result is still only a synthetic initialization diagnostic.
    It is not the final training-data lambda calibration.
    """
    static.copy_shared_branch_from(m7)
    dynamic_mean.copy_shared_branch_from(m7)
    dynamic_sqrt.copy_shared_branch_from(m7)

    m7.eval()
    static.eval()
    dynamic_mean.eval()
    dynamic_sqrt.eval()

    with torch.no_grad():
        m7_info = m7_branch_diagnostics(
            m7,
            x,
        )
        static_info = static(
            x,
            return_diagnostics=True,
        )
        mean_info = dynamic_mean(
            x,
            return_diagnostics=True,
        )
        sqrt_info = dynamic_sqrt(
            x,
            return_diagnostics=True,
        )

    m7_rms = rms(
        m7_info["branch_update"]
    )

    static_rms = rms(
        static_info["branch_update"]
    )
    mean_rms = rms(
        mean_info["branch_update"]
    )
    sqrt_rms = rms(
        sqrt_info["branch_update"]
    )

    current_scale = float(
        static.attention_output_scale.item()
    )

    static_scale1_rms = (
        static_rms / current_scale
    )

    lambda_candidate = (
        m7_rms / (static_scale1_rms + EPS)
    )

    predicted_mean_ratio = (
        mean_rms
        / current_scale
        * lambda_candidate
        / (m7_rms + EPS)
    )

    predicted_sqrt_ratio = (
        sqrt_rms
        / current_scale
        * lambda_candidate
        / (m7_rms + EPS)
    )

    print("\n[paired M7/M9 branch diagnostic]")
    print(
        "M7 branch RMS:         ",
        float(m7_rms.item()),
    )
    print(
        "Static RMS @0.06:      ",
        float(static_rms.item()),
    )
    print(
        "Dynamic mean RMS:      ",
        float(mean_rms.item()),
    )
    print(
        "Dynamic sqrt RMS:      ",
        float(sqrt_rms.item()),
    )
    print(
        "Synthetic λ candidate: ",
        float(lambda_candidate.item()),
    )
    print(
        "Dynamic mean/M7 ratio "
        "with same λ:           ",
        float(predicted_mean_ratio.item()),
    )
    print(
        "Dynamic sqrt/M7 ratio "
        "with same λ:           ",
        float(predicted_sqrt_ratio.item()),
    )
    print(
        "⚠️ This λ is diagnostic only; "
        "formal calibration must use fixed training batches."
    )


def gradient_test(
    name: str,
    block: nn.Module,
    x: torch.Tensor,
) -> None:
    block.train()
    block.zero_grad(set_to_none=True)

    info = block(
        x,
        return_diagnostics=True,
    )

    torch.manual_seed(SEED + 99)
    probe = torch.randn_like(info["out"])

    loss = (
        info["out"] * probe
    ).mean()

    loss.backward()

    post_grad = grad_norm(
        block.post_proj.weight
    )
    gate_grad = grad_norm(
        block.residual_gate
    )

    print(f"\n[{name}: gradient diagnostic]")
    print(
        "loss:                  ",
        float(loss.detach().item()),
    )
    print(
        "post_proj grad norm:   ",
        post_grad,
    )
    print(
        "gate grad norm:        ",
        gate_grad,
    )

    assert math.isfinite(post_grad)
    assert math.isfinite(gate_grad)
    assert post_grad > 0.0
    assert gate_grad > 0.0

    if isinstance(
        block,
        StaticSoftmaxFieldAttentionBlock,
    ):
        logits_grad = (
            block.static_attention_logits.grad
        )

        assert logits_grad is not None
        assert torch.isfinite(logits_grad).all()

        diagonal_grad = logits_grad.diagonal()
        offdiag_mask = ~torch.eye(
            FIELDS,
            dtype=torch.bool,
            device=logits_grad.device,
        )
        offdiag_grad = logits_grad[
            offdiag_mask
        ]

        print(
            "logits offdiag grad:  ",
            float(
                offdiag_grad.norm().item()
            ),
        )
        print(
            "logits diag grad max: ",
            float(
                diagonal_grad.abs().max().item()
            ),
        )

        assert offdiag_grad.norm().item() > 0.0

        assert torch.equal(
            diagonal_grad,
            torch.zeros_like(diagonal_grad),
        )

    if isinstance(
        block,
        StateDependentFieldAttentionBlock,
    ):
        q_grad = grad_norm(
            block.q_proj.weight
        )
        k_grad = grad_norm(
            block.k_proj.weight
        )

        print(
            "q_proj grad norm:     ",
            q_grad,
        )
        print(
            "k_proj grad norm:     ",
            k_grad,
        )

        assert math.isfinite(q_grad)
        assert math.isfinite(k_grad)
        assert q_grad > 0.0
        assert k_grad > 0.0


def state_dict_reload_test(
    name: str,
    block: nn.Module,
    factory: Callable[[], nn.Module],
    x: torch.Tensor,
    device: torch.device,
) -> None:
    block.eval()

    with torch.no_grad():
        y_before = block(x)

    buffer = io.BytesIO()
    torch.save(
        block.state_dict(),
        buffer,
    )
    buffer.seek(0)

    restored = factory().to(device)

    restored.load_state_dict(
        torch.load(
            buffer,
            map_location=device,
            weights_only=True,
        )
    )
    restored.eval()

    with torch.no_grad():
        y_after = restored(x)

    max_diff = (
        y_before - y_after
    ).abs().max()

    print(f"\n[{name}: state_dict reload]")
    print(
        "max abs diff:          ",
        float(max_diff.item()),
    )

    assert torch.equal(
        y_before,
        y_after,
    )


def amp_test(
    block: nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> None:
    if device.type != "cuda":
        print(
            "\n[AMP test]\n"
            "Skipped: CUDA is not available."
        )
        return

    block.eval()

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            info = block(
                x,
                return_diagnostics=True,
            )

    attention = info["attention"]

    print("\n[CUDA AMP test]")
    print(
        "attention dtype:       ",
        attention.dtype,
    )
    print(
        "attention finite:      ",
        bool(
            torch.isfinite(
                attention
            ).all().item()
        ),
    )
    print(
        "output finite:         ",
        bool(
            torch.isfinite(
                info["out"]
            ).all().item()
        ),
    )

    check_attention_invariants(
        "CUDA AMP",
        attention,
        atol=5e-4,
    )

    assert torch.isfinite(
        attention
    ).all()
    assert torch.isfinite(
        info["out"]
    ).all()


def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("========== M9 BLOCK TEST ==========")
    print("device:", device)
    print("torch:", torch.__version__)
    print("seed:", SEED)

    x = torch.randn(
        BATCH_SIZE,
        FIELDS,
        CHANNELS,
        HEIGHT,
        WIDTH,
        device=device,
    )

    x_other = torch.randn(
        BATCH_SIZE,
        FIELDS,
        CHANNELS,
        HEIGHT,
        WIDTH,
        device=device,
    )

    m7 = FieldCouplingBlock(
        channels=CHANNELS,
        num_fields=FIELDS,
        hidden_channels=CHANNELS,
        dropout=0.0,
        init_gate=-4.0,
        use_norm=True,
    ).to(device)

    static = StaticSoftmaxFieldAttentionBlock(
        channels=CHANNELS,
        num_fields=FIELDS,
        hidden_channels=CHANNELS,
        dropout=0.0,
        init_gate=-4.0,
        use_norm=True,
        attention_output_scale=DEFAULT_SCALE,
    ).to(device)

    dynamic_mean = StateDependentFieldAttentionBlock(
        channels=CHANNELS,
        num_fields=FIELDS,
        hidden_channels=CHANNELS,
        qk_channels=CHANNELS,
        dropout=0.0,
        init_gate=-4.0,
        use_norm=True,
        attention_output_scale=DEFAULT_SCALE,
        score_scale_mode="spatial_mean",
        qk_init_std=0.02,
    ).to(device)

    dynamic_sqrt = StateDependentFieldAttentionBlock(
        channels=CHANNELS,
        num_fields=FIELDS,
        hidden_channels=CHANNELS,
        qk_channels=CHANNELS,
        dropout=0.0,
        init_gate=-4.0,
        use_norm=True,
        attention_output_scale=DEFAULT_SCALE,
        score_scale_mode="sqrt_token_dim",
        qk_init_std=0.02,
    ).to(device)

    # Pair shared branch parameters before comparison.
    static.copy_shared_branch_from(m7)
    dynamic_mean.copy_shared_branch_from(m7)
    dynamic_sqrt.copy_shared_branch_from(m7)

    for name, block in (
        ("static", static),
        ("dynamic_spatial_mean", dynamic_mean),
        ("dynamic_sqrt_token_dim", dynamic_sqrt),
    ):
        block.eval()

        with torch.no_grad():
            info = block(
                x,
                return_diagnostics=True,
            )

        assert info["out"].shape == x.shape
        assert torch.isfinite(
            info["out"]
        ).all()

        check_attention_invariants(
            name,
            info["attention"],
        )

        print_dynamic_statistics(
            name,
            info["scores"],
            info["attention"],
        )

        scale_zero_identity_test(
            name,
            block,
            x,
        )

    direction_test(device)

    input_dependence_test(
        "dynamic_spatial_mean",
        dynamic_mean,
        x,
        x_other,
    )

    input_dependence_test(
        "dynamic_sqrt_token_dim",
        dynamic_sqrt,
        x,
        x_other,
    )

    paired_branch_test(
        m7=m7,
        static=static,
        dynamic_mean=dynamic_mean,
        dynamic_sqrt=dynamic_sqrt,
        x=x,
    )

    gradient_test(
        "static",
        static,
        x,
    )

    gradient_test(
        "dynamic_spatial_mean",
        dynamic_mean,
        x,
    )

    gradient_test(
        "dynamic_sqrt_token_dim",
        dynamic_sqrt,
        x,
    )

    state_dict_reload_test(
        name="static",
        block=static,
        factory=lambda: (
            StaticSoftmaxFieldAttentionBlock(
                channels=CHANNELS,
                num_fields=FIELDS,
                hidden_channels=CHANNELS,
                dropout=0.0,
                init_gate=-4.0,
                use_norm=True,
                attention_output_scale=DEFAULT_SCALE,
            )
        ),
        x=x,
        device=device,
    )

    state_dict_reload_test(
        name="dynamic_spatial_mean",
        block=dynamic_mean,
        factory=lambda: (
            StateDependentFieldAttentionBlock(
                channels=CHANNELS,
                num_fields=FIELDS,
                hidden_channels=CHANNELS,
                qk_channels=CHANNELS,
                dropout=0.0,
                init_gate=-4.0,
                use_norm=True,
                attention_output_scale=DEFAULT_SCALE,
                score_scale_mode="spatial_mean",
                qk_init_std=0.02,
            )
        ),
        x=x,
        device=device,
    )

    state_dict_reload_test(
        name="dynamic_sqrt_token_dim",
        block=dynamic_sqrt,
        factory=lambda: (
            StateDependentFieldAttentionBlock(
                channels=CHANNELS,
                num_fields=FIELDS,
                hidden_channels=CHANNELS,
                qk_channels=CHANNELS,
                dropout=0.0,
                init_gate=-4.0,
                use_norm=True,
                attention_output_scale=DEFAULT_SCALE,
                score_scale_mode="sqrt_token_dim",
                qk_init_std=0.02,
            )
        ),
        x=x,
        device=device,
    )

    amp_test(
        dynamic_sqrt,
        x,
        device,
    )

    print(
        "\n✅ Comprehensive M9 block tests passed."
    )
    print(
        "⚠️ No score scaling or lambda has been "
        "frozen by this test."
    )


if __name__ == "__main__":
    main()
