"""
Audit the complete M8-C training path before creating the full training run.

Checks:
1. Forward shape.
2. C0/C1 exact initial alignment.
3. M6 checkpoint loading compatibility, when checkpoint exists.
4. Gradient reaches shared structured branches.
5. Gradient reaches base strength logits.
6. Gradient reaches C1 conditioned heads.
7. C0 does not use conditioned heads.
8. ParameterToken remains active in both modes.

No optimizer step is performed.
No file is modified.
No checkpoint is saved.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.operators.fno2d_m8_c_structured import (
    M8CStructuredFNO2d,
)


M6_CKPT_CANDIDATES = [
    PROJECT_ROOT
    / "checkpoints/cross_param/"
    "m6_fieldwise_encoder_h4_unseen_pr_best.pth",

    PROJECT_ROOT
    / "checkpoints/cross_param/"
    "m6_fieldwise_encoder_h4_unseen_ra_best.pth",
]


def grad_abs_sum(parameter: torch.Tensor | None) -> float:
    if parameter is None:
        return 0.0

    if parameter.grad is None:
        return 0.0

    return float(
        parameter.grad.detach().abs().sum().item()
    )


def module_grad_abs_sum(module: torch.nn.Module) -> float:
    total = 0.0

    for parameter in module.parameters():
        total += grad_abs_sum(parameter)

    return total


def build_model(
    mode: str,
) -> M8CStructuredFNO2d:
    return M8CStructuredFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        context_length=4,
        num_fields=4,
        field_width=8,
        coupling_mode=mode,
        coupling_hidden_channels=8,
        coupling_condition_hidden_dim=32,
        coupling_condition_scale=1.0,
        coupling_init_strength_logit=-4.0,
        token_hidden_dim=64,
        alpha_token=1.0,
    )


def load_m6_if_available(
    model: M8CStructuredFNO2d,
) -> None:
    existing_ckpts = [
        path
        for path in M6_CKPT_CANDIDATES
        if path.exists()
    ]

    if not existing_ckpts:
        print(
            "⚠️ 未发现 M6 checkpoint，"
            "跳过真实 checkpoint 兼容性检查。"
        )
        return

    ckpt_path = existing_ckpts[0]

    print(
        "📌 使用 M6 checkpoint 做兼容性检查："
    )
    print(f"   {ckpt_path}")

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print("   missing_keys:")
    for key in missing_keys:
        print(f"      {key}")

    print("   unexpected_keys:")
    for key in unexpected_keys:
        print(f"      {key}")

    if unexpected_keys:
        raise RuntimeError(
            "M6 -> M8-C 出现 unexpected_keys："
            f"{unexpected_keys}"
        )

    allowed_missing_prefixes = (
        "param_token.",
        "field_coupling.",
    )

    bad_missing = [
        key
        for key in missing_keys
        if not key.startswith(
            allowed_missing_prefixes
        )
    ]

    if bad_missing:
        raise RuntimeError(
            "M6 -> M8-C 除 param_token.* 和 "
            "field_coupling.* 外仍有缺失："
            f"{bad_missing}"
        )

    print(
        "✅ M6 -> M8-C checkpoint 兼容性通过"
    )


def zero_all_grads(
    model: torch.nn.Module,
) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def run_backward_audit(
    model: M8CStructuredFNO2d,
    x: torch.Tensor,
    param: torch.Tensor,
    target: torch.Tensor,
    mode: str,
):
    zero_all_grads(model)

    output = model(
        x,
        param,
        coupling_mode=mode,
    )

    loss = F.mse_loss(
        output,
        target,
    )

    loss.backward()

    coupling = model.field_coupling

    result = {
        "loss": float(loss.detach().item()),

        "base_strength_logits": grad_abs_sum(
            coupling.base_strength_logits
        ),

        "buoyancy_branch": module_grad_abs_sum(
            coupling.buoyancy_branch
        ),

        "advection_buoyancy_branch": (
            module_grad_abs_sum(
                coupling.advection_buoyancy_branch
            )
        ),

        "advection_velocity_branch": (
            module_grad_abs_sum(
                coupling.advection_velocity_branch
            )
        ),

        "pressure_branch": module_grad_abs_sum(
            coupling.pressure_branch
        ),

        "viscous_branch": module_grad_abs_sum(
            coupling.viscous_branch
        ),

        "thermal_diffusion_branch": (
            module_grad_abs_sum(
                coupling.thermal_diffusion_branch
            )
        ),

        "buoyancy_conditioner": module_grad_abs_sum(
            coupling.buoyancy_conditioner
        ),

        "viscous_conditioner": module_grad_abs_sum(
            coupling.viscous_conditioner
        ),

        "thermal_diffusion_conditioner": (
            module_grad_abs_sum(
                coupling.thermal_diffusion_conditioner
            )
        ),

        "param_token": module_grad_abs_sum(
            model.param_token
        ),
    }

    return result


def print_audit(
    title: str,
    result: dict,
) -> None:
    print()
    print(title)

    for key, value in result.items():
        if key == "loss":
            print(
                f"   {key}: {value:.8e}"
            )
        else:
            print(
                f"   grad_abs_sum/{key}: "
                f"{value:.8e}"
            )


def assert_positive(
    result: dict,
    names: tuple[str, ...],
    label: str,
) -> None:
    failed = [
        name
        for name in names
        if not result[name] > 0.0
    ]

    if failed:
        raise RuntimeError(
            f"{label} 以下梯度为零：{failed}"
        )


def main() -> None:
    torch.manual_seed(7)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("========== M8-C Training-Path Audit ==========")
    print(f"device: {device}")
    print(
        "branch:",
        os.popen(
            "git branch --show-current"
        ).read().strip(),
    )
    print(
        "commit:",
        os.popen(
            "git rev-parse HEAD"
        ).read().strip(),
    )

    batch_size = 2
    height = 32
    width = 64

    x = torch.randn(
        batch_size,
        16,
        height,
        width,
        device=device,
    )

    target = torch.randn(
        batch_size,
        4,
        height,
        width,
        device=device,
    )

    param = torch.tensor(
        [
            [6.0, -0.30103],
            [8.0, 0.30103],
        ],
        dtype=torch.float32,
        device=device,
    )

    model = build_model(
        mode="structured_static"
    )

    load_m6_if_available(model)

    model = model.to(device)
    model.train()

    with torch.no_grad():
        y_c0 = model(
            x,
            param,
            coupling_mode="structured_static",
        )

        y_c1_initial = model(
            x,
            param,
            coupling_mode=(
                "structured_parameter_conditioned"
            ),
        )

    alignment_max_diff = (
        y_c0 - y_c1_initial
    ).abs().max().item()

    print()
    print("========== Initial Alignment ==========")
    print(
        "C0-C1 initial max diff:",
        f"{alignment_max_diff:.12e}",
    )

    if alignment_max_diff >= 1e-7:
        raise RuntimeError(
            "C0/C1 初始输出未严格对齐："
            f"{alignment_max_diff}"
        )

    c0_result = run_backward_audit(
        model=model,
        x=x,
        param=param,
        target=target,
        mode="structured_static",
    )

    c1_result = run_backward_audit(
        model=model,
        x=x,
        param=param,
        target=target,
        mode=(
            "structured_parameter_conditioned"
        ),
    )

    print_audit(
        "========== C0 Gradient Audit ==========",
        c0_result,
    )

    print_audit(
        "========== C1 Gradient Audit ==========",
        c1_result,
    )

    shared_names = (
        "base_strength_logits",
        "buoyancy_branch",
        "advection_buoyancy_branch",
        "advection_velocity_branch",
        "pressure_branch",
        "viscous_branch",
        "thermal_diffusion_branch",
        "param_token",
    )

    assert_positive(
        c0_result,
        shared_names,
        "C0 shared path",
    )

    assert_positive(
        c1_result,
        shared_names,
        "C1 shared path",
    )

    conditioned_names = (
        "buoyancy_conditioner",
        "viscous_conditioner",
        "thermal_diffusion_conditioner",
    )

    # In static mode, conditioned heads are not used.
    for name in conditioned_names:
        if c0_result[name] != 0.0:
            raise RuntimeError(
                "C0 不应对条件修正头产生梯度："
                f"{name}={c0_result[name]}"
            )

    # In C1 mode, all three limited conditioned heads must
    # receive gradients, even though their final layers
    # are zero-initialized.
    assert_positive(
        c1_result,
        conditioned_names,
        "C1 conditioned path",
    )

    print()
    print(
        "✅ M8-C operator、M6 warm start、"
        "共享分支梯度和 C1 条件梯度全部通过。"
    )


if __name__ == "__main__":
    main()
