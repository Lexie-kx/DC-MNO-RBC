"""
M8 Module-Level Condition Gate.

Lightweight controlled ablation:

    FieldWiseEncoder
    -> statically coupled field features
    -> FNO backbone with ParameterToken
    -> delta prediction

A single condition-dependent scalar controls the relative contribution of:

1. ParameterToken
2. Static FieldCoupling residual

The internal 12 directed coupling edges remain unchanged.

At zero initialization:

    token_weight = 1
    coupling_weight = 1

Therefore, after loading an M8-A checkpoint, the initial model is exactly
aligned with M8-A.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)
import torch.nn as nn
import torch.nn.functional as F

from models.operators.fno2d_m8_full_conditioned import (
    M8FullConditionedFNO2d,
)
from models.operators.fno2d_paramtoken import apply_param_token


class M8ModuleGateFNO2d(M8FullConditionedFNO2d):
    """
    M8-Gate-O5-H4 lightweight ablation.

    Input:
        x:
            [B, 16, H, W]

        param:
            [B, 2] = [log10(Ra), log10(Pr)]

    Output:
        normalized delta prediction:
            [B, 4, H, W]

    Gate:

        score = tanh(
            Linear(
                normalized [log10(Ra), log10(Pr)]
            )
        )

        token_weight =
            1 + gate_scale * score

        coupling_weight =
            1 - gate_scale * score

    A positive score moves toward stronger ParameterToken and weaker
    FieldCoupling.

    A negative score moves toward stronger FieldCoupling and weaker
    ParameterToken.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 4,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 32,
        context_length: int = 4,
        num_fields: int = 4,
        field_width: int | None = None,
        coupling_hidden_channels: int | None = None,
        coupling_dropout: float = 0.0,
        coupling_init_gate: float = -4.0,
        coupling_use_norm: bool = True,
        coupling_param_hidden_dim: int = 64,
        coupling_condition_scale: float = 0.10,
        token_hidden_dim: int = 64,
        alpha_token: float = 1.0,
        gate_scale: float = 0.25,
    ) -> None:
        if gate_scale <= 0.0 or gate_scale > 1.0:
            raise ValueError(
                "gate_scale must be in (0, 1], "
                f"got {gate_scale}"
            )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            context_length=context_length,
            num_fields=num_fields,
            field_width=field_width,
            coupling_mode="static",
            coupling_hidden_channels=coupling_hidden_channels,
            coupling_dropout=coupling_dropout,
            coupling_init_gate=coupling_init_gate,
            coupling_use_norm=coupling_use_norm,
            coupling_param_hidden_dim=coupling_param_hidden_dim,
            coupling_condition_scale=coupling_condition_scale,
            token_hidden_dim=token_hidden_dim,
            alpha_token=alpha_token,
        )

        self.gate_scale = float(gate_scale)

        # Only three trainable parameters:
        # two input weights and one bias.
        self.module_gate = nn.Linear(
            in_features=2,
            out_features=1,
            bias=True,
        )

        # Exact initial alignment with M8-A:
        # score=0 -> token_weight=coupling_weight=1.
        nn.init.zeros_(self.module_gate.weight)
        nn.init.zeros_(self.module_gate.bias)

    @staticmethod
    def normalize_gate_param(
        param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize the two condition coordinates to comparable scales.

        Dataset ranges:

            log10(Ra): 6, 7, 8
            log10(Pr): approximately -0.30103, 0, 0.30103

        Result:

            normalized Ra approximately in [-1, 1]
            normalized Pr approximately in [-1, 1]
        """
        if param.ndim != 2 or param.shape[1] != 2:
            raise ValueError(
                "param must be [B, 2] = "
                "[log10(Ra), log10(Pr)], "
                f"got {tuple(param.shape)}"
            )

        log_ra = param[:, 0]
        log_pr = param[:, 1]

        ra_norm = log_ra - 7.0
        pr_norm = log_pr / 0.3010299956639812

        return torch.stack(
            [ra_norm, pr_norm],
            dim=1,
        )

    def module_weights(
        self,
        param: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return:

            gate_score:
                [B]

            token_weight:
                [B]

            coupling_weight:
                [B]
        """
        param_norm = self.normalize_gate_param(param)

        raw_score = self.module_gate(param_norm).squeeze(-1)
        gate_score = torch.tanh(raw_score)

        token_weight = (
            1.0
            + self.gate_scale * gate_score
        )

        coupling_weight = (
            1.0
            - self.gate_scale * gate_score
        )

        return (
            gate_score,
            token_weight,
            coupling_weight,
        )

    def encode_fields_with_gate(
        self,
        x: torch.Tensor,
        param: torch.Tensor,
        coupling_weight: torch.Tensor,
    ):
        """
        Apply static FieldCoupling, then gate only its residual contribution.
        """
        if x.ndim != 4:
            raise ValueError(
                f"x must be [B, C, H, W], got {tuple(x.shape)}"
            )

        batch_size, channels, height, spatial_width = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Wrong input channels: got {channels}, "
                f"expected {self.in_channels}"
            )

        if coupling_weight.shape != (batch_size,):
            raise ValueError(
                "coupling_weight must be [B], "
                f"got {tuple(coupling_weight.shape)}"
            )

        x_hist = x.reshape(
            batch_size,
            self.context_length,
            self.num_fields,
            height,
            spatial_width,
        )

        field_features = []

        for field_idx, encoder in enumerate(
            self.field_encoders
        ):
            x_field = x_hist[:, :, field_idx, :, :]
            field_feature = encoder(x_field)
            field_features.append(field_feature)

        field_stack = torch.stack(
            field_features,
            dim=1,
        )

        # Static FieldCoupling only.
        full_coupled_stack, coupling_info = self.field_coupling(
            field_stack,
            param=param,
            mode="static",
            return_diagnostics=True,
        )

        # Isolate the FieldCoupling residual:
        #
        # full_coupled_stack =
        #     field_stack + coupling_residual
        coupling_residual = (
            full_coupled_stack - field_stack
        )

        coupling_weight_view = coupling_weight.view(
            batch_size,
            1,
            1,
            1,
            1,
        )

        gated_coupled_stack = (
            field_stack
            + coupling_weight_view * coupling_residual
        )

        coupled_field_features = [
            gated_coupled_stack[:, field_idx, :, :, :]
            for field_idx in range(self.num_fields)
        ]

        fused = torch.cat(
            coupled_field_features,
            dim=1,
        )
        fused = self.fusion(fused)

        return (
            fused,
            field_features,
            coupled_field_features,
            coupling_info,
            coupling_residual,
        )

    def forward(
        self,
        x: torch.Tensor,
        param: torch.Tensor,
        return_features: bool = False,
    ):
        if param.ndim != 2 or param.shape[1] != 2:
            raise ValueError(
                "param must be [B, 2] = "
                "[log10(Ra), log10(Pr)], "
                f"got {tuple(param.shape)}"
            )

        param = param.to(
            device=x.device,
            dtype=x.dtype,
        )

        (
            gate_score,
            token_weight,
            coupling_weight,
        ) = self.module_weights(param)

        (
            x,
            field_features,
            coupled_field_features,
            coupling_info,
            coupling_residual,
        ) = self.encode_fields_with_gate(
            x=x,
            param=param,
            coupling_weight=coupling_weight,
        )

        raw_tokens = self.param_token(param)

        tokens = (
            self.alpha_token
            * token_weight[:, None, None]
            * raw_tokens
        )

        x = self.conv0(x) + self.w0(x)
        x = apply_param_token(
            x,
            tokens[:, 0, :],
        )
        x = F.gelu(x)

        x = self.conv1(x) + self.w1(x)
        x = apply_param_token(
            x,
            tokens[:, 1, :],
        )
        x = F.gelu(x)

        x = self.conv2(x) + self.w2(x)
        x = apply_param_token(
            x,
            tokens[:, 2, :],
        )
        x = F.gelu(x)

        x = self.conv3(x) + self.w3(x)
        x = apply_param_token(
            x,
            tokens[:, 3, :],
        )

        x = x.permute(0, 2, 3, 1)
        x = self.mlp0(x)
        x = F.gelu(x)
        x = self.mlp1(x)
        x = x.permute(0, 3, 1, 2)

        if return_features:
            return {
                "out": x,
                "coupling_mode": "static",
                "field_features": field_features,
                "coupled_field_features": (
                    coupled_field_features
                ),
                "coupling_residual": coupling_residual,
                "raw_param_tokens": raw_tokens,
                "param_tokens": tokens,
                "module_gate_score": gate_score,
                "token_module_weight": token_weight,
                "coupling_module_weight": coupling_weight,
                "gate_scale": self.gate_scale,
                "coupling_gate": coupling_info[
                    "coupling_gate"
                ],
                "base_coupling_matrix": coupling_info[
                    "base_coupling_matrix"
                ],
                "conditioned_delta_matrix": coupling_info[
                    "conditioned_delta_matrix"
                ],
                "effective_coupling_matrix": coupling_info[
                    "effective_coupling_matrix"
                ],
            }

        return x


if __name__ == "__main__":
    torch.manual_seed(7)

    batch_size = 2
    height = 16
    spatial_width = 64

    x = torch.randn(
        batch_size,
        16,
        height,
        spatial_width,
    )

    param = torch.tensor(
        [
            [6.0, -0.3010299956639812],
            [8.0,  0.3010299956639812],
        ],
        dtype=torch.float32,
    )

    base_model = M8FullConditionedFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=8,
        modes2=8,
        width=32,
        context_length=4,
        num_fields=4,
        coupling_mode="static",
    )

    gate_model = M8ModuleGateFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=8,
        modes2=8,
        width=32,
        context_length=4,
        num_fields=4,
        gate_scale=0.25,
    )

    missing_keys, unexpected_keys = (
        gate_model.load_state_dict(
            base_model.state_dict(),
            strict=False,
        )
    )

    print("missing_keys:", missing_keys)
    print("unexpected_keys:", unexpected_keys)

    allowed_missing = {
        "module_gate.weight",
        "module_gate.bias",
    }

    if set(missing_keys) != allowed_missing:
        raise RuntimeError(
            "Unexpected missing keys: "
            f"{missing_keys}"
        )

    if unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint keys: "
            f"{unexpected_keys}"
        )

    base_model.eval()
    gate_model.eval()

    with torch.no_grad():
        base_output = base_model(
            x,
            param,
            coupling_mode="static",
        )

        gate_info = gate_model(
            x,
            param,
            return_features=True,
        )

    gate_output = gate_info["out"]

    alignment_diff = (
        base_output - gate_output
    ).abs().max().item()

    print("input:", tuple(x.shape))
    print("param:", tuple(param.shape))
    print("output:", tuple(gate_output.shape))
    print("initial alignment max diff:", alignment_diff)
    print(
        "gate score:",
        gate_info["module_gate_score"]
        .detach()
        .cpu()
        .tolist(),
    )
    print(
        "token module weight:",
        gate_info["token_module_weight"]
        .detach()
        .cpu()
        .tolist(),
    )
    print(
        "coupling module weight:",
        gate_info["coupling_module_weight"]
        .detach()
        .cpu()
        .tolist(),
    )

    assert gate_output.shape == (
        batch_size,
        4,
        height,
        spatial_width,
    )

    assert alignment_diff < 1e-7

    assert torch.allclose(
        gate_info["token_module_weight"],
        torch.ones_like(
            gate_info["token_module_weight"]
        ),
    )

    assert torch.allclose(
        gate_info["coupling_module_weight"],
        torch.ones_like(
            gate_info["coupling_module_weight"]
        ),
    )

    print(
        "✅ M8ModuleGateFNO2d smoke test passed."
    )
