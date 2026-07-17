"""
M9-0a-StaticSoftmaxFieldAttention-H4 operator.

Position:
    FieldWise Encoder
    -> Static Softmax Field Attention
    -> Fusion
    -> FNO
    -> normalized delta prediction

Experiment type:
    Controlled experiment.

No ParameterToken.
No parameter conditioning.
No physics prior.
No PDE loss.
No rollout logic inside this file.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.field_attention import (
    StaticSoftmaxFieldAttentionBlock,
)
from models.operators.fno2d import SpectralConv2d


class M9StaticAttentionFNO2d(nn.Module):
    """
    M9-0a: fixed Softmax field-axis Attention.

    Input:
        x: [B, 16, H, W]

    Field order:
        [buoyancy, u_x, u_y, pressure]

    attention[target_field, source_field]:
        target field receives information from source field.
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
        attention_hidden_channels: int | None = None,
        attention_dropout: float = 0.0,
        attention_init_gate: float = -4.0,
        attention_use_norm: bool = True,
        attention_output_scale: float = 0.05675224,
        score_temperature: float = 0.03,
    ) -> None:
        super().__init__()

        if in_channels != context_length * num_fields:
            raise ValueError(
                f"in_channels={in_channels} must equal "
                f"context_length*num_fields="
                f"{context_length * num_fields}"
            )

        if field_width is None:
            if width % num_fields != 0:
                raise ValueError(
                    f"width={width} cannot be divided by "
                    f"num_fields={num_fields}"
                )

            field_width = width // num_fields

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.context_length = context_length
        self.num_fields = num_fields
        self.field_width = field_width

        fused_width = field_width * num_fields

        # Same names and structure as M6.
        self.field_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        context_length,
                        field_width,
                        kernel_size=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        field_width,
                        field_width,
                        kernel_size=1,
                    ),
                )
                for _ in range(num_fields)
            ]
        )

        # The only M9-0a component added to the M6 backbone.
        self.field_attention = (
            StaticSoftmaxFieldAttentionBlock(
                channels=field_width,
                num_fields=num_fields,
                hidden_channels=(
                    attention_hidden_channels
                    or field_width
                ),
                dropout=attention_dropout,
                init_gate=attention_init_gate,
                use_norm=attention_use_norm,
                attention_output_scale=(
                    attention_output_scale
                ),
                score_temperature=score_temperature,
            )
        )

        # Same M6 fusion and FNO module names.
        self.fusion = nn.Conv2d(
            fused_width,
            width,
            kernel_size=1,
        )

        self.conv0 = SpectralConv2d(
            width,
            width,
            modes1,
            modes2,
        )
        self.conv1 = SpectralConv2d(
            width,
            width,
            modes1,
            modes2,
        )
        self.conv2 = SpectralConv2d(
            width,
            width,
            modes1,
            modes2,
        )
        self.conv3 = SpectralConv2d(
            width,
            width,
            modes1,
            modes2,
        )

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.mlp0 = nn.Linear(width, 128)
        self.mlp1 = nn.Linear(128, out_channels)

    def encode_fields(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        attention_override: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        dict[str, Any] | None,
    ]:
        """
        Returns:
            fused:
                [B, width, H, W]

            field_features:
                four tensors [B, field_width, H, W]

            attended_field_features:
                four tensors [B, field_width, H, W]

            attention_diagnostics:
                None, or the block diagnostics dictionary
        """
        if x.ndim != 4:
            raise ValueError(
                f"x must be [B,16,H,W], got {tuple(x.shape)}"
            )

        batch_size, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"wrong input channels: got {channels}, "
                f"expected {self.in_channels}"
            )

        x_hist = x.reshape(
            batch_size,
            self.context_length,
            self.num_fields,
            height,
            width,
        )

        field_features = []

        for field_idx, encoder in enumerate(
            self.field_encoders
        ):
            x_field = x_hist[
                :,
                :,
                field_idx,
                :,
                :,
            ]

            field_features.append(
                encoder(x_field)
            )

        field_stack = torch.stack(
            field_features,
            dim=1,
        )

        attention_result = self.field_attention(
            field_stack,
            return_diagnostics=return_attention,
            attention_override=attention_override,
        )

        if return_attention:
            if not isinstance(
                attention_result,
                dict,
            ):
                raise TypeError(
                    "Expected diagnostics dictionary."
                )

            attended_stack = attention_result["out"]
            attention_diagnostics = attention_result
        else:
            if not isinstance(
                attention_result,
                torch.Tensor,
            ):
                raise TypeError(
                    "Expected attended feature tensor."
                )

            attended_stack = attention_result
            attention_diagnostics = None

        attended_field_features = [
            attended_stack[
                :,
                field_idx,
                :,
                :,
                :,
            ]
            for field_idx in range(
                self.num_fields
            )
        ]

        fused = torch.cat(
            attended_field_features,
            dim=1,
        )
        fused = self.fusion(fused)

        return (
            fused,
            field_features,
            attended_field_features,
            attention_diagnostics,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        attention_override: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        (
            x,
            field_features,
            attended_field_features,
            attention_diagnostics,
        ) = self.encode_fields(
            x,
            return_attention=return_features,
            attention_override=attention_override,
        )

        x = F.gelu(
            self.conv0(x) + self.w0(x)
        )
        x = F.gelu(
            self.conv1(x) + self.w1(x)
        )
        x = F.gelu(
            self.conv2(x) + self.w2(x)
        )
        x = self.conv3(x) + self.w3(x)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.mlp0(x))
        x = self.mlp1(x)
        x = x.permute(0, 3, 1, 2)

        if not return_features:
            return x

        return {
            "out": x,
            "field_features": field_features,
            "attended_field_features": (
                attended_field_features
            ),
            "field_attention": (
                attention_diagnostics
            ),
        }


StaticAttentionFNO2d = M9StaticAttentionFNO2d


if __name__ == "__main__":
    torch.manual_seed(20260717)

    model = M9StaticAttentionFNO2d()
    x = torch.randn(2, 16, 32, 64)

    info = model(
        x,
        return_features=True,
    )

    print("input:", tuple(x.shape))
    print("output:", tuple(info["out"].shape))
    print(
        "attention:",
        tuple(
            info["field_attention"][
                "attention"
            ].shape
        ),
    )

    assert info["out"].shape == (
        2,
        4,
        32,
        64,
    )

    assert info["field_attention"][
        "attention"
    ].shape == (
        2,
        4,
        4,
    )

    print(
        "✅ M9StaticAttentionFNO2d "
        "basic smoke passed."
    )
