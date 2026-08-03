"""
M8-B3 direct residual adapter.

Frozen M8-A coupling:
    base_coupled = M8-A static FieldCoupling(field_features)

Direct adapter:
    adapter_residual = R(field_features, condition)

Combined:
    coupled = base_coupled + adapter_residual

The new adapter does not pass through M8-A's frozen coupling gate.
"""

from __future__ import annotations

import torch

from models.operators.fno2d_m8_full_conditioned import (
    M8FullConditionedFNO2d,
)
from models.blocks.m8_b3_direct_residual_adapter import (
    M8B3DirectResidualFieldAdapter,
)


class M8B3DirectResidualAdapterFNO2d(
    M8FullConditionedFNO2d
):
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
        adapter_mode: str = "parameter_adapter",
        direct_hidden_channels: int | None = None,
        direct_param_hidden_dim: int = 64,
        direct_condition_scale: float = 0.10,
        direct_gate_hidden_dim: int = 32,
        direct_gate_init_bias: float = -2.0,
    ) -> None:
        # The inherited M8-A coupling always remains static.
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
            coupling_hidden_channels=(
                coupling_hidden_channels
            ),
            coupling_dropout=coupling_dropout,
            coupling_init_gate=(
                coupling_init_gate
            ),
            coupling_use_norm=(
                coupling_use_norm
            ),
            coupling_param_hidden_dim=(
                coupling_param_hidden_dim
            ),
            coupling_condition_scale=(
                coupling_condition_scale
            ),
            token_hidden_dim=token_hidden_dim,
            alpha_token=alpha_token,
        )

        self.adapter_mode = adapter_mode

        self.direct_adapter = (
            M8B3DirectResidualFieldAdapter(
                channels=self.field_width,
                num_fields=self.num_fields,
                hidden_channels=(
                    direct_hidden_channels
                    or self.field_width
                ),
                param_hidden_dim=(
                    direct_param_hidden_dim
                ),
                condition_scale=(
                    direct_condition_scale
                ),
                adapter_mode=adapter_mode,
                gate_hidden_dim=(
                    direct_gate_hidden_dim
                ),
                gate_init_bias=(
                    direct_gate_init_bias
                ),
                use_norm=True,
            )
        )

    def encode_fields(
        self,
        x: torch.Tensor,
        param: torch.Tensor,
        coupling_mode: str,
    ):
        if x.ndim != 4:
            raise ValueError(
                "x must be [B, C, H, W]"
            )

        (
            batch_size,
            channels,
            height,
            spatial_width,
        ) = x.shape

        if channels != self.in_channels:
            raise ValueError(
                "Wrong input channel count"
            )

        x_hist = x.reshape(
            batch_size,
            self.context_length,
            self.num_fields,
            height,
            spatial_width,
        )

        field_features = []

        for field_index, encoder in enumerate(
            self.field_encoders
        ):
            field_input = x_hist[
                :,
                :,
                field_index,
                :,
                :,
            ]

            field_features.append(
                encoder(field_input)
            )

        field_stack = torch.stack(
            field_features,
            dim=1,
        )

        # Frozen M8-A static coupling path.
        (
            base_coupled_stack,
            base_info,
        ) = self.field_coupling(
            field_stack,
            param=param,
            mode="static",
            return_diagnostics=True,
        )

        # New direct residual path.
        (
            adapter_residual,
            adapter_info,
        ) = self.direct_adapter(
            field_stack,
            param,
            return_diagnostics=True,
        )

        combined_stack = (
            base_coupled_stack
            + adapter_residual
        )

        coupled_field_features = [
            combined_stack[
                :,
                field_index,
                :,
                :,
                :,
            ]
            for field_index
            in range(self.num_fields)
        ]

        fused = torch.cat(
            coupled_field_features,
            dim=1,
        )

        fused = self.fusion(fused)

        coupling_info = dict(base_info)

        coupling_info[
            "conditioned_delta_matrix"
        ] = adapter_info[
            "gated_delta_matrix"
        ]

        coupling_info[
            "direct_adapter_delta_matrix"
        ] = adapter_info[
            "delta_matrix"
        ]

        coupling_info[
            "direct_adapter_gate"
        ] = adapter_info[
            "adapter_gate_batch"
        ]

        coupling_info[
            "adapter_mode"
        ] = self.adapter_mode

        return (
            fused,
            field_features,
            coupled_field_features,
            coupling_info,
        )
