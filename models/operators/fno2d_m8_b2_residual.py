"""
M8-B2 residual-adapter neural operator.

The complete trained M8-A model is retained as the static
base. Only the residual coupling adapter is intended to be
trained during the first controlled experiment.
"""

from __future__ import annotations

from models.operators.fno2d_m8_full_conditioned import (
    M8FullConditionedFNO2d,
)
from models.blocks.m8_b2_residual_adapter import (
    M8B2ResidualAdapterFieldCouplingBlock,
)


class M8B2ResidualAdapterFNO2d(
    M8FullConditionedFNO2d
):
    """
    M8-A static base plus a zero-initialized coupling adapter.

    adapter_mode:
        static_adapter:
            Fair M8-A2 control.

        parameter_adapter:
            M8-B2 formal candidate.
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
        adapter_mode: str = "parameter_adapter",
        adapter_gate_hidden_dim: int = 32,
        adapter_gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            context_length=context_length,
            num_fields=num_fields,
            field_width=field_width,
            coupling_mode="parameter_conditioned",
            coupling_hidden_channels=(
                coupling_hidden_channels
            ),
            coupling_dropout=coupling_dropout,
            coupling_init_gate=(
                coupling_init_gate
            ),
            coupling_use_norm=coupling_use_norm,
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
        self.adapter_gate_hidden_dim = (
            adapter_gate_hidden_dim
        )
        self.adapter_gate_init_bias = float(
            adapter_gate_init_bias
        )

        # Replace only the field-coupling block.
        # All parameter names inherited from M8-A remain
        # compatible, except for the new adapter_gate keys.
        self.field_coupling = (
            M8B2ResidualAdapterFieldCouplingBlock(
                channels=self.field_width,
                num_fields=self.num_fields,
                hidden_channels=(
                    coupling_hidden_channels
                    or self.field_width
                ),
                dropout=coupling_dropout,
                init_gate=coupling_init_gate,
                use_norm=coupling_use_norm,
                param_hidden_dim=(
                    coupling_param_hidden_dim
                ),
                condition_scale=(
                    coupling_condition_scale
                ),
                adapter_mode=adapter_mode,
                adapter_gate_hidden_dim=(
                    adapter_gate_hidden_dim
                ),
                adapter_gate_init_bias=(
                    adapter_gate_init_bias
                ),
            )
        )
