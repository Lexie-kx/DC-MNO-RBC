"""
M8-B2 residual field-coupling adapter.

Fair controlled adapter modes
-----------------------------
static_adapter:
    The adapter receives a constant four-dimensional input.
    It learns one shared residual coupling correction for all conditions.

parameter_adapter:
    The adapter receives:
        [log10(Ra), log10(Pr), log10(nu), log10(kappa)]
    It learns a condition-dependent residual coupling correction.

Both modes:
    - use exactly the same module structure;
    - have exactly the same parameter count;
    - start from zero residual correction;
    - retain the frozen M8-A static coupling as the base.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.blocks.param_conditioned_field_coupling import (
    ParamConditionedFieldCouplingBlock,
)


class M8B2ResidualAdapterFieldCouplingBlock(
    ParamConditionedFieldCouplingBlock
):
    """
    M8-A base coupling plus a controlled residual adapter.

    Matrix convention:
        matrix[target_field, source_field]

    effective_matrix:
        base_matrix
        + target_gate(adapter_input)
        * residual_delta(adapter_input)

    adapter_mode:
        static_adapter:
            adapter_input = [1, 1, 1, 1]

        parameter_adapter:
            adapter_input =
                [logRa, logPr, logNu, logKappa]
    """

    VALID_ADAPTER_MODES = {
        "static_adapter",
        "parameter_adapter",
    }

    def __init__(
        self,
        channels: int,
        num_fields: int = 4,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
        init_gate: float = -4.0,
        use_norm: bool = True,
        param_hidden_dim: int = 64,
        condition_scale: float = 0.10,
        adapter_mode: str = "parameter_adapter",
        adapter_gate_hidden_dim: int = 32,
        adapter_gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__(
            channels=channels,
            num_fields=num_fields,
            hidden_channels=hidden_channels,
            dropout=dropout,
            init_gate=init_gate,
            use_norm=use_norm,
            param_hidden_dim=param_hidden_dim,
            condition_scale=condition_scale,
        )

        if adapter_mode not in self.VALID_ADAPTER_MODES:
            raise ValueError(
                f"Unknown adapter_mode={adapter_mode}. "
                f"Expected one of "
                f"{sorted(self.VALID_ADAPTER_MODES)}"
            )

        if adapter_gate_hidden_dim <= 0:
            raise ValueError(
                "adapter_gate_hidden_dim must be positive, "
                f"got {adapter_gate_hidden_dim}"
            )

        self.adapter_mode = adapter_mode
        self.adapter_gate_hidden_dim = (
            adapter_gate_hidden_dim
        )
        self.adapter_gate_init_bias = float(
            adapter_gate_init_bias
        )

        # Per-target-field adapter strength:
        # [B, 4] in (0, 1).
        #
        # The last weight is initialized to zero, so the
        # initial gate is determined only by init_bias.
        self.adapter_gate = nn.Sequential(
            nn.Linear(
                4,
                adapter_gate_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                adapter_gate_hidden_dim,
                num_fields,
            ),
        )

        nn.init.zeros_(
            self.adapter_gate[-1].weight
        )
        nn.init.constant_(
            self.adapter_gate[-1].bias,
            self.adapter_gate_init_bias,
        )

    def adapter_features(
        self,
        param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the adapter input.

        The tensor shape is always [B, 4], ensuring that
        static_adapter and parameter_adapter have exactly
        the same network structure.
        """
        param4 = self.expand_param(param)

        if self.adapter_mode == "static_adapter":
            return torch.ones_like(param4)

        return param4

    def adapter_gate_values(
        self,
        param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return per-target-field adapter gates.

        Shape:
            [B, num_fields]
        """
        features = self.adapter_features(param)
        logits = self.adapter_gate(features)
        return torch.sigmoid(logits)

    def conditioned_delta_matrix(
        self,
        param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the gated residual coupling correction.

        Shape:
            [B, target_field, source_field]
        """
        features = self.adapter_features(param)

        raw_delta = self.param_conditioner(
            features
        )

        delta = raw_delta.view(
            param.shape[0],
            self.num_fields,
            self.num_fields,
        )

        delta = (
            torch.tanh(delta)
            * self.condition_scale
        )

        # Each target field independently decides how
        # strongly the residual correction is applied.
        target_gate = self.adapter_gate_values(
            param
        )

        delta = (
            delta
            * target_gate.unsqueeze(-1)
        )

        mask = self.offdiag_mask.to(
            device=delta.device,
            dtype=delta.dtype,
        )

        return delta * mask.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        param: torch.Tensor | None = None,
        mode: str = "parameter_conditioned",
        return_diagnostics: bool = False,
    ):
        result = super().forward(
            x=x,
            param=param,
            mode=mode,
            return_diagnostics=return_diagnostics,
        )

        if not return_diagnostics:
            return result

        output, diagnostics = result

        if (
            mode == "parameter_conditioned"
            and param is not None
        ):
            adapter_gate_batch = (
                self.adapter_gate_values(param)
            )
        else:
            adapter_gate_batch = torch.zeros(
                x.shape[0],
                self.num_fields,
                device=x.device,
                dtype=x.dtype,
            )

        diagnostics[
            "adapter_gate_batch"
        ] = adapter_gate_batch

        diagnostics[
            "adapter_mode"
        ] = self.adapter_mode

        return output, diagnostics
