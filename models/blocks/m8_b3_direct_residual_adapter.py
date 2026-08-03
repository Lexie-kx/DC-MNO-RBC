"""
Direct residual field adapter for M8-B3.

static_adapter:
    Receives a constant input and learns one shared correction.

parameter_adapter:
    Receives [logRa, logPr, logNu, logKappa] and learns
    condition-dependent corrections.

The adapter is added directly after the frozen M8-A coupling
output. It is not multiplied by M8-A's frozen coupling gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class M8B3DirectResidualFieldAdapter(nn.Module):
    VALID_ADAPTER_MODES = {
        "static_adapter",
        "parameter_adapter",
    }

    def __init__(
        self,
        channels: int,
        num_fields: int = 4,
        hidden_channels: int | None = None,
        param_hidden_dim: int = 64,
        condition_scale: float = 0.10,
        adapter_mode: str = "parameter_adapter",
        gate_hidden_dim: int = 32,
        gate_init_bias: float = -2.0,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        if adapter_mode not in self.VALID_ADAPTER_MODES:
            raise ValueError(
                f"Unknown adapter_mode={adapter_mode}"
            )

        if condition_scale <= 0:
            raise ValueError(
                "condition_scale must be positive"
            )

        self.channels = channels
        self.num_fields = num_fields
        self.hidden_channels = (
            hidden_channels or channels
        )
        self.param_hidden_dim = param_hidden_dim
        self.condition_scale = float(
            condition_scale
        )
        self.adapter_mode = adapter_mode
        self.gate_hidden_dim = gate_hidden_dim
        self.gate_init_bias = float(
            gate_init_bias
        )

        self.norm = (
            nn.GroupNorm(1, channels)
            if use_norm
            else nn.Identity()
        )

        self.pre_proj = nn.Conv2d(
            channels,
            self.hidden_channels,
            kernel_size=1,
            bias=False,
        )

        self.act = nn.GELU()

        self.post_proj = nn.Conv2d(
            self.hidden_channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        # Condition -> F x F residual matrix.
        self.param_conditioner = nn.Sequential(
            nn.Linear(
                4,
                param_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                param_hidden_dim,
                num_fields * num_fields,
            ),
        )

        # Condition -> per-target-field gate.
        self.adapter_gate = nn.Sequential(
            nn.Linear(
                4,
                gate_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                gate_hidden_dim,
                num_fields,
            ),
        )

        offdiag = torch.ones(
            num_fields,
            num_fields,
            dtype=torch.float32,
        )
        offdiag.fill_diagonal_(0.0)

        self.register_buffer(
            "offdiag_mask",
            offdiag,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(
            self.pre_proj.weight,
            a=5**0.5,
        )
        nn.init.kaiming_uniform_(
            self.post_proj.weight,
            a=5**0.5,
        )
        # Exact zero-output initialization.
        nn.init.zeros_(
            self.param_conditioner[-1].weight
        )
        nn.init.zeros_(
            self.param_conditioner[-1].bias
        )

        nn.init.zeros_(
            self.adapter_gate[-1].weight
        )
        nn.init.constant_(
            self.adapter_gate[-1].bias,
            self.gate_init_bias,
        )

    @staticmethod
    def expand_param(
        param: torch.Tensor,
    ) -> torch.Tensor:
        if (
            param.ndim != 2
            or param.shape[1] != 2
        ):
            raise ValueError(
                "param must be [B, 2] = "
                "[log10(Ra), log10(Pr)]"
            )

        log_ra = param[:, 0]
        log_pr = param[:, 1]

        log_nu = -0.5 * (
            log_ra - log_pr
        )

        log_kappa = -0.5 * (
            log_ra + log_pr
        )

        return torch.stack(
            [
                log_ra,
                log_pr,
                log_nu,
                log_kappa,
            ],
            dim=1,
        )

    def adapter_features(
        self,
        param: torch.Tensor,
    ) -> torch.Tensor:
        param4 = self.expand_param(param)

        if self.adapter_mode == "static_adapter":
            return torch.ones_like(param4)

        return param4

    def gate_values(
        self,
        param: torch.Tensor,
    ) -> torch.Tensor:
        features = self.adapter_features(
            param
        )

        return torch.sigmoid(
            self.adapter_gate(features)
        )

    def delta_matrix(
        self,
        param: torch.Tensor,
    ) -> torch.Tensor:
        features = self.adapter_features(
            param
        )

        delta = self.param_conditioner(
            features
        )

        delta = delta.view(
            param.shape[0],
            self.num_fields,
            self.num_fields,
        )

        delta = (
            torch.tanh(delta)
            * self.condition_scale
        )

        mask = self.offdiag_mask.to(
            device=delta.device,
            dtype=delta.dtype,
        )

        return (
            delta
            * mask.unsqueeze(0)
        )

    def forward(
        self,
        x: torch.Tensor,
        param: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        """
        Args:
            x: [B, F, C, H, W]

        Returns:
            direct residual:
                [B, F, C, H, W]
        """
        if x.ndim != 5:
            raise ValueError(
                "x must be [B, F, C, H, W]"
            )

        (
            batch_size,
            num_fields,
            channels,
            height,
            width,
        ) = x.shape

        if num_fields != self.num_fields:
            raise ValueError(
                "Wrong number of fields"
            )

        if channels != self.channels:
            raise ValueError(
                "Wrong channel width"
            )

        param = param.to(
            device=x.device,
            dtype=x.dtype,
        )

        h = x.reshape(
            batch_size * num_fields,
            channels,
            height,
            width,
        )

        h = self.norm(h)
        h = self.pre_proj(h)
        h = self.act(h)

        h = h.reshape(
            batch_size,
            num_fields,
            self.hidden_channels,
            height,
            width,
        )

        delta = self.delta_matrix(param)
        gate = self.gate_values(param)

        # Apply a separate gate for every target field.
        gated_delta = (
            delta
            * gate.unsqueeze(-1)
        )

        mixed = torch.einsum(
            "bij,bjchw->bichw",
            gated_delta,
            h,
        )

        mixed = mixed.reshape(
            batch_size * num_fields,
            self.hidden_channels,
            height,
            width,
        )

        mixed = self.post_proj(mixed)

        residual = mixed.reshape(
            batch_size,
            num_fields,
            channels,
            height,
            width,
        )

        if return_diagnostics:
            return residual, {
                "adapter_gate_batch": gate,
                "delta_matrix": delta,
                "gated_delta_matrix": (
                    gated_delta
                ),
                "adapter_mode": (
                    self.adapter_mode
                ),
            }

        return residual
