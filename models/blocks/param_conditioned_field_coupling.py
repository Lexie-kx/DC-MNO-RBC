"""
Static and parameter-conditioned field coupling for M8.

Modes
-----
static:
    Uses a shared learnable off-diagonal coupling matrix.

parameter_conditioned:
    Uses the same shared matrix plus a sample-dependent correction
    generated from [log10(Ra), log10(Pr), log10(nu), log10(kappa)].

The conditional correction head is zero-initialized. Therefore, before
training, parameter_conditioned mode exactly matches static mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ParamConditionedFieldCouplingBlock(nn.Module):
    """
    Field-to-field coupling with a controlled parameter-conditioned extension.

    Input:
        x:     [B, F, C, H, W]
        param: [B, 2] = [log10(Ra), log10(Pr)]

    Coupling matrix convention:
        matrix[target_field, source_field]

    Modes:
        static:
            effective_matrix = base_matrix

        parameter_conditioned:
            effective_matrix = base_matrix + parameter_delta_matrix
    """

    VALID_MODES = {"static", "parameter_conditioned"}

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
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_fields <= 1:
            raise ValueError(f"num_fields must be > 1, got {num_fields}")
        if param_hidden_dim <= 0:
            raise ValueError(
                f"param_hidden_dim must be positive, got {param_hidden_dim}"
            )
        if condition_scale <= 0:
            raise ValueError(
                f"condition_scale must be positive, got {condition_scale}"
            )

        self.channels = channels
        self.num_fields = num_fields
        self.hidden_channels = hidden_channels or channels
        self.param_hidden_dim = param_hidden_dim
        self.condition_scale = float(condition_scale)

        # Keep names aligned with the old FieldCouplingBlock where possible.
        self.norm = nn.GroupNorm(1, channels) if use_norm else nn.Identity()

        self.pre_proj = nn.Conv2d(
            channels,
            self.hidden_channels,
            kernel_size=1,
        )
        self.act = nn.GELU()
        self.post_proj = nn.Conv2d(
            self.hidden_channels,
            channels,
            kernel_size=1,
        )
        self.dropout = (
            nn.Dropout2d(dropout)
            if dropout > 0
            else nn.Identity()
        )

        # Shared base coupling matrix:
        # [target_field, source_field]
        self.coupling_matrix = nn.Parameter(
            torch.zeros(num_fields, num_fields, dtype=torch.float32)
        )

        # Only cross-field coupling is allowed.
        offdiag = torch.ones(
            num_fields,
            num_fields,
            dtype=torch.float32,
        )
        offdiag.fill_diagonal_(0.0)
        self.register_buffer("offdiag_mask", offdiag)

        # Shared per-target residual gate.
        # sigmoid(-4) ≈ 0.018, so coupling starts close to identity.
        self.residual_gate = nn.Parameter(
            torch.full(
                (num_fields,),
                float(init_gate),
                dtype=torch.float32,
            )
        )

        # Ra/Pr -> sample-dependent F x F correction.
        #
        # The final layer is zero-initialized, so initially:
        # parameter_delta_matrix == 0
        self.param_conditioner = nn.Sequential(
            nn.Linear(4, param_hidden_dim),
            nn.GELU(),
            nn.Linear(
                param_hidden_dim,
                num_fields * num_fields,
            ),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.pre_proj.weight, a=5**0.5)
        nn.init.zeros_(self.pre_proj.bias)

        nn.init.kaiming_uniform_(self.post_proj.weight, a=5**0.5)
        nn.init.zeros_(self.post_proj.bias)

        with torch.no_grad():
            self.coupling_matrix.normal_(mean=0.0, std=0.02)
            self.coupling_matrix.mul_(self.offdiag_mask)

        # Exact zero-condition alignment.
        nn.init.zeros_(self.param_conditioner[-1].weight)
        nn.init.zeros_(self.param_conditioner[-1].bias)

    @staticmethod
    def expand_param(param: torch.Tensor) -> torch.Tensor:
        """
        [log10(Ra), log10(Pr)]
        ->
        [log10(Ra), log10(Pr), log10(nu), log10(kappa)]
        """
        if param.ndim != 2 or param.shape[1] != 2:
            raise ValueError(
                "param must be [B, 2] = [log10(Ra), log10(Pr)], "
                f"got {tuple(param.shape)}"
            )

        log_ra = param[:, 0]
        log_pr = param[:, 1]

        log_nu = -0.5 * (log_ra - log_pr)
        log_kappa = -0.5 * (log_ra + log_pr)

        return torch.stack(
            [log_ra, log_pr, log_nu, log_kappa],
            dim=1,
        )

    def gate_values(self) -> torch.Tensor:
        """Return shared residual gates in [0, 1]."""
        return torch.sigmoid(self.residual_gate)

    def base_coupling_matrix(self) -> torch.Tensor:
        """Return the shared off-diagonal base matrix."""
        return self.coupling_matrix * self.offdiag_mask

    def conditioned_delta_matrix(
        self,
        param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return sample-dependent coupling correction.

        Shape:
            [B, F, F]
        """
        param4 = self.expand_param(param)

        delta = self.param_conditioner(param4)
        delta = delta.view(
            param.shape[0],
            self.num_fields,
            self.num_fields,
        )

        # Bound the dynamic correction and keep it initially small.
        delta = torch.tanh(delta) * self.condition_scale

        mask = self.offdiag_mask.to(
            device=delta.device,
            dtype=delta.dtype,
        )
        return delta * mask.unsqueeze(0)

    def effective_coupling_matrix(
        self,
        param: torch.Tensor | None = None,
        mode: str = "static",
    ) -> torch.Tensor:
        """
        Returns:
            static:
                [F, F]

            parameter_conditioned:
                [B, F, F]
        """
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown coupling mode: {mode}. "
                f"Expected one of {sorted(self.VALID_MODES)}"
            )

        base = self.base_coupling_matrix()

        if mode == "static":
            return base

        if param is None:
            raise ValueError(
                "param is required for parameter_conditioned mode"
            )

        delta = self.conditioned_delta_matrix(param)
        return base.unsqueeze(0) + delta

    def forward(
        self,
        x: torch.Tensor,
        param: torch.Tensor | None = None,
        mode: str = "static",
        return_diagnostics: bool = False,
    ):
        """
        Args:
            x:
                [B, F, C, H, W]

            param:
                [B, 2], required only for parameter_conditioned mode.

            mode:
                "static" or "parameter_conditioned"
        """
        if x.ndim != 5:
            raise ValueError(
                "Expected x as [B, F, C, H, W], "
                f"got {tuple(x.shape)}"
            )

        bsz, fields, channels, height, width = x.shape

        if fields != self.num_fields:
            raise ValueError(
                f"Expected num_fields={self.num_fields}, got {fields}"
            )
        if channels != self.channels:
            raise ValueError(
                f"Expected channels={self.channels}, got {channels}"
            )

        if param is not None:
            param = param.to(device=x.device, dtype=x.dtype)

        # Shared projection applied to each field independently.
        h = x.reshape(
            bsz * fields,
            channels,
            height,
            width,
        )
        h = self.norm(h)
        h = self.pre_proj(h)
        h = self.act(h)
        h = h.reshape(
            bsz,
            fields,
            self.hidden_channels,
            height,
            width,
        )

        weight = self.effective_coupling_matrix(
            param=param,
            mode=mode,
        ).to(device=h.device, dtype=h.dtype)

        if mode == "static":
            mixed = torch.einsum(
                "ij,bjchw->bichw",
                weight,
                h,
            )

            effective_batch = weight.unsqueeze(0).expand(
                bsz,
                -1,
                -1,
            )
            delta_batch = torch.zeros_like(effective_batch)

        else:
            mixed = torch.einsum(
                "bij,bjchw->bichw",
                weight,
                h,
            )

            effective_batch = weight
            base = self.base_coupling_matrix().to(
                device=weight.device,
                dtype=weight.dtype,
            )
            delta_batch = weight - base.unsqueeze(0)

        mixed = mixed.reshape(
            bsz * fields,
            self.hidden_channels,
            height,
            width,
        )
        mixed = self.post_proj(mixed)
        mixed = self.dropout(mixed)
        mixed = mixed.reshape(
            bsz,
            fields,
            channels,
            height,
            width,
        )

        gate = self.gate_values().to(
            device=x.device,
            dtype=x.dtype,
        )
        gate_view = gate.view(1, fields, 1, 1, 1)

        y = x + gate_view * mixed

        if return_diagnostics:
            return y, {
                "coupling_gate": gate,
                "base_coupling_matrix": self.base_coupling_matrix(),
                "conditioned_delta_matrix": delta_batch,
                "effective_coupling_matrix": effective_batch,
            }

        return y


if __name__ == "__main__":
    torch.manual_seed(7)

    block = ParamConditionedFieldCouplingBlock(
        channels=8,
        num_fields=4,
    )

    x = torch.randn(2, 4, 8, 16, 64)
    param = torch.tensor(
        [
            [6.0, 0.0],
            [8.0, 0.30103],
        ],
        dtype=torch.float32,
    )

    y_static, static_info = block(
        x,
        param=param,
        mode="static",
        return_diagnostics=True,
    )

    y_conditioned, conditioned_info = block(
        x,
        param=param,
        mode="parameter_conditioned",
        return_diagnostics=True,
    )

    max_diff = torch.max(
        torch.abs(y_static - y_conditioned)
    ).item()

    print("input:", tuple(x.shape))
    print("static output:", tuple(y_static.shape))
    print("conditioned output:", tuple(y_conditioned.shape))
    print("gate:", block.gate_values().detach().cpu().tolist())
    print("initial static-conditioned max diff:", max_diff)
    print(
        "conditioned delta abs max:",
        conditioned_info["conditioned_delta_matrix"]
        .abs()
        .max()
        .item(),
    )

    assert y_static.shape == x.shape
    assert y_conditioned.shape == x.shape
    assert max_diff < 1e-7

    print("✅ ParamConditionedFieldCouplingBlock smoke test passed.")
