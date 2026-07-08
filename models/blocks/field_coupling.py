"""
Field-to-field coupling block for M7-FieldCoupling-H4.

This module is intentionally controlled:
- Input : [B, F, C, H, W]
- Output: [B, F, C, H, W]
- F is the number of physical fields, default 4:
  [buoyancy, u_x, u_y, pressure]

Design:
- Per-field 1x1 projection
- Learnable F x F off-diagonal field coupling matrix
- Residual gate initialized close to identity
- No ParameterToken
- No PDE loss
- No rollout logic
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FieldCouplingBlock(nn.Module):
    """
    Explicit field-to-field coupling block.

    Args:
        channels: feature channels per field.
        num_fields: number of physical fields.
        hidden_channels: hidden channels inside coupling projection.
        dropout: dropout probability.
        init_gate: initial residual gate logit. Negative value makes the block
            start close to identity, useful when inserted into a pretrained model.
        use_norm: whether to apply per-field GroupNorm before projection.
    """

    def __init__(
        self,
        channels: int,
        num_fields: int = 4,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
        init_gate: float = -4.0,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_fields <= 1:
            raise ValueError(f"num_fields must be > 1, got {num_fields}")

        self.channels = channels
        self.num_fields = num_fields
        self.hidden_channels = hidden_channels or channels

        self.norm = nn.GroupNorm(1, channels) if use_norm else nn.Identity()

        self.pre_proj = nn.Conv2d(channels, self.hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.post_proj = nn.Conv2d(self.hidden_channels, channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Coupling matrix shape:
        #   [target_field, source_field]
        self.coupling_matrix = nn.Parameter(
            torch.zeros(num_fields, num_fields, dtype=torch.float32)
        )

        # Only allow cross-field coupling here.
        # Self information is already preserved by the residual path.
        offdiag = torch.ones(num_fields, num_fields, dtype=torch.float32)
        offdiag.fill_diagonal_(0.0)
        self.register_buffer("offdiag_mask", offdiag)

        # Per-target-field gate.
        # sigmoid(-4) ≈ 0.018, so the block starts close to identity.
        self.residual_gate = nn.Parameter(
            torch.full((num_fields,), float(init_gate), dtype=torch.float32)
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

    def gate_values(self) -> torch.Tensor:
        """Return current residual gate values in [0, 1]."""
        return torch.sigmoid(self.residual_gate)

    def effective_coupling_matrix(self) -> torch.Tensor:
        """Return the off-diagonal coupling matrix used in forward."""
        return self.coupling_matrix * self.offdiag_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, F, C, H, W]

        Returns:
            y: [B, F, C, H, W]
        """
        if x.ndim != 5:
            raise ValueError(
                f"FieldCouplingBlock expects [B, F, C, H, W], got {tuple(x.shape)}"
            )

        bsz, fields, channels, height, width = x.shape

        if fields != self.num_fields:
            raise ValueError(f"Expected num_fields={self.num_fields}, got {fields}")
        if channels != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {channels}")

        # Apply shared projection to each field independently.
        h = x.reshape(bsz * fields, channels, height, width)
        h = self.norm(h)
        h = self.pre_proj(h)
        h = self.act(h)
        h = h.reshape(bsz, fields, self.hidden_channels, height, width)

        # Field-to-field mixing at each spatial location.
        # target i receives information from source j.
        weight = self.effective_coupling_matrix().to(dtype=h.dtype, device=h.device)
        mixed = torch.einsum("ij,bjchw->bichw", weight, h)

        mixed = mixed.reshape(bsz * fields, self.hidden_channels, height, width)
        mixed = self.post_proj(mixed)
        mixed = self.dropout(mixed)
        mixed = mixed.reshape(bsz, fields, channels, height, width)

        gate = self.gate_values().to(dtype=x.dtype, device=x.device)
        gate = gate.view(1, fields, 1, 1, 1)

        return x + gate * mixed


if __name__ == "__main__":
    block = FieldCouplingBlock(channels=32, num_fields=4)
    x = torch.randn(2, 4, 32, 16, 64)
    y = block(x)

    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    print("gate  :", block.gate_values().detach().cpu().tolist())
