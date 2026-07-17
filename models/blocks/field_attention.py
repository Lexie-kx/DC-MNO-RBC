"""
M9-0 field-axis attention blocks.

Input / output:
    [B, F, C, H, W]

Field order:
    [buoyancy, u_x, u_y, pressure]

M9-0a:
    StaticSoftmaxFieldAttentionBlock

M9-0b:
    StateDependentFieldAttentionBlock

Controlled design:
- same M7-style Norm -> pre_proj -> GELU -> field mixing
  -> post_proj -> per-target gate -> residual
- off-diagonal attention only
- attention[b, i, j] means target field i receives source field j
- softmax is applied over source field j (dim=-1)
- V = h; no extra V projection
- attention_output_scale is fixed and saved in state_dict
- no ParameterToken, physics prior, PDE loss, or rollout logic
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class _BaseSoftmaxFieldAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_fields: int = 4,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
        init_gate: float = -4.0,
        use_norm: bool = True,
        attention_output_scale: float = 0.06,
        score_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_fields <= 1:
            raise ValueError(f"num_fields must be > 1, got {num_fields}")
        if hidden_channels is not None and hidden_channels <= 0:
            raise ValueError(
                f"hidden_channels must be positive or None, got {hidden_channels}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if attention_output_scale < 0.0:
            raise ValueError(
                "attention_output_scale must be non-negative, "
                f"got {attention_output_scale}"
            )
        if score_temperature <= 0.0:
            raise ValueError(
                f"score_temperature must be positive, got {score_temperature}"
            )

        self.channels = channels
        self.num_fields = num_fields
        self.hidden_channels = hidden_channels or channels
        self.score_temperature = float(score_temperature)

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
            if dropout > 0.0
            else nn.Identity()
        )

        # 与 M7 相同：每个目标场一个 sigmoid residual gate。
        self.residual_gate = nn.Parameter(
            torch.full(
                (num_fields,),
                float(init_gate),
                dtype=torch.float32,
            )
        )

        # True 表示允许该 target-source 边。
        # 对角线为 False：禁止 self-attention，
        # 自身信息由 residual path 保留。
        self.register_buffer(
            "offdiag_mask",
            ~torch.eye(num_fields, dtype=torch.bool),
        )

        # 固定缩放，不参与训练。
        # 当前 0.06 只是待校准候选值。
        self.register_buffer(
            "attention_output_scale",
            torch.tensor(
                float(attention_output_scale),
                dtype=torch.float32,
            ),
        )

        self.reset_shared_parameters()

    def reset_shared_parameters(self) -> None:
        nn.init.kaiming_uniform_(
            self.pre_proj.weight,
            a=5**0.5,
        )
        nn.init.zeros_(self.pre_proj.bias)

        nn.init.kaiming_uniform_(
            self.post_proj.weight,
            a=5**0.5,
        )
        nn.init.zeros_(self.post_proj.bias)

    def gate_values(self) -> torch.Tensor:
        """Return effective per-target gate values in [0, 1]."""
        return torch.sigmoid(self.residual_gate)

    @torch.no_grad()
    def set_attention_output_scale(self, value: float) -> None:
        """Update the fixed attention output scale."""
        if value < 0.0:
            raise ValueError(
                f"scale must be non-negative, got {value}"
            )

        self.attention_output_scale.fill_(float(value))

    @torch.no_grad()
    def copy_shared_branch_from(self, other: nn.Module) -> None:
        """
        Copy Norm / pre_proj / post_proj / residual_gate from a compatible
        M7, M9-0a, or M9-0b block.

        This is used for paired initialization during branch-norm calibration.
        """
        for name in (
            "norm",
            "pre_proj",
            "post_proj",
            "residual_gate",
        ):
            if not hasattr(other, name):
                raise AttributeError(
                    f"source block has no attribute {name!r}"
                )

        self.norm.load_state_dict(other.norm.state_dict())
        self.pre_proj.load_state_dict(
            other.pre_proj.state_dict()
        )
        self.post_proj.load_state_dict(
            other.post_proj.state_dict()
        )

        if self.residual_gate.shape != other.residual_gate.shape:
            raise ValueError(
                "residual_gate shape mismatch: "
                f"{tuple(self.residual_gate.shape)} vs "
                f"{tuple(other.residual_gate.shape)}"
            )

        self.residual_gate.copy_(other.residual_gate)

    def _check_x(
        self,
        x: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if x.ndim != 5:
            raise ValueError(
                f"expected [B,F,C,H,W], got {tuple(x.shape)}"
            )

        bsz, fields, channels, height, width = x.shape

        if fields != self.num_fields:
            raise ValueError(
                f"expected num_fields={self.num_fields}, "
                f"got {fields}"
            )

        if channels != self.channels:
            raise ValueError(
                f"expected channels={self.channels}, "
                f"got {channels}"
            )

        return bsz, fields, channels, height, width

    def project_fields(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the same M7-style shared field projection.

        Input:
            x: [B, F, C, H, W]

        Output:
            h: [B, F, hidden_channels, H, W]
        """
        bsz, fields, channels, height, width = self._check_x(x)

        h = x.reshape(
            bsz * fields,
            channels,
            height,
            width,
        )
        h = self.norm(h)
        h = self.pre_proj(h)
        h = self.act(h)

        return h.reshape(
            bsz,
            fields,
            self.hidden_channels,
            height,
            width,
        )

    def compute_scores(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return unmasked scores with shape:

            [B, target_field, source_field]
        """
        raise NotImplementedError

    def scores_to_attention(
        self,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert raw scores into off-diagonal row-normalized attention.

        attention[b, i, j]:
            target field i receives source field j.

        Softmax is applied along source-field dimension j.
        """
        if scores.ndim == 2:
            scores = scores.unsqueeze(0)

        if (
            scores.ndim != 3
            or tuple(scores.shape[-2:])
            != (self.num_fields, self.num_fields)
        ):
            raise ValueError(
                "scores must be [B,F,F] or [F,F], "
                f"got {tuple(scores.shape)}"
            )

        if not torch.is_floating_point(scores):
            raise TypeError(
                f"scores must be floating point, "
                f"got {scores.dtype}"
            )

        logits = scores / self.score_temperature

        mask = self.offdiag_mask.view(
            1,
            self.num_fields,
            self.num_fields,
        )

        # 混合精度安全的对角屏蔽。
        logits = logits.masked_fill(
            ~mask,
            torch.finfo(logits.dtype).min,
        )

        # 沿 source field j 做 Softmax。
        attention = torch.softmax(
            logits,
            dim=-1,
        )

        if not torch.isfinite(attention).all():
            raise FloatingPointError(
                "attention contains NaN/Inf"
            )

        return attention

    @staticmethod
    def mix_values(
        h: torch.Tensor,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mix source fields into target fields.

        h:
            [B, source_field, C, H, W]

        attention:
            [B, target_field, source_field]

        Returns:
            mixed: [B, target_field, C, H, W]
        """
        if h.ndim != 5 or attention.ndim != 3:
            raise ValueError(
                "h must be [B,F,C,H,W] and "
                "attention must be [B,F,F]"
            )

        if h.shape[0] != attention.shape[0]:
            raise ValueError("batch size mismatch")

        if h.shape[1] != attention.shape[-1]:
            raise ValueError(
                "source-field dimension mismatch"
            )

        return torch.einsum(
            "bij,bjchw->bichw",
            attention,
            h,
        )

    def _prepare_attention_override(
        self,
        attention: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Validate externally supplied attention.

        This interface is reserved for:
        - normal evaluation
        - shuffled-attention evaluation
        - average-attention evaluation
        """
        attention = attention.to(
            dtype=dtype,
            device=device,
        )

        if attention.ndim == 2:
            attention = attention.unsqueeze(0)

        if (
            attention.ndim != 3
            or tuple(attention.shape[-2:])
            != (self.num_fields, self.num_fields)
        ):
            raise ValueError(
                "attention_override must be [F,F] "
                "or [B,F,F], "
                f"got {tuple(attention.shape)}"
            )

        if attention.shape[0] == 1 and batch_size > 1:
            attention = attention.expand(
                batch_size,
                -1,
                -1,
            )
        elif attention.shape[0] != batch_size:
            raise ValueError(
                "attention_override batch="
                f"{attention.shape[0]}, "
                f"expected {batch_size}"
            )

        if not torch.isfinite(attention).all():
            raise ValueError(
                "attention_override contains NaN/Inf"
            )

        atol = (
            5e-3
            if dtype in (
                torch.float16,
                torch.bfloat16,
            )
            else 1e-5
        )

        diagonal = attention.diagonal(
            dim1=-2,
            dim2=-1,
        )

        if diagonal.abs().max().item() > atol:
            raise ValueError(
                "attention_override diagonal must be zero"
            )

        row_error = (
            attention.sum(dim=-1) - 1.0
        ).abs().max().item()

        if row_error > atol:
            raise ValueError(
                "attention_override rows must sum to 1, "
                f"max error={row_error}"
            )

        if attention.min().item() < -atol:
            raise ValueError(
                "attention_override must be non-negative"
            )

        return attention

    def _post_project(
        self,
        mixed: torch.Tensor,
    ) -> torch.Tensor:
        bsz, fields, hidden, height, width = mixed.shape

        if (
            fields != self.num_fields
            or hidden != self.hidden_channels
        ):
            raise ValueError(
                "mixed feature shape does not match "
                "configured fields/channels"
            )

        update = mixed.reshape(
            bsz * fields,
            hidden,
            height,
            width,
        )
        update = self.post_proj(update)
        update = self.dropout(update)

        return update.reshape(
            bsz,
            fields,
            self.channels,
            height,
            width,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
        attention_override: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        bsz, fields, _, _, _ = self._check_x(x)

        h = self.project_fields(x)

        scores = self.compute_scores(h)
        computed_attention = self.scores_to_attention(
            scores
        )

        if (
            computed_attention.shape[0] == 1
            and bsz > 1
        ):
            computed_attention = computed_attention.expand(
                bsz,
                -1,
                -1,
            )

        if attention_override is None:
            attention = computed_attention
        else:
            attention = self._prepare_attention_override(
                attention=attention_override,
                batch_size=bsz,
                dtype=h.dtype,
                device=h.device,
            )

        attention = attention.to(
            dtype=h.dtype,
            device=h.device,
        )

        mixed = self.mix_values(
            h,
            attention,
        )

        projected_update = self._post_project(
            mixed
        )

        gate = self.gate_values().to(
            dtype=x.dtype,
            device=x.device,
        )
        gate = gate.view(
            1,
            fields,
            1,
            1,
            1,
        )

        scale = self.attention_output_scale.to(
            dtype=x.dtype,
            device=x.device,
        )

        branch_update = (
            gate
            * scale
            * projected_update
        )

        out = x + branch_update

        if not return_diagnostics:
            return out

        return {
            "out": out,
            "h": h,
            "scores": scores,
            "computed_attention": computed_attention,
            "attention": attention,
            "mixed": mixed,
            "projected_update": projected_update,
            "branch_update": branch_update,
            "gate": gate.reshape(fields),
            "attention_output_scale": scale.reshape(()),
        }


class StaticSoftmaxFieldAttentionBlock(
    _BaseSoftmaxFieldAttentionBlock
):
    """
    M9-0a control experiment.

    The raw parameter is [F,F].

    Four diagonal entries are retained for simple visualization,
    but are permanently masked and receive zero gradient.
    """

    def __init__(
        self,
        channels: int,
        num_fields: int = 4,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
        init_gate: float = -4.0,
        use_norm: bool = True,
        attention_output_scale: float = 0.06,
        score_temperature: float = 1.0,
    ) -> None:
        super().__init__(
            channels=channels,
            num_fields=num_fields,
            hidden_channels=hidden_channels,
            dropout=dropout,
            init_gate=init_gate,
            use_norm=use_norm,
            attention_output_scale=attention_output_scale,
            score_temperature=score_temperature,
        )

        self.static_attention_logits = nn.Parameter(
            torch.zeros(
                num_fields,
                num_fields,
                dtype=torch.float32,
            )
        )

    def compute_scores(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        if h.ndim != 5:
            raise ValueError(
                f"h must be [B,F,C,H,W], "
                f"got {tuple(h.shape)}"
            )

        return self.static_attention_logits.unsqueeze(
            0
        ).expand(
            h.shape[0],
            -1,
            -1,
        )

    def effective_attention_parameter_count(
        self,
    ) -> int:
        """Return the 12 effective off-diagonal logits."""
        return (
            self.num_fields
            * (self.num_fields - 1)
        )


class StateDependentFieldAttentionBlock(
    _BaseSoftmaxFieldAttentionBlock
):
    """
    M9-0b candidate.

    Full spatial Q/K feature maps participate in each score, but each
    sample still receives one global [F,F] attention matrix.

    This is not local position-dependent attention.
    """

    VALID_SCORE_SCALE_MODES = {
        "spatial_mean",
        "sqrt_token_dim",
    }

    def __init__(
        self,
        channels: int,
        num_fields: int = 4,
        hidden_channels: int | None = None,
        qk_channels: int | None = None,
        dropout: float = 0.0,
        init_gate: float = -4.0,
        use_norm: bool = True,
        attention_output_scale: float = 0.06,
        score_scale_mode: str = "spatial_mean",
        score_temperature: float = 1.0,
        qk_init_std: float = 0.02,
    ) -> None:
        super().__init__(
            channels=channels,
            num_fields=num_fields,
            hidden_channels=hidden_channels,
            dropout=dropout,
            init_gate=init_gate,
            use_norm=use_norm,
            attention_output_scale=attention_output_scale,
            score_temperature=score_temperature,
        )

        if (
            score_scale_mode
            not in self.VALID_SCORE_SCALE_MODES
        ):
            raise ValueError(
                "score_scale_mode must be one of "
                f"{sorted(self.VALID_SCORE_SCALE_MODES)}, "
                f"got {score_scale_mode!r}"
            )

        if (
            qk_channels is not None
            and qk_channels <= 0
        ):
            raise ValueError(
                "qk_channels must be positive or None, "
                f"got {qk_channels}"
            )

        if qk_init_std <= 0.0:
            raise ValueError(
                "qk_init_std must be positive, "
                f"got {qk_init_std}"
            )

        self.qk_channels = (
            qk_channels
            or self.hidden_channels
        )
        self.score_scale_mode = score_scale_mode
        self.qk_init_std = float(qk_init_std)

        # Q/K projection is shared across all four fields.
        self.q_proj = nn.Conv2d(
            self.hidden_channels,
            self.qk_channels,
            kernel_size=1,
        )
        self.k_proj = nn.Conv2d(
            self.hidden_channels,
            self.qk_channels,
            kernel_size=1,
        )

        self.reset_qk_parameters()

    def reset_qk_parameters(self) -> None:
        nn.init.normal_(
            self.q_proj.weight,
            mean=0.0,
            std=self.qk_init_std,
        )
        nn.init.normal_(
            self.k_proj.weight,
            mean=0.0,
            std=self.qk_init_std,
        )

        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)

    def _project_qk(
        self,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 5:
            raise ValueError(
                f"h must be [B,F,C,H,W], "
                f"got {tuple(h.shape)}"
            )

        bsz, fields, hidden, height, width = h.shape

        if (
            fields != self.num_fields
            or hidden != self.hidden_channels
        ):
            raise ValueError(
                "h shape does not match configured "
                "fields/channels"
            )

        h_flat = h.reshape(
            bsz * fields,
            hidden,
            height,
            width,
        )

        q = self.q_proj(h_flat).reshape(
            bsz,
            fields,
            self.qk_channels,
            height,
            width,
        )

        k = self.k_proj(h_flat).reshape(
            bsz,
            fields,
            self.qk_channels,
            height,
            width,
        )

        return q, k

    def _score_denominator(
        self,
        height: int,
        width: int,
    ) -> float:
        if self.score_scale_mode == "spatial_mean":
            return (
                float(height * width)
                * math.sqrt(
                    float(self.qk_channels)
                )
            )

        if self.score_scale_mode == "sqrt_token_dim":
            return math.sqrt(
                float(
                    self.qk_channels
                    * height
                    * width
                )
            )

        raise RuntimeError(
            "unsupported score_scale_mode="
            f"{self.score_scale_mode!r}"
        )

    def compute_scores(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        q, k = self._project_qk(h)

        height, width = q.shape[-2:]

        # scores[b, i, j]:
        # target field i compares with source field j.
        scores = torch.einsum(
            "bichw,bjchw->bij",
            q,
            k,
        )

        return (
            scores
            / self._score_denominator(
                height,
                width,
            )
        )


if __name__ == "__main__":
    torch.manual_seed(20260717)

    x = torch.randn(
        2,
        4,
        8,
        16,
        64,
    )

    static_block = StaticSoftmaxFieldAttentionBlock(
        channels=8,
        num_fields=4,
        hidden_channels=8,
        attention_output_scale=0.06,
    )

    dynamic_mean = StateDependentFieldAttentionBlock(
        channels=8,
        num_fields=4,
        hidden_channels=8,
        qk_channels=8,
        attention_output_scale=0.06,
        score_scale_mode="spatial_mean",
    )

    dynamic_sqrt = StateDependentFieldAttentionBlock(
        channels=8,
        num_fields=4,
        hidden_channels=8,
        qk_channels=8,
        attention_output_scale=0.06,
        score_scale_mode="sqrt_token_dim",
    )

    for name, block in (
        ("static", static_block),
        ("dynamic_spatial_mean", dynamic_mean),
        ("dynamic_sqrt_token_dim", dynamic_sqrt),
    ):
        info = block(
            x,
            return_diagnostics=True,
        )

        attention = info["attention"]

        diagonal = attention.diagonal(
            dim1=-2,
            dim2=-1,
        )

        row_error = (
            attention.sum(dim=-1) - 1.0
        ).abs().max()

        print(f"\n[{name}]")
        print(
            "input shape:     ",
            tuple(x.shape),
        )
        print(
            "output shape:    ",
            tuple(info["out"].shape),
        )
        print(
            "scores shape:    ",
            tuple(info["scores"].shape),
        )
        print(
            "attention shape: ",
            tuple(attention.shape),
        )
        print(
            "diag max abs:    ",
            float(
                diagonal.abs().max().item()
            ),
        )
        print(
            "row-sum max err: ",
            float(row_error.item()),
        )
        print(
            "scores std:      ",
            float(
                info["scores"].std().item()
            ),
        )
        print(
            "attention sample std:",
            float(
                attention.std(
                    dim=0,
                    unbiased=False,
                ).mean().item()
            ),
        )
        print(
            "branch relative norm:",
            float(
                info["branch_update"].norm()
                / (x.norm() + 1e-12)
            ),
        )

        assert info["out"].shape == x.shape

        assert torch.equal(
            diagonal,
            torch.zeros_like(diagonal),
        )

        assert torch.allclose(
            attention.sum(dim=-1),
            torch.ones_like(
                attention.sum(dim=-1)
            ),
            atol=1e-6,
            rtol=1e-6,
        )

        assert torch.isfinite(
            info["out"]
        ).all()

        assert torch.isfinite(
            attention
        ).all()

    print(
        "\n✅ field_attention.py "
        "basic smoke test passed."
    )
    print(
        "⚠️ λ=0.06 and score_scale_mode "
        "are still diagnostic candidates."
    )
