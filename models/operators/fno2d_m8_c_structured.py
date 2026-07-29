"""
M8-C Physics-Inspired Structured Coupling FNO.

Two controlled modes share exactly the same model structure:

1. structured_static
   M8-C0 control experiment.

2. structured_parameter_conditioned
   M8-C1 formal candidate.

Shared structure:
    FieldWiseEncoder
    -> M8-C Structured Coupling Tokens
    -> Fusion
    -> 4-layer FNO backbone
    -> ParameterToken injection after every FNO block
    -> normalized four-field delta prediction

Important scope:
- Delta prediction is retained.
- H4 rollout logic belongs to the training script.
- No PDE term is explicitly computed.
- No PDE loss is implemented here.
- No scaling consistency is implemented here.

Input:
    x:
        [B, 16, H, W]

    param:
        [B, 2] = [log10(Ra), log10(Pr)]

Output:
    normalized delta prediction:
        [B, 4, H, W]

Field order:
    [buoyancy, u_x, u_y, pressure]
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../..",
        )
    )
)


from models.blocks.m8_c_structured_coupling import (
    M8CStructuredCouplingBlock,
)
from models.operators.fno2d import SpectralConv2d
from models.operators.fno2d_paramtoken import (
    ParamTokenEmbedding,
    apply_param_token,
)


class M8CStructuredFNO2d(nn.Module):
    """
    Shared operator implementation for M8-C0 and M8-C1.

    Modes
    -----
    structured_static:
        M8-C0 control experiment.

    structured_parameter_conditioned:
        M8-C1 formal candidate.
    """

    VALID_COUPLING_MODES = {
        "structured_static",
        "structured_parameter_conditioned",
    }

    FIELD_ORDER = (
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
    )

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
        coupling_mode: str = "structured_static",
        coupling_hidden_channels: int | None = None,
        coupling_condition_hidden_dim: int = 32,
        coupling_condition_scale: float = 1.0,
        coupling_init_strength_logit: float = -4.0,
        token_hidden_dim: int = 64,
        alpha_token: float = 1.0,
    ) -> None:
        super().__init__()

        if in_channels != context_length * num_fields:
            raise ValueError(
                f"in_channels={in_channels} must equal "
                "context_length*num_fields="
                f"{context_length * num_fields}"
            )

        if out_channels != num_fields:
            raise ValueError(
                f"out_channels={out_channels} must equal "
                f"num_fields={num_fields}"
            )

        if num_fields != 4:
            raise ValueError(
                "M8-C v1 requires exactly four fields "
                "[buoyancy, u_x, u_y, pressure], "
                f"got num_fields={num_fields}"
            )

        if coupling_mode not in self.VALID_COUPLING_MODES:
            raise ValueError(
                f"Unknown coupling_mode={coupling_mode}. "
                f"Expected one of "
                f"{sorted(self.VALID_COUPLING_MODES)}"
            )

        if field_width is None:
            if width % num_fields != 0:
                raise ValueError(
                    f"width={width} cannot be divided by "
                    f"num_fields={num_fields}; "
                    "please set field_width explicitly."
                )

            field_width = width // num_fields

        if field_width <= 0:
            raise ValueError(
                f"field_width must be positive, got {field_width}"
            )

        if alpha_token < 0:
            raise ValueError(
                "alpha_token must be non-negative, "
                f"got {alpha_token}"
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.width = int(width)
        self.context_length = int(context_length)
        self.num_fields = int(num_fields)
        self.field_width = int(field_width)
        self.coupling_mode = coupling_mode
        self.token_hidden_dim = int(token_hidden_dim)
        self.alpha_token = float(alpha_token)

        fused_width = (
            self.field_width
            * self.num_fields
        )

        # ---------------------------------------------------------
        # M6-style FieldWiseEncoder
        #
        # Each physical field receives its own encoder.
        # Input per encoder:
        #     [B, context_length, H, W]
        #
        # Output per encoder:
        #     [B, field_width, H, W]
        # ---------------------------------------------------------
        self.field_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.context_length,
                        self.field_width,
                        kernel_size=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        self.field_width,
                        self.field_width,
                        kernel_size=1,
                    ),
                )
                for _ in range(
                    self.num_fields
                )
            ]
        )

        # ---------------------------------------------------------
        # M8-C structured coupling module
        #
        # C0:
        #     shared structured strengths
        #
        # C1:
        #     the same structured branches
        #     + limited parameter-conditioned corrections
        # ---------------------------------------------------------
        self.field_coupling = (
            M8CStructuredCouplingBlock(
                channels=self.field_width,
                num_fields=self.num_fields,
                hidden_channels=(
                    coupling_hidden_channels
                    or self.field_width
                ),
                condition_hidden_dim=(
                    coupling_condition_hidden_dim
                ),
                condition_scale=(
                    coupling_condition_scale
                ),
                init_strength_logit=(
                    coupling_init_strength_logit
                ),
            )
        )

        # ---------------------------------------------------------
        # Field fusion
        # ---------------------------------------------------------
        self.fusion = nn.Conv2d(
            fused_width,
            self.width,
            kernel_size=1,
        )

        # ---------------------------------------------------------
        # Shared four-layer FNO backbone
        # ---------------------------------------------------------
        self.conv0 = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
        )
        self.conv1 = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
        )
        self.conv2 = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
        )
        self.conv3 = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
        )

        self.w0 = nn.Conv2d(
            self.width,
            self.width,
            kernel_size=1,
        )
        self.w1 = nn.Conv2d(
            self.width,
            self.width,
            kernel_size=1,
        )
        self.w2 = nn.Conv2d(
            self.width,
            self.width,
            kernel_size=1,
        )
        self.w3 = nn.Conv2d(
            self.width,
            self.width,
            kernel_size=1,
        )

        # ---------------------------------------------------------
        # Existing ParameterToken remains unchanged.
        #
        # ParameterToken and structured coupling conditioning
        # deliberately coexist in M8-C1:
        #
        # - ParameterToken conditions the FNO backbone.
        # - Structured corrections condition selected coupling
        #   branch strengths.
        # ---------------------------------------------------------
        self.param_token = ParamTokenEmbedding(
            param_dim=4,
            hidden_dim=self.token_hidden_dim,
            width=self.width,
            num_layers=4,
        )

        # ---------------------------------------------------------
        # Four-field normalized delta output head
        # ---------------------------------------------------------
        self.mlp0 = nn.Linear(
            self.width,
            128,
        )
        self.mlp1 = nn.Linear(
            128,
            self.out_channels,
        )

    def set_coupling_mode(
        self,
        coupling_mode: str,
    ) -> None:
        """
        Change M8-C mode without changing model weights.
        """

        if coupling_mode not in self.VALID_COUPLING_MODES:
            raise ValueError(
                f"Unknown coupling_mode={coupling_mode}. "
                f"Expected one of "
                f"{sorted(self.VALID_COUPLING_MODES)}"
            )

        self.coupling_mode = coupling_mode

    def encode_fields(
        self,
        x: torch.Tensor,
        param: torch.Tensor,
        coupling_mode: str,
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
        Dict[str, object],
    ]:
        """
        Encode and couple the four physical fields.

        Returns
        -------
        fused:
            [B, width, H, W]

        field_features:
            list of four tensors:
            [B, field_width, H, W]

        coupled_field_features:
            list of four tensors:
            [B, field_width, H, W]

        coupling_info:
            structured-coupling diagnostics
        """

        if x.ndim != 4:
            raise ValueError(
                "x must be [B, C, H, W], "
                f"got {tuple(x.shape)}"
            )

        (
            batch_size,
            channels,
            height,
            spatial_width,
        ) = x.shape

        if channels != self.in_channels:
            raise ValueError(
                "Wrong input channels: "
                f"got {channels}, "
                f"expected {self.in_channels}"
            )

        if param.ndim != 2:
            raise ValueError(
                "param must be a rank-2 tensor, "
                f"got {tuple(param.shape)}"
            )

        if param.shape != (
            batch_size,
            2,
        ):
            raise ValueError(
                "param must have shape "
                f"{(batch_size, 2)}, "
                f"got {tuple(param.shape)}"
            )

        # [B, T*F, H, W]
        # ->
        # [B, T, F, H, W]
        x_hist = x.reshape(
            batch_size,
            self.context_length,
            self.num_fields,
            height,
            spatial_width,
        )

        field_features: List[
            torch.Tensor
        ] = []

        for field_idx, encoder in enumerate(
            self.field_encoders
        ):
            # One physical field over all context frames:
            #
            # [B, T, H, W]
            x_field = x_hist[
                :,
                :,
                field_idx,
                :,
                :,
            ]

            # [B, field_width, H, W]
            field_feature = encoder(
                x_field
            )

            field_features.append(
                field_feature
            )

        # [B, F, field_width, H, W]
        field_stack = torch.stack(
            field_features,
            dim=1,
        )

        (
            coupled_stack,
            coupling_info,
        ) = self.field_coupling(
            field_stack,
            param=param,
            mode=coupling_mode,
            return_diagnostics=True,
        )

        coupled_field_features = [
            coupled_stack[
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

        # [B, F*field_width, H, W]
        fused = torch.cat(
            coupled_field_features,
            dim=1,
        )

        # [B, width, H, W]
        fused = self.fusion(
            fused
        )

        return (
            fused,
            field_features,
            coupled_field_features,
            coupling_info,
        )

    def forward(
        self,
        x: torch.Tensor,
        param: torch.Tensor,
        return_features: bool = False,
        coupling_mode: str | None = None,
    ):
        if x.ndim != 4:
            raise ValueError(
                "x must be [B, C, H, W], "
                f"got {tuple(x.shape)}"
            )

        if param.ndim != 2 or param.shape[1] != 2:
            raise ValueError(
                "param must be [B, 2] = "
                "[log10(Ra), log10(Pr)], "
                f"got {tuple(param.shape)}"
            )

        if param.shape[0] != x.shape[0]:
            raise ValueError(
                "x and param batch sizes differ: "
                f"x batch={x.shape[0]}, "
                f"param batch={param.shape[0]}"
            )

        active_mode = (
            self.coupling_mode
            if coupling_mode is None
            else coupling_mode
        )

        if active_mode not in self.VALID_COUPLING_MODES:
            raise ValueError(
                f"Unknown coupling_mode={active_mode}. "
                f"Expected one of "
                f"{sorted(self.VALID_COUPLING_MODES)}"
            )

        param = param.to(
            device=x.device,
            dtype=x.dtype,
        )

        (
            x,
            field_features,
            coupled_field_features,
            coupling_info,
        ) = self.encode_fields(
            x=x,
            param=param,
            coupling_mode=active_mode,
        )

        raw_tokens = self.param_token(
            param
        )

        tokens = (
            self.alpha_token
            * raw_tokens
        )

        # FNO block 0
        x = (
            self.conv0(x)
            + self.w0(x)
        )
        x = apply_param_token(
            x,
            tokens[:, 0, :],
        )
        x = F.gelu(x)

        # FNO block 1
        x = (
            self.conv1(x)
            + self.w1(x)
        )
        x = apply_param_token(
            x,
            tokens[:, 1, :],
        )
        x = F.gelu(x)

        # FNO block 2
        x = (
            self.conv2(x)
            + self.w2(x)
        )
        x = apply_param_token(
            x,
            tokens[:, 2, :],
        )
        x = F.gelu(x)

        # FNO block 3
        x = (
            self.conv3(x)
            + self.w3(x)
        )
        x = apply_param_token(
            x,
            tokens[:, 3, :],
        )

        # [B, width, H, W]
        # ->
        # [B, H, W, width]
        x = x.permute(
            0,
            2,
            3,
            1,
        )

        x = self.mlp0(x)
        x = F.gelu(x)
        x = self.mlp1(x)

        # [B, H, W, 4]
        # ->
        # [B, 4, H, W]
        x = x.permute(
            0,
            3,
            1,
            2,
        )

        if not return_features:
            return x

        return {
            "out": x,
            "coupling_mode": active_mode,
            "field_order": self.FIELD_ORDER,
            "field_features": field_features,
            "coupled_field_features": (
                coupled_field_features
            ),
            "raw_param_tokens": raw_tokens,
            "param_tokens": tokens,
            "alpha_token": self.alpha_token,
            "coupling_info": coupling_info,
            "coupling_branch_names": (
                coupling_info[
                    "branch_names"
                ]
            ),
            "coupling_base_strengths": (
                coupling_info[
                    "base_strengths"
                ]
            ),
            "coupling_effective_strengths": (
                coupling_info[
                    "effective_strengths"
                ]
            ),
            "coupling_conditioned_delta_logits": (
                coupling_info[
                    "conditioned_delta_logits"
                ]
            ),
            "coupling_weighted_branch_rms": (
                coupling_info[
                    "weighted_branch_rms"
                ]
            ),
            "coupling_total_contribution_rms": (
                coupling_info[
                    "total_contribution_rms"
                ]
            ),
        }


# Clear aliases for later training/evaluation scripts.
M8CStructuredStaticFNO2d = (
    M8CStructuredFNO2d
)

M8CStructuredParamFNO2d = (
    M8CStructuredFNO2d
)


if __name__ == "__main__":
    torch.manual_seed(7)

    batch_size = 3
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
            [6.0, -0.30103],
            [7.0, 0.0],
            [8.0, 0.30103],
        ],
        dtype=torch.float32,
    )

    model = M8CStructuredFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=8,
        modes2=8,
        width=32,
        context_length=4,
        num_fields=4,
        field_width=8,
        coupling_mode="structured_static",
        coupling_hidden_channels=8,
        coupling_condition_hidden_dim=32,
        coupling_condition_scale=1.0,
        coupling_init_strength_logit=-4.0,
        token_hidden_dim=64,
        alpha_token=1.0,
    )

    model.eval()

    with torch.no_grad():
        y_static = model(
            x,
            param,
            coupling_mode=(
                "structured_static"
            ),
        )

        y_conditioned_zero = model(
            x,
            param,
            coupling_mode=(
                "structured_parameter_conditioned"
            ),
        )

        static_info = model(
            x,
            param,
            return_features=True,
            coupling_mode=(
                "structured_static"
            ),
        )

        conditioned_zero_info = model(
            x,
            param,
            return_features=True,
            coupling_mode=(
                "structured_parameter_conditioned"
            ),
        )

    initial_max_diff = torch.max(
        torch.abs(
            y_static
            - y_conditioned_zero
        )
    ).item()

    initial_mean_diff = torch.mean(
        torch.abs(
            y_static
            - y_conditioned_zero
        )
    ).item()

    initial_delta_abs_max = (
        conditioned_zero_info[
            "coupling_conditioned_delta_logits"
        ]
        .abs()
        .max()
        .item()
    )

    print(
        "input shape:",
        tuple(x.shape),
    )
    print(
        "param shape:",
        tuple(param.shape),
    )
    print(
        "static output shape:",
        tuple(y_static.shape),
    )
    print(
        "conditioned output shape:",
        tuple(
            y_conditioned_zero.shape
        ),
    )
    print(
        "branch names:",
        static_info[
            "coupling_branch_names"
        ],
    )
    print(
        "base strengths:",
        static_info[
            "coupling_base_strengths"
        ].detach().cpu().tolist(),
    )
    print(
        "effective strengths shape:",
        tuple(
            static_info[
                "coupling_effective_strengths"
            ].shape
        ),
    )
    print(
        "weighted branch RMS shape:",
        tuple(
            static_info[
                "coupling_weighted_branch_rms"
            ].shape
        ),
    )
    print(
        "param tokens shape:",
        tuple(
            static_info[
                "param_tokens"
            ].shape
        ),
    )
    print(
        "initial static-conditioned max diff:",
        f"{initial_max_diff:.12e}",
    )
    print(
        "initial static-conditioned mean diff:",
        f"{initial_mean_diff:.12e}",
    )
    print(
        "initial conditioned delta abs max:",
        f"{initial_delta_abs_max:.12e}",
    )

    assert y_static.shape == (
        batch_size,
        4,
        height,
        spatial_width,
    )

    assert (
        y_conditioned_zero.shape
        == y_static.shape
    )

    assert len(
        static_info[
            "field_features"
        ]
    ) == 4

    assert len(
        static_info[
            "coupled_field_features"
        ]
    ) == 4

    assert static_info[
        "param_tokens"
    ].shape == (
        batch_size,
        4,
        32,
    )

    assert static_info[
        "coupling_effective_strengths"
    ].shape == (
        batch_size,
        5,
    )

    assert static_info[
        "coupling_weighted_branch_rms"
    ].shape == (
        batch_size,
        5,
    )

    assert initial_max_diff < 1e-7
    assert initial_mean_diff < 1e-7
    assert initial_delta_abs_max < 1e-7

    # Confirm that the limited C1 conditioned path can
    # actually affect the complete operator output.
    with torch.no_grad():
        model.field_coupling.buoyancy_conditioner[
            -1
        ].bias.fill_(0.50)

        y_conditioned_active = model(
            x,
            param,
            coupling_mode=(
                "structured_parameter_conditioned"
            ),
        )

        active_info = model(
            x,
            param,
            return_features=True,
            coupling_mode=(
                "structured_parameter_conditioned"
            ),
        )

    active_max_diff = torch.max(
        torch.abs(
            y_static
            - y_conditioned_active
        )
    ).item()

    print(
        "activated conditioned-static max diff:",
        f"{active_max_diff:.12e}",
    )

    print(
        "activated conditioned delta logits:",
        active_info[
            "coupling_conditioned_delta_logits"
        ].detach().cpu().tolist(),
    )

    assert active_max_diff > 0.0

    active_delta = active_info[
        "coupling_conditioned_delta_logits"
    ]

    # Advection and pressure remain shared in v1.
    assert torch.max(
        torch.abs(
            active_delta[:, 1]
        )
    ).item() < 1e-7

    assert torch.max(
        torch.abs(
            active_delta[:, 2]
        )
    ).item() < 1e-7

    print(
        "✅ M8CStructuredFNO2d smoke test passed."
    )
