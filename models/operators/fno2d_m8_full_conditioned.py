"""
M8 Full Static / Parameter-Conditioned Coupling FNO.

Two controlled modes share exactly the same model structure:

1. static
   FieldWiseEncoder
   + static FieldCoupling
   + ParameterToken
   + FNO backbone

2. parameter_conditioned
   FieldWiseEncoder
   + parameter-conditioned FieldCoupling
   + ParameterToken
   + FNO backbone

Both modes retain:
- Delta prediction
- H4 multi-step training protocol
- No PDE loss at this stage
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from models.operators.fno2d import SpectralConv2d
from models.operators.fno2d_paramtoken import (
    ParamTokenEmbedding,
    apply_param_token,
)
from models.blocks.param_conditioned_field_coupling import (
    ParamConditionedFieldCouplingBlock,
)


class M8FullConditionedFNO2d(nn.Module):
    """
    Shared implementation for:

    M8-A FullStatic-H4
        coupling_mode="static"

    M8-B ParamConditionedCoupling-H4
        coupling_mode="parameter_conditioned"

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

    VALID_COUPLING_MODES = {
        "static",
        "parameter_conditioned",
    }

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
        coupling_mode: str = "static",
        coupling_hidden_channels: int | None = None,
        coupling_dropout: float = 0.0,
        coupling_init_gate: float = -4.0,
        coupling_use_norm: bool = True,
        coupling_param_hidden_dim: int = 64,
        coupling_condition_scale: float = 0.10,
        token_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        if in_channels != context_length * num_fields:
            raise ValueError(
                f"in_channels={in_channels} must equal "
                f"context_length*num_fields="
                f"{context_length * num_fields}"
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.context_length = context_length
        self.num_fields = num_fields
        self.field_width = field_width
        self.coupling_mode = coupling_mode
        self.token_hidden_dim = token_hidden_dim

        fused_width = field_width * num_fields

        # ---------------------------------------------------------
        # M6 FieldWiseEncoder backbone
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Shared M8 field coupling block
        #
        # static:
        #   fixed trainable coupling matrix
        #
        # parameter_conditioned:
        #   base matrix + Ra/Pr-conditioned correction
        # ---------------------------------------------------------
        self.field_coupling = (
            ParamConditionedFieldCouplingBlock(
                channels=field_width,
                num_fields=num_fields,
                hidden_channels=(
                    coupling_hidden_channels or field_width
                ),
                dropout=coupling_dropout,
                init_gate=coupling_init_gate,
                use_norm=coupling_use_norm,
                param_hidden_dim=coupling_param_hidden_dim,
                condition_scale=coupling_condition_scale,
            )
        )

        self.fusion = nn.Conv2d(
            fused_width,
            width,
            kernel_size=1,
        )

        # ---------------------------------------------------------
        # Shared FNO backbone
        # ---------------------------------------------------------
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

        self.w0 = nn.Conv2d(width, width, kernel_size=1)
        self.w1 = nn.Conv2d(width, width, kernel_size=1)
        self.w2 = nn.Conv2d(width, width, kernel_size=1)
        self.w3 = nn.Conv2d(width, width, kernel_size=1)

        # Existing M7 ParameterToken remains unchanged.
        self.param_token = ParamTokenEmbedding(
            param_dim=4,
            hidden_dim=token_hidden_dim,
            width=width,
            num_layers=4,
        )

        self.mlp0 = nn.Linear(width, 128)
        self.mlp1 = nn.Linear(128, out_channels)

    def set_coupling_mode(self, coupling_mode: str) -> None:
        """Change coupling mode without changing model weights."""
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
    ):
        """
        Returns:
            fused:
                [B, width, H, W]

            field_features:
                list of F tensors [B, field_width, H, W]

            coupled_field_features:
                list of F tensors [B, field_width, H, W]

            coupling_info:
                coupling diagnostics
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

        # [B, T*C, H, W] -> [B, T, C, H, W]
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
            # [B, T, H, W]
            x_field = x_hist[:, :, field_idx, :, :]

            # [B, field_width, H, W]
            field_feature = encoder(x_field)
            field_features.append(field_feature)

        # [B, F, field_width, H, W]
        field_stack = torch.stack(
            field_features,
            dim=1,
        )

        coupled_stack, coupling_info = self.field_coupling(
            field_stack,
            param=param,
            mode=coupling_mode,
            return_diagnostics=True,
        )

        coupled_field_features = [
            coupled_stack[:, field_idx, :, :, :]
            for field_idx in range(self.num_fields)
        ]

        # [B, F*field_width, H, W]
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
        )

    def forward(
        self,
        x: torch.Tensor,
        param: torch.Tensor,
        return_features: bool = False,
        coupling_mode: str | None = None,
    ):
        if param.ndim != 2 or param.shape[1] != 2:
            raise ValueError(
                "param must be [B, 2] = "
                "[log10(Ra), log10(Pr)], "
                f"got {tuple(param.shape)}"
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

        tokens = self.param_token(param)

        # FNO block 0
        x = self.conv0(x) + self.w0(x)
        x = apply_param_token(
            x,
            tokens[:, 0, :],
        )
        x = F.gelu(x)

        # FNO block 1
        x = self.conv1(x) + self.w1(x)
        x = apply_param_token(
            x,
            tokens[:, 1, :],
        )
        x = F.gelu(x)

        # FNO block 2
        x = self.conv2(x) + self.w2(x)
        x = apply_param_token(
            x,
            tokens[:, 2, :],
        )
        x = F.gelu(x)

        # FNO block 3
        x = self.conv3(x) + self.w3(x)
        x = apply_param_token(
            x,
            tokens[:, 3, :],
        )

        # [B, width, H, W] -> [B, H, W, width]
        x = x.permute(0, 2, 3, 1)
        x = self.mlp0(x)
        x = F.gelu(x)
        x = self.mlp1(x)
        x = x.permute(0, 3, 1, 2)

        if return_features:
            return {
                "out": x,
                "coupling_mode": active_mode,
                "field_features": field_features,
                "coupled_field_features": (
                    coupled_field_features
                ),
                "param_tokens": tokens,
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


# Clear aliases for later training/evaluation scripts.
M8FullStaticFNO2d = M8FullConditionedFNO2d
M8ParamConditionedCouplingFNO2d = (
    M8FullConditionedFNO2d
)


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
            [6.0, -0.30103],
            [8.0, 0.30103],
        ],
        dtype=torch.float32,
    )

    model = M8FullConditionedFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=8,
        modes2=8,
        width=32,
        context_length=4,
        num_fields=4,
        coupling_mode="static",
    )

    model.eval()

    with torch.no_grad():
        y_static = model(
            x,
            param,
            coupling_mode="static",
        )

        y_conditioned_zero = model(
            x,
            param,
            coupling_mode="parameter_conditioned",
        )

        zero_alignment_diff = torch.max(
            torch.abs(
                y_static - y_conditioned_zero
            )
        ).item()

        info = model(
            x,
            param,
            return_features=True,
            coupling_mode="parameter_conditioned",
        )

    print("input:", tuple(x.shape))
    print("param:", tuple(param.shape))
    print("static output:", tuple(y_static.shape))
    print(
        "conditioned output:",
        tuple(y_conditioned_zero.shape),
    )
    print(
        "initial static-conditioned max diff:",
        zero_alignment_diff,
    )
    print(
        "initial conditioned delta abs max:",
        info["conditioned_delta_matrix"]
        .abs()
        .max()
        .item(),
    )
    print(
        "param tokens abs max:",
        info["param_tokens"].abs().max().item(),
    )

    assert y_static.shape == (
        batch_size,
        4,
        height,
        spatial_width,
    )
    assert y_conditioned_zero.shape == y_static.shape
    assert len(info["field_features"]) == 4
    assert len(info["coupled_field_features"]) == 4
    assert info["param_tokens"].shape == (
        batch_size,
        4,
        32,
    )
    assert zero_alignment_diff < 1e-7

    # Confirm that the conditional branch can actually affect output
    # after its final layer becomes non-zero.
    with torch.no_grad():
        model.field_coupling.param_conditioner[
            -1
        ].bias.fill_(0.10)

        y_conditioned_active = model(
            x,
            param,
            coupling_mode="parameter_conditioned",
        )

        active_difference = torch.max(
            torch.abs(
                y_static - y_conditioned_active
            )
        ).item()

    print(
        "activated conditioned-static max diff:",
        active_difference,
    )

    assert active_difference > 0.0

    print(
        "✅ M8FullConditionedFNO2d smoke test passed."
    )
