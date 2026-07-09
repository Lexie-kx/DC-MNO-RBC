import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.operators.fno2d import SpectralConv2d
from models.operators.fno2d_paramtoken import ParamTokenEmbedding, apply_param_token


class M7FieldWiseParamTokenFNO2d(nn.Module):
    """
    M7-ParamTokenOnly-H4

    Controlled upgrade from M6-FieldWiseEncoder-H4:
    - Keep field-wise encoders
    - Add ParameterToken after each FNO block
    - No FieldCoupling
    - No CouplingToken
    - No PDE loss

    Input:
        x:     [B, 16, H, W]
        param: [B, 2] = [log10(Ra), log10(Pr)]

    Output:
        pred_delta_norm: [B, 4, H, W]

    Field order:
        [buoyancy, u_x, u_y, pressure]
    """

    def __init__(
        self,
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        context_length=4,
        num_fields=4,
        field_width=None,
        token_hidden_dim=64,
    ):
        super().__init__()

        if in_channels != context_length * num_fields:
            raise ValueError(
                f"in_channels={in_channels} must equal "
                f"context_length*num_fields={context_length * num_fields}"
            )

        if field_width is None:
            if width % num_fields != 0:
                raise ValueError(
                    f"width={width} cannot be divided by num_fields={num_fields}; "
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
        self.token_hidden_dim = token_hidden_dim

        fused_width = field_width * num_fields

        # Same field-wise encoders as M6 FieldWiseEncoder.
        self.field_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(context_length, field_width, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(field_width, field_width, kernel_size=1),
                )
                for _ in range(num_fields)
            ]
        )

        # Same fusion layer as M6 FieldWiseEncoder.
        self.fusion = nn.Conv2d(fused_width, width, kernel_size=1)

        # Same FNO backbone as M6 FieldWiseEncoder.
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        # New M7-ParamTokenOnly component.
        # Zero initialized inside ParamTokenEmbedding, so initial behavior
        # is close to M6 FieldWiseEncoder when loading M6 weights.
        self.param_token = ParamTokenEmbedding(
            param_dim=4,
            hidden_dim=token_hidden_dim,
            width=width,
            num_layers=4,
        )

        self.mlp0 = nn.Linear(width, 128)
        self.mlp1 = nn.Linear(128, out_channels)

    def encode_fields(self, x):
        """
        Args:
            x: [B, 16, H, W]

        Returns:
            fused: [B, width, H, W]
            field_features: list of 4 tensors, each [B, field_width, H, W]
        """
        batch_size, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Wrong input channels: got {channels}, expected {self.in_channels}"
            )

        # [B, T*C, H, W] -> [B, T, C, H, W]
        x_hist = x.reshape(
            batch_size,
            self.context_length,
            self.num_fields,
            height,
            width,
        )

        field_features = []

        for field_idx, encoder in enumerate(self.field_encoders):
            # [B, T, H, W]
            x_field = x_hist[:, :, field_idx, :, :]
            # [B, field_width, H, W]
            field_feat = encoder(x_field)
            field_features.append(field_feat)

        # [B, field_width*4, H, W]
        fused = torch.cat(field_features, dim=1)
        fused = self.fusion(fused)

        return fused, field_features

    def forward(self, x, param, return_features=False):
        if x.ndim != 4:
            raise ValueError(f"x should be [B, C, H, W], got {x.shape}")

        if param.ndim != 2 or param.shape[1] != 2:
            raise ValueError(
                f"param should be [B, 2] = [log10(Ra), log10(Pr)], got {param.shape}"
            )

        param = param.to(device=x.device, dtype=x.dtype)

        x, field_features = self.encode_fields(x)
        tokens = self.param_token(param)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = apply_param_token(x, tokens[:, 0, :])
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = apply_param_token(x, tokens[:, 1, :])
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = apply_param_token(x, tokens[:, 2, :])
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = apply_param_token(x, tokens[:, 3, :])

        # [B, width, H, W] -> [B, H, W, width]
        x = x.permute(0, 2, 3, 1)
        x = self.mlp0(x)
        x = F.gelu(x)
        x = self.mlp1(x)
        x = x.permute(0, 3, 1, 2)

        if return_features:
            return {
                "out": x,
                "field_features": field_features,
                "param_tokens": tokens,
            }

        return x


# Aliases for clearer imports later.
FieldWiseParamTokenFNO2d = M7FieldWiseParamTokenFNO2d
M7ParamTokenOnlyFNO2d = M7FieldWiseParamTokenFNO2d


if __name__ == "__main__":
    batch_size = 2
    height = 64
    width = 256

    x = torch.randn(batch_size, 16, height, width)
    param = torch.tensor(
        [
            [6.0, 0.0],
            [7.0, 0.30103],
        ],
        dtype=torch.float32,
    )

    model = M7FieldWiseParamTokenFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        context_length=4,
        num_fields=4,
    )

    y = model(x, param)
    print("input:", x.shape)
    print("param:", param.shape)
    print("output:", y.shape)

    info = model(x, param, return_features=True)
    print("return_features out:", info["out"].shape)
    print("num field_features:", len(info["field_features"]))
    print("field feature shape:", info["field_features"][0].shape)
    print("param_tokens:", info["param_tokens"].shape)

    assert y.shape == (batch_size, 4, height, width)
    assert len(info["field_features"]) == 4
    assert info["field_features"][0].shape == (batch_size, 8, height, width)
    assert info["param_tokens"].shape == (batch_size, 4, 32)

    print("✅ M7FieldWiseParamTokenFNO2d smoke test passed.")
