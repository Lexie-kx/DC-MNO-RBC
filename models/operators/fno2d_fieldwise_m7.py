import torch
import torch.nn as nn
import torch.nn.functional as F

from models.operators.fno2d import SpectralConv2d
from models.blocks.field_coupling import FieldCouplingBlock


class M7FieldCouplingFNO2d(nn.Module):
    """
    M7-FieldCoupling-H4

    Controlled upgrade from M6-FieldWiseEncoder-H4:
    - Keep field-wise encoders
    - Insert explicit field-to-field coupling before fusion
    - No ParameterToken
    - No CouplingToken
    - No PDE loss

    Input:
        x: [B, 16, H, W]
        16 = context_length(4) * num_fields(4)

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
        coupling_hidden_channels=None,
        coupling_dropout=0.0,
        coupling_init_gate=-4.0,
        coupling_use_norm=True,
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

        # New M7 component:
        # Explicit field-to-field coupling.
        #
        # Input : [B, 4, field_width, H, W]
        # Output: [B, 4, field_width, H, W]
        self.field_coupling = FieldCouplingBlock(
            channels=field_width,
            num_fields=num_fields,
            hidden_channels=coupling_hidden_channels or field_width,
            dropout=coupling_dropout,
            init_gate=coupling_init_gate,
            use_norm=coupling_use_norm,
        )

        # Fuse coupled field features into FNO hidden width.
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

        self.mlp0 = nn.Linear(width, 128)
        self.mlp1 = nn.Linear(128, out_channels)

    def encode_fields(self, x):
        """
        Args:
            x: [B, 16, H, W]

        Returns:
            fused: [B, width, H, W]
            field_features: list of 4 tensors, each [B, field_width, H, W]
            coupled_field_features: list of 4 tensors, each [B, field_width, H, W]
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

        # [B, 4, field_width, H, W]
        field_stack = torch.stack(field_features, dim=1)

        # M7 explicit field-to-field coupling.
        coupled_stack = self.field_coupling(field_stack)

        # Back to list for easier debugging / return_features.
        coupled_field_features = [
            coupled_stack[:, field_idx, :, :, :]
            for field_idx in range(self.num_fields)
        ]

        # [B, field_width*4, H, W]
        fused = torch.cat(coupled_field_features, dim=1)
        fused = self.fusion(fused)

        return fused, field_features, coupled_field_features

    def forward(self, x, return_features=False):
        x, field_features, coupled_field_features = self.encode_fields(x)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

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
                "coupled_field_features": coupled_field_features,
                "coupling_gate": self.field_coupling.gate_values(),
                "coupling_matrix": self.field_coupling.effective_coupling_matrix(),
            }

        return x


# Aliases for clearer imports later.
FieldCouplingFNO2d = M7FieldCouplingFNO2d


if __name__ == "__main__":
    batch_size = 2
    height = 64
    width = 256

    x = torch.randn(batch_size, 16, height, width)

    model = M7FieldCouplingFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        context_length=4,
        num_fields=4,
    )

    y = model(x)
    print("input:", x.shape)
    print("output:", y.shape)

    info = model(x, return_features=True)
    print("return_features out:", info["out"].shape)
    print("num field_features:", len(info["field_features"]))
    print("field feature shape:", info["field_features"][0].shape)
    print("num coupled_field_features:", len(info["coupled_field_features"]))
    print("coupled feature shape:", info["coupled_field_features"][0].shape)
    print("coupling gate:", info["coupling_gate"].detach().cpu().tolist())
    print("coupling matrix:", info["coupling_matrix"].detach().cpu())

    assert y.shape == (batch_size, 4, height, width)
    assert len(info["field_features"]) == 4
    assert len(info["coupled_field_features"]) == 4
    assert info["field_features"][0].shape == (batch_size, 8, height, width)
    assert info["coupled_field_features"][0].shape == (batch_size, 8, height, width)

    print("✅ M7FieldCouplingFNO2d smoke test passed.")
