import torch
import torch.nn as nn
import torch.nn.functional as F

from models.operators.fno2d import SpectralConv2d


class FieldWiseFNO2d(nn.Module):
    """
    M6-FieldWiseEncoder-H4

    正式主实验第一步：
    - 只加入 field-wise encoder
    - 不加入 ParameterToken
    - 不加入 CouplingToken

    输入:
        x: [B, 16, H, W]
        16 = 4 个历史帧 * 4 个物理场

    注意:
        原始 history 是 [T, C, H, W]，flatten 后是 [T*C, H, W]。
        所以这里必须先还原成 [B, T, C, H, W]，
        不能直接用 x[:, 0:4] 当作 buoyancy。
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
    ):
        super().__init__()

        if in_channels != context_length * num_fields:
            raise ValueError(
                f"in_channels={in_channels} 必须等于 "
                f"context_length*num_fields={context_length * num_fields}"
            )

        if field_width is None:
            if width % num_fields != 0:
                raise ValueError(
                    f"width={width} 不能被 num_fields={num_fields} 整除，"
                    "请显式设置 field_width。"
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

        # 每个物理场单独编码：
        # buoyancy / u_x / u_y / pressure
        # 每个场输入: [B, context_length, H, W]
        # 每个场输出: [B, field_width, H, W]
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

        # 四个场的特征融合到 FNO hidden width
        self.fusion = nn.Conv2d(fused_width, width, kernel_size=1)

        # FNO backbone，保持和 PlainFNO2d 一样的深度
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
        x: [B, 16, H, W]

        return:
            fused: [B, width, H, W]
            field_features: list，长度为 4，每个元素 [B, field_width, H, W]
        """
        batch_size, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"输入通道数不对：got {channels}, expected {self.in_channels}"
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
            field_feat = encoder(x_field)
            field_features.append(field_feat)

        # [B, field_width*4, H, W]
        fused = torch.cat(field_features, dim=1)
        fused = self.fusion(fused)

        return fused, field_features

    def forward(self, x, return_features=False):
        x, field_features = self.encode_fields(x)

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
            }

        return x


M6FieldWiseEncoderFNO2d = FieldWiseFNO2d


if __name__ == "__main__":
    batch_size = 2
    height = 64
    width = 256

    x = torch.randn(batch_size, 16, height, width)

    model = FieldWiseFNO2d(
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

    assert y.shape == (batch_size, 4, height, width)
    assert len(info["field_features"]) == 4
    assert info["field_features"][0].shape == (batch_size, 8, height, width)

    print("✅ FieldWiseFNO2d smoke test passed.")
