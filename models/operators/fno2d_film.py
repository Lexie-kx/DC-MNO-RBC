import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.operators.fno2d import SpectralConv2d
from models.blocks.film import ParamFiLM, apply_film


class FiLMFNO2d(nn.Module):
    """
    M3-Delta + FiLM parameter-conditioned FNO.

    输入:
        x:     [B, 16, H, W]
        param: [B, 2] = [log10(Ra), log10(Pr)]

    输出:
        pred_delta_norm: [B, 4, H, W]
    """

    def __init__(
        self,
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        film_hidden_dim=64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.p = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.film = ParamFiLM(
            param_dim=4,
            hidden_dim=film_hidden_dim,
            width=self.width,
            num_layers=4,
        )

        self.mlp0 = nn.Linear(self.width, 128)
        self.mlp1 = nn.Linear(128, out_channels)

    def forward(self, x, param):
        if x.ndim != 4:
            raise ValueError(f"x 应为 [B, C, H, W]，但得到 {x.shape}")

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"x 输入通道应为 {self.in_channels}，但得到 {x.shape[1]}"
            )

        if param.ndim != 2 or param.shape[1] != 2:
            raise ValueError(
                f"param 应为 [B, 2] = [log10(Ra), log10(Pr)]，但得到 {param.shape}"
            )

        param = param.to(device=x.device, dtype=x.dtype)

        gammas, betas = self.film(param)

        x = x.permute(0, 2, 3, 1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = apply_film(x, gammas[:, 0, :], betas[:, 0, :])
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = apply_film(x, gammas[:, 1, :], betas[:, 1, :])
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = apply_film(x, gammas[:, 2, :], betas[:, 2, :])
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = apply_film(x, gammas[:, 3, :], betas[:, 3, :])

        x = x.permute(0, 2, 3, 1)
        x = self.mlp0(x)
        x = F.gelu(x)
        x = self.mlp1(x)
        x = x.permute(0, 3, 1, 2)

        return x


if __name__ == "__main__":
    B = 4
    C = 16
    H = 256
    W = 64

    dummy_x = torch.randn(B, C, H, W)
    dummy_param = torch.tensor(
        [
            [6.0, -0.3010],
            [6.0, 0.0],
            [7.0, -0.3010],
            [7.0, 0.0],
        ],
        dtype=torch.float32
    )

    model = FiLMFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        film_hidden_dim=64,
    )

    pred = model(dummy_x, dummy_param)

    print(f"input x shape: {dummy_x.shape}")
    print(f"param shape:   {dummy_param.shape}")
    print(f"output shape:  {pred.shape}")

    assert pred.shape == (B, 4, H, W)
    print("✅ FiLMFNO2d forward shape test passed.")
