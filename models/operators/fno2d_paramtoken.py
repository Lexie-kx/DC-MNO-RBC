import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.operators.fno2d import SpectralConv2d


class ParamTokenEmbedding(nn.Module):
    """
    Parameter Token module.

    输入:
        param: [B, 2] = [log10(Ra), log10(Pr)]

    内部扩展为:
        [log10(Ra), log10(Pr), log10(nu), log10(kappa)]

    其中:
        log10(nu)    = -0.5 * (log10(Ra) - log10(Pr))
        log10(kappa) = -0.5 * (log10(Ra) + log10(Pr))

    输出:
        tokens: [B, num_layers, width]

    说明:
        第一版 ParameterToken 不做 attention。
        它把全局物理参数编码成每一层 FNO hidden feature 的 additive token。
    """

    def __init__(
        self,
        param_dim=4,
        hidden_dim=64,
        width=32,
        num_layers=4,
    ):
        super().__init__()

        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.num_layers = num_layers

        self.net = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers * width),
        )

        # 初始 token 为 0，使模型初始接近普通 FNO：
        # h + 0 = h
        # 这样从 M5-Delta-H4 / M3-Delta 初始化时更稳定。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def expand_param(self, param):
        if param.ndim != 2 or param.shape[1] != 2:
            raise ValueError(
                f"param 应为 [B, 2] = [log10(Ra), log10(Pr)]，但得到 {param.shape}"
            )

        log_ra = param[:, 0]
        log_pr = param[:, 1]

        log_nu = -0.5 * (log_ra - log_pr)
        log_kappa = -0.5 * (log_ra + log_pr)

        param4 = torch.stack(
            [log_ra, log_pr, log_nu, log_kappa],
            dim=1
        )

        return param4

    def forward(self, param):
        param4 = self.expand_param(param)
        out = self.net(param4)

        B = param.shape[0]
        tokens = out.view(B, self.num_layers, self.width)

        return tokens


def apply_param_token(x, token):
    """
    x:     [B, C, H, W]
    token: [B, C]

    return:
        x + token[:, :, None, None]
    """

    if x.ndim != 4:
        raise ValueError(f"x 应为 [B, C, H, W]，但得到 {x.shape}")

    B, C, H, W = x.shape

    if token.shape != (B, C):
        raise ValueError(f"token shape 应为 {(B, C)}，但得到 {token.shape}")

    token = token.view(B, C, 1, 1)

    return x + token


class ParamTokenFNO2d(nn.Module):
    """
    M3-Delta / M5-Delta-H4 + Parameter Token parameter-conditioned FNO.

    输入:
        x:     [B, 16, H, W]
        param: [B, 2] = [log10(Ra), log10(Pr)]

    输出:
        pred_delta_norm: [B, 4, H, W]

    和 FiLMFNO2d 的区别:
        FiLMFNO2d:
            x = gamma(param) * x + beta(param)

        ParamTokenFNO2d:
            x = x + token(param)

    第一版不引入 attention，只验证 parameter token 是否比 shallow FiLM 更稳定。
    """

    def __init__(
        self,
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        token_hidden_dim=64,
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

        self.param_token = ParamTokenEmbedding(
            param_dim=4,
            hidden_dim=token_hidden_dim,
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

        tokens = self.param_token(param)

        x = x.permute(0, 2, 3, 1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

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

    model = ParamTokenFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        token_hidden_dim=64,
    )

    pred = model(dummy_x, dummy_param)

    print(f"input x shape: {dummy_x.shape}")
    print(f"param shape:   {dummy_param.shape}")
    print(f"output shape:  {pred.shape}")

    assert pred.shape == (B, 4, H, W)
    print("✅ ParamTokenFNO2d forward shape test passed.")
