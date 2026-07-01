import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.operators.fno2d import SpectralConv2d
from models.operators.fno2d_paramtoken import ParamTokenEmbedding, apply_param_token


class CouplingTokenEmbedding(nn.Module):
    """
    Coupling Token module.

    输入:
        x: [B, 16, H, W]

    其中 16 = 4 history frames × 4 fields:
        fields = [buoyancy, u_x, u_y, pressure]

    第一版 CouplingToken 不做 attention。
    它从历史场中提取轻量统计特征，表示当前多物理场状态：

        per field:
            mean
            std
            rms

    得到:
        4 fields × 3 stats = 12 features

    输出:
        coupling_tokens: [B, num_layers, width]

    说明:
        最后一层零初始化，使 coupling token 初始为 0。
        因此从 ParamToken checkpoint 初始化时，模型初始接近原 M5-Delta-ParameterToken-H4。
    """

    def __init__(
        self,
        num_fields=4,
        num_stats=3,
        hidden_dim=64,
        width=32,
        num_layers=4,
        eps=1e-6,
    ):
        super().__init__()

        self.num_fields = num_fields
        self.num_stats = num_stats
        self.hidden_dim = hidden_dim
        self.width = width
        self.num_layers = num_layers
        self.eps = eps

        in_dim = num_fields * num_stats

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers * width),
        )

        # 初始 coupling token 为 0，使新模型初始接近 ParamToken 模型。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def extract_field_stats(self, x):
        if x.ndim != 4:
            raise ValueError(f"x 应为 [B, 16, H, W]，但得到 {x.shape}")

        B, C, H, W = x.shape

        if C != 16:
            raise ValueError(
                f"CouplingToken 第一版要求 x 通道数为 16 = 4 frames × 4 fields，但得到 {C}"
            )

        # flatten 顺序来自训练脚本:
        # context_norm: [B, T=4, C=4, H, W]
        # model_input = context.reshape(B, T*C, H, W)
        # 因此这里还原为 [B, T=4, F=4, H, W]
        x_hist = x.reshape(B, 4, 4, H, W)

        # 对 time 和 spatial 维度做统计，保留 field 维度。
        # field_mean/std/rms: [B, 4]
        field_mean = x_hist.mean(dim=(1, 3, 4))
        field_std = x_hist.std(dim=(1, 3, 4), unbiased=False)
        field_rms = torch.sqrt((x_hist ** 2).mean(dim=(1, 3, 4)) + self.eps)

        stats = torch.cat(
            [field_mean, field_std, field_rms],
            dim=1
        )

        return stats

    def forward(self, x):
        stats = self.extract_field_stats(x)
        out = self.net(stats)

        B = x.shape[0]
        tokens = out.view(B, self.num_layers, self.width)

        return tokens


class ParamCouplingTokenFNO2d(nn.Module):
    """
    M6-Delta-ParamToken-CouplingToken-H4 model backbone.

    输入:
        x:     [B, 16, H, W]
        param: [B, 2] = [log10(Ra), log10(Pr)]

    输出:
        pred_delta_norm: [B, 4, H, W]

    相比 ParamTokenFNO2d:
        原来每层注入:
            x = x + param_token[layer]

        现在每层注入:
            x = x + param_token[layer] + coupling_token[layer]

    ParamToken 回答:
        当前是什么 Ra / Pr / nu / kappa 工况。

    CouplingToken 回答:
        当前四个物理场历史状态的统计结构是什么。
    """

    def __init__(
        self,
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        token_hidden_dim=64,
        coupling_hidden_dim=64,
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

        self.coupling_token = CouplingTokenEmbedding(
            num_fields=4,
            num_stats=3,
            hidden_dim=coupling_hidden_dim,
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

        param_tokens = self.param_token(param)
        coupling_tokens = self.coupling_token(x)

        x = x.permute(0, 2, 3, 1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = apply_param_token(x, param_tokens[:, 0, :])
        x = apply_param_token(x, coupling_tokens[:, 0, :])
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = apply_param_token(x, param_tokens[:, 1, :])
        x = apply_param_token(x, coupling_tokens[:, 1, :])
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = apply_param_token(x, param_tokens[:, 2, :])
        x = apply_param_token(x, coupling_tokens[:, 2, :])
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = apply_param_token(x, param_tokens[:, 3, :])
        x = apply_param_token(x, coupling_tokens[:, 3, :])

        x = x.permute(0, 2, 3, 1)
        x = self.mlp0(x)
        x = F.gelu(x)
        x = self.mlp1(x)
        x = x.permute(0, 3, 1, 2)

        return x


# 备用别名，方便后续脚本命名更直观。
M6DeltaParamCouplingTokenFNO2d = ParamCouplingTokenFNO2d


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

    model = ParamCouplingTokenFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        token_hidden_dim=64,
        coupling_hidden_dim=64,
    )

    pred = model(dummy_x, dummy_param)

    print(f"input x shape: {dummy_x.shape}")
    print(f"param shape:   {dummy_param.shape}")
    print(f"output shape:  {pred.shape}")

    assert pred.shape == (B, 4, H, W)

    with torch.no_grad():
        coupling_tokens = model.coupling_token(dummy_x)
        param_tokens = model.param_token(dummy_param)

    print(f"param token shape:    {param_tokens.shape}")
    print(f"coupling token shape: {coupling_tokens.shape}")

    assert param_tokens.shape == (B, 4, 32)
    assert coupling_tokens.shape == (B, 4, 32)

    print("✅ ParamCouplingTokenFNO2d forward shape test passed.")
