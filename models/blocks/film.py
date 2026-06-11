import torch
import torch.nn as nn


class ParamFiLM(nn.Module):
    """
    Parameter-conditioned FiLM module.

    输入:
        param: [B, 2] = [log10(Ra), log10(Pr)]

    内部扩展为:
        [log10(Ra), log10(Pr), log10(nu), log10(kappa)]

    其中:
        log10(nu)    = -0.5 * (log10(Ra) - log10(Pr))
        log10(kappa) = -0.5 * (log10(Ra) + log10(Pr))

    输出:
        gammas: [B, num_layers, width]
        betas:  [B, num_layers, width]
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
            nn.Linear(hidden_dim, num_layers * width * 2),
        )

        # 让 FiLM 初始接近恒等映射：
        # h * (1 + 0) + 0 = h
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
        out = out.view(B, self.num_layers, 2, self.width)

        gammas = out[:, :, 0, :]
        betas = out[:, :, 1, :]

        return gammas, betas


def apply_film(x, gamma, beta):
    """
    x:     [B, C, H, W]
    gamma: [B, C]
    beta:  [B, C]

    return:
        x * (1 + gamma) + beta
    """

    if x.ndim != 4:
        raise ValueError(f"x 应为 [B, C, H, W]，但得到 {x.shape}")

    B, C, H, W = x.shape

    if gamma.shape != (B, C):
        raise ValueError(f"gamma shape 应为 {(B, C)}，但得到 {gamma.shape}")

    if beta.shape != (B, C):
        raise ValueError(f"beta shape 应为 {(B, C)}，但得到 {beta.shape}")

    gamma = gamma.view(B, C, 1, 1)
    beta = beta.view(B, C, 1, 1)

    return x * (1.0 + gamma) + beta
