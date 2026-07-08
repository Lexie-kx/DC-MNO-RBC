import torch
import torch.nn as nn
import torch.nn.functional as F

from models.operators.fno2d_fieldwise import FieldWiseFNO2d


def apply_coupling_token(x, token):
    """
    x:     [B, width, H, W]
    token: [B, width]
    """
    if token.ndim != 2:
        raise ValueError(f"token 应为 [B, width]，但得到 {token.shape}")

    if x.shape[0] != token.shape[0] or x.shape[1] != token.shape[1]:
        raise ValueError(
            f"x/token shape 不匹配: x={x.shape}, token={token.shape}"
        )

    return x + token[:, :, None, None]


class FieldWiseCouplingTokenEmbedding(nn.Module):
    """
    CouplingToken for FieldWiseEncoder.

    正式 M6-CouplingTokenOnly 版本：
    - 不从原始 16 通道 hard concat 输入直接抽统计；
    - 而是从 field-wise encoder 之后的 field_features 抽统计；
    - 因此它真正依赖 FieldWiseEncoder 主干。

    输入:
        field_features: list，长度为 4
        每个元素: [B, field_width, H, W]

    统计特征:
        每个场:
            mean / std / rms
        场间:
            pooled feature 的 pairwise cosine similarity

    输出:
        coupling_tokens: [B, num_layers, width]

    关键设计:
        最后一层零初始化。
        这样从 M6-FieldWiseEncoder-H4 checkpoint 初始化时，
        新模型初始近似等价于 FieldWiseEncoder baseline。
    """

    def __init__(
        self,
        num_fields=4,
        field_width=8,
        hidden_dim=64,
        width=32,
        num_layers=4,
        eps=1e-6,
        use_pairwise=True,
    ):
        super().__init__()

        self.num_fields = num_fields
        self.field_width = field_width
        self.hidden_dim = hidden_dim
        self.width = width
        self.num_layers = num_layers
        self.eps = eps
        self.use_pairwise = use_pairwise

        per_field_stats_dim = num_fields * 3
        pairwise_dim = num_fields * (num_fields - 1) // 2 if use_pairwise else 0
        in_dim = per_field_stats_dim + pairwise_dim

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers * width),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def extract_feature_stats(self, field_features):
        if not isinstance(field_features, (list, tuple)):
            raise ValueError("field_features 应为 list/tuple。")

        if len(field_features) != self.num_fields:
            raise ValueError(
                f"field_features 长度应为 {self.num_fields}，但得到 {len(field_features)}"
            )

        means = []
        stds = []
        rms = []
        pooled = []

        for i, feat in enumerate(field_features):
            if feat.ndim != 4:
                raise ValueError(
                    f"field_features[{i}] 应为 [B, field_width, H, W]，但得到 {feat.shape}"
                )

            if feat.shape[1] != self.field_width:
                raise ValueError(
                    f"field_features[{i}] channel 应为 {self.field_width}，但得到 {feat.shape[1]}"
                )

            means.append(feat.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1))
            stds.append(feat.std(dim=(1, 2, 3), unbiased=False).unsqueeze(1))
            rms.append(torch.sqrt((feat ** 2).mean(dim=(1, 2, 3)) + self.eps).unsqueeze(1))
            pooled.append(feat.mean(dim=(2, 3)))  # [B, field_width]

        stats = torch.cat(means + stds + rms, dim=1)  # [B, 12]

        if self.use_pairwise:
            pair_stats = []
            for i in range(self.num_fields):
                for j in range(i + 1, self.num_fields):
                    sim = F.cosine_similarity(
                        pooled[i],
                        pooled[j],
                        dim=1,
                        eps=self.eps,
                    ).unsqueeze(1)
                    pair_stats.append(sim)

            stats = torch.cat([stats] + pair_stats, dim=1)  # [B, 18]

        return stats

    def forward(self, field_features):
        stats = self.extract_feature_stats(field_features)
        out = self.net(stats)

        batch_size = stats.shape[0]
        tokens = out.view(batch_size, self.num_layers, self.width)

        return tokens


class FieldWiseCouplingTokenFNO2d(FieldWiseFNO2d):
    """
    M6-FieldWise-CouplingToken-H4.

    正式 CouplingToken-only ablation:
    - FieldWiseEncoder
    - CouplingToken
    - 不加 ParameterToken

    对照对象:
        M6-FieldWiseEncoder-H4

    输入:
        x: [B, 16, H, W]

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
        context_length=4,
        num_fields=4,
        field_width=None,
        coupling_hidden_dim=64,
        use_pairwise=True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            context_length=context_length,
            num_fields=num_fields,
            field_width=field_width,
        )

        self.coupling_token = FieldWiseCouplingTokenEmbedding(
            num_fields=self.num_fields,
            field_width=self.field_width,
            hidden_dim=coupling_hidden_dim,
            width=self.width,
            num_layers=4,
            use_pairwise=use_pairwise,
        )

    def forward(self, x, return_features=False):
        x, field_features = self.encode_fields(x)
        coupling_tokens = self.coupling_token(field_features)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = apply_coupling_token(x, coupling_tokens[:, 0, :])
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = apply_coupling_token(x, coupling_tokens[:, 1, :])
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = apply_coupling_token(x, coupling_tokens[:, 2, :])
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = apply_coupling_token(x, coupling_tokens[:, 3, :])

        x = x.permute(0, 2, 3, 1)
        x = self.mlp0(x)
        x = F.gelu(x)
        x = self.mlp1(x)
        x = x.permute(0, 3, 1, 2)

        if return_features:
            return {
                "out": x,
                "field_features": field_features,
                "coupling_tokens": coupling_tokens,
            }

        return x


M6FieldWiseCouplingTokenFNO2d = FieldWiseCouplingTokenFNO2d


if __name__ == "__main__":
    batch_size = 2
    height = 64
    width = 256

    x = torch.randn(batch_size, 16, height, width)

    model = FieldWiseCouplingTokenFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        context_length=4,
        num_fields=4,
    )

    y = model(x)
    info = model(x, return_features=True)

    print("input:", x.shape)
    print("output:", y.shape)
    print("num field_features:", len(info["field_features"]))
    print("field feature shape:", info["field_features"][0].shape)
    print("coupling token shape:", info["coupling_tokens"].shape)
    print("coupling token max abs at init:", info["coupling_tokens"].abs().max().item())

    assert y.shape == (batch_size, 4, height, width)
    assert len(info["field_features"]) == 4
    assert info["field_features"][0].shape == (batch_size, 8, height, width)
    assert info["coupling_tokens"].shape == (batch_size, 4, 32)
    assert info["coupling_tokens"].abs().max().item() == 0.0

    print("✅ FieldWiseCouplingTokenFNO2d smoke test passed.")
