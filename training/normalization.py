import torch
import json

class BaseNormalizer:
    def normalize_x(self, x):
        return x

    def normalize_y(self, y):
        return y

    def denormalize_y(self, y):
        return y

class IdentityNormalizer(BaseNormalizer):
    pass

class FieldWiseNormalizer(BaseNormalizer):
    FIELD_NAMES = ["buoyancy", "u_x", "u_y", "pressure"]

    def __init__(self, stats_path, device="cpu", eps=1e-8):
        self.eps = eps

        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.mean = torch.tensor(
            [stats[name]["mean"] for name in self.FIELD_NAMES],
            dtype=torch.float32,
            device=device,
        )

        self.std = torch.tensor(
            [stats[name]["std"] for name in self.FIELD_NAMES],
            dtype=torch.float32,
            device=device,
        )

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def _stats_for_y(self, y):
        if y.dim() == 3:      # [4, H, W]
            return self.mean.view(4, 1, 1), self.std.view(4, 1, 1)
        elif y.dim() == 4:    # [B, 4, H, W]
            return self.mean.view(1, 4, 1, 1), self.std.view(1, 4, 1, 1)
        else:
            raise ValueError(f"Expected y shape [4,H,W] or [B,4,H,W], got {tuple(y.shape)}")

    def normalize_x(self, x):
        if x.dim() == 3:      # [16, H, W]
            C, H, W = x.shape
            assert C == 16, f"Expected 16 channels, got {C}"
            x_view = x.view(4, 4, H, W)
            mean = self.mean.view(1, 4, 1, 1)
            std = self.std.view(1, 4, 1, 1)

        elif x.dim() == 4:    # [B, 16, H, W]
            B, C, H, W = x.shape
            assert C == 16, f"Expected 16 channels, got {C}"
            x_view = x.view(B, 4, 4, H, W)
            mean = self.mean.view(1, 1, 4, 1, 1)
            std = self.std.view(1, 1, 4, 1, 1)

        else:
            raise ValueError(f"Expected x shape [16,H,W] or [B,16,H,W], got {tuple(x.shape)}")

        return ((x_view - mean) / (std + self.eps)).view_as(x)

    def normalize_y(self, y):
        mean, std = self._stats_for_y(y)
        return (y - mean) / (std + self.eps)

    def denormalize_y(self, y_norm):
        mean, std = self._stats_for_y(y_norm)
        return y_norm * (std + self.eps) + mean