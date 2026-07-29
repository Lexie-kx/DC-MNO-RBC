"""
M8-C Physics-Inspired Structured Coupling Tokens.

Two controlled modes share exactly the same branch structure:

1. structured_static
   M8-C0 control experiment:
   shared structured branch strengths.

2. structured_parameter_conditioned
   M8-C1 formal candidate:
   the same structured branches, with limited parameter-conditioned
   corrections for buoyancy, viscous, and thermal-diffusion strengths.

Important scope:
- This module does NOT explicitly compute PDE terms.
- Branch names describe restricted, physics-inspired information pathways.
- No PDE loss is implemented here.
- No rollout logic is implemented here.

Input:
    x: [B, F=4, C, H, W]

Field order:
    0: buoyancy
    1: u_x
    2: u_y
    3: pressure

Output:
    y: [B, F=4, C, H, W]
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class LocalStructuredBranch(nn.Module):
    """
    Small local feature branch with fixed input and output roles.

    Architecture:
        GroupNorm
        -> 1x1 projection
        -> GELU
        -> 3x3 local feature extraction
        -> GELU
        -> 1x1 output projection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )
        if out_channels <= 0:
            raise ValueError(
                f"out_channels must be positive, got {out_channels}"
            )
        if hidden_channels <= 0:
            raise ValueError(
                f"hidden_channels must be positive, got {hidden_channels}"
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)

        self.norm = nn.GroupNorm(
            num_groups=1,
            num_channels=self.in_channels,
        )

        self.in_proj = nn.Conv2d(
            self.in_channels,
            self.hidden_channels,
            kernel_size=1,
        )

        self.local_conv = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
        )

        self.out_proj = nn.Conv2d(
            self.hidden_channels,
            self.out_channels,
            kernel_size=1,
        )

        self.act = nn.GELU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(
            self.in_proj.weight,
            a=5**0.5,
        )
        nn.init.zeros_(self.in_proj.bias)

        nn.init.kaiming_uniform_(
            self.local_conv.weight,
            a=5**0.5,
        )
        nn.init.zeros_(self.local_conv.bias)

        # Small initial output keeps the whole coupling block close to identity.
        nn.init.normal_(
            self.out_proj.weight,
            mean=0.0,
            std=0.02,
        )
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "LocalStructuredBranch expects [B, C, H, W], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {x.shape[1]}"
            )

        x = self.norm(x)
        x = self.in_proj(x)
        x = self.act(x)
        x = self.local_conv(x)
        x = self.act(x)
        x = self.out_proj(x)

        return x


class M8CStructuredCouplingBlock(nn.Module):
    """
    Structured coupling block shared by M8-C0 and M8-C1.

    Branch families:
        buoyancy
        advection
        pressure
        viscous
        thermal_diffusion

    structured_static:
        Uses shared learnable branch strengths.

    structured_parameter_conditioned:
        Uses the same shared strengths plus limited corrections:

        buoyancy correction:
            [log10(Ra), log10(Pr)] -> scalar logit correction

        viscous correction:
            log10(nu) -> scalar logit correction

        thermal diffusion correction:
            log10(kappa) -> scalar logit correction

        Advection and pressure remain shared in the first version.
    """

    VALID_MODES = {
        "structured_static",
        "structured_parameter_conditioned",
    }

    BRANCH_NAMES = (
        "buoyancy",
        "advection",
        "pressure",
        "viscous",
        "thermal_diffusion",
    )

    FIELD_ORDER = (
        "buoyancy",
        "u_x",
        "u_y",
        "pressure",
    )

    BUOYANCY_INDEX = 0
    UX_INDEX = 1
    UY_INDEX = 2
    PRESSURE_INDEX = 3

    def __init__(
        self,
        channels: int,
        num_fields: int = 4,
        hidden_channels: int | None = None,
        condition_hidden_dim: int = 32,
        condition_scale: float = 1.0,
        init_strength_logit: float = -4.0,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be positive, got {channels}"
            )

        if num_fields != 4:
            raise ValueError(
                "M8-C v1 requires exactly four fields "
                "[buoyancy, u_x, u_y, pressure], "
                f"got num_fields={num_fields}"
            )

        if condition_hidden_dim <= 0:
            raise ValueError(
                "condition_hidden_dim must be positive, "
                f"got {condition_hidden_dim}"
            )

        if condition_scale <= 0:
            raise ValueError(
                f"condition_scale must be positive, got {condition_scale}"
            )

        self.channels = int(channels)
        self.num_fields = int(num_fields)
        self.hidden_channels = int(
            hidden_channels or channels
        )
        self.condition_hidden_dim = int(condition_hidden_dim)
        self.condition_scale = float(condition_scale)

        c = self.channels
        h = self.hidden_channels

        # ---------------------------------------------------------
        # 1. Buoyancy-driving branch
        #
        # buoyancy -> u_y
        # ---------------------------------------------------------
        self.buoyancy_branch = LocalStructuredBranch(
            in_channels=c,
            out_channels=c,
            hidden_channels=h,
        )

        # ---------------------------------------------------------
        # 2. Advection family
        #
        # a) [buoyancy, u_x, u_y] -> buoyancy
        # b) [u_x, u_y] -> [u_x, u_y]
        #
        # Both subheads share the same advection strength.
        # ---------------------------------------------------------
        self.advection_buoyancy_branch = LocalStructuredBranch(
            in_channels=3 * c,
            out_channels=c,
            hidden_channels=h,
        )

        self.advection_velocity_branch = LocalStructuredBranch(
            in_channels=2 * c,
            out_channels=2 * c,
            hidden_channels=h,
        )

        # ---------------------------------------------------------
        # 3. Pressure-constraint branch
        #
        # pressure -> [u_x, u_y]
        # ---------------------------------------------------------
        self.pressure_branch = LocalStructuredBranch(
            in_channels=c,
            out_channels=2 * c,
            hidden_channels=h,
        )

        # ---------------------------------------------------------
        # 4. Viscous branch
        #
        # [u_x, u_y] -> [u_x, u_y]
        # ---------------------------------------------------------
        self.viscous_branch = LocalStructuredBranch(
            in_channels=2 * c,
            out_channels=2 * c,
            hidden_channels=h,
        )

        # ---------------------------------------------------------
        # 5. Thermal-diffusion branch
        #
        # buoyancy -> buoyancy
        # ---------------------------------------------------------
        self.thermal_diffusion_branch = LocalStructuredBranch(
            in_channels=c,
            out_channels=c,
            hidden_channels=h,
        )

        # Shared base strength logits for the five branch families.
        self.base_strength_logits = nn.Parameter(
            torch.full(
                (len(self.BRANCH_NAMES),),
                float(init_strength_logit),
                dtype=torch.float32,
            )
        )

        # Limited parameter-conditioned correction heads.
        self.buoyancy_conditioner = self._make_conditioner(
            in_dim=2,
        )
        self.viscous_conditioner = self._make_conditioner(
            in_dim=1,
        )
        self.thermal_diffusion_conditioner = (
            self._make_conditioner(
                in_dim=1,
            )
        )

        self.reset_conditioners()

    def _make_conditioner(
        self,
        in_dim: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(
                in_dim,
                self.condition_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.condition_hidden_dim,
                1,
            ),
        )

    def reset_conditioners(self) -> None:
        """
        Zero-initialize all final layers.

        Therefore, before training:

            structured_static output
            ==
            structured_parameter_conditioned output
        """

        for conditioner in (
            self.buoyancy_conditioner,
            self.viscous_conditioner,
            self.thermal_diffusion_conditioner,
        ):
            final_layer = conditioner[-1]

            if not isinstance(final_layer, nn.Linear):
                raise TypeError(
                    "Conditioner final layer must be nn.Linear"
                )

            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    @staticmethod
    def expand_param(
        param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert:

            [log10(Ra), log10(Pr)]

        into:

            [
                log10(Ra),
                log10(Pr),
                log10(nu),
                log10(kappa),
            ]

        where:

            log10(nu)
            =
            -0.5 * (log10(Ra) - log10(Pr))

            log10(kappa)
            =
            -0.5 * (log10(Ra) + log10(Pr))
        """

        if param.ndim != 2 or param.shape[1] != 2:
            raise ValueError(
                "param must be [B, 2] = "
                "[log10(Ra), log10(Pr)], "
                f"got {tuple(param.shape)}"
            )

        log_ra = param[:, 0]
        log_pr = param[:, 1]

        log_nu = -0.5 * (
            log_ra - log_pr
        )
        log_kappa = -0.5 * (
            log_ra + log_pr
        )

        return torch.stack(
            [
                log_ra,
                log_pr,
                log_nu,
                log_kappa,
            ],
            dim=1,
        )

    def base_strengths(self) -> torch.Tensor:
        """
        Return shared base strengths in [0, 1].

        Shape:
            [5]
        """

        return torch.sigmoid(
            self.base_strength_logits
        )

    def conditioned_delta_logits(
        self,
        param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return limited sample-dependent logit corrections.

        Shape:
            [B, 5]

        Branch order:
            buoyancy
            advection
            pressure
            viscous
            thermal_diffusion

        Advection and pressure corrections remain exactly zero
        in M8-C v1.
        """

        param4 = self.expand_param(param)

        log_ra_pr = param4[:, 0:2]
        log_nu = param4[:, 2:3]
        log_kappa = param4[:, 3:4]

        buoyancy_delta = self.buoyancy_conditioner(
            log_ra_pr
        )
        viscous_delta = self.viscous_conditioner(
            log_nu
        )
        diffusion_delta = (
            self.thermal_diffusion_conditioner(
                log_kappa
            )
        )

        buoyancy_delta = (
            torch.tanh(buoyancy_delta)
            * self.condition_scale
        )
        viscous_delta = (
            torch.tanh(viscous_delta)
            * self.condition_scale
        )
        diffusion_delta = (
            torch.tanh(diffusion_delta)
            * self.condition_scale
        )

        zero = torch.zeros_like(
            buoyancy_delta
        )

        return torch.cat(
            [
                buoyancy_delta,
                zero,
                zero,
                viscous_delta,
                diffusion_delta,
            ],
            dim=1,
        )

    def effective_strengths(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        mode: str,
        param: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            strengths:
                [B, 5], values in [0, 1]

            delta_logits:
                [B, 5]
        """

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown mode={mode}. "
                f"Expected one of {sorted(self.VALID_MODES)}"
            )

        base_logits = self.base_strength_logits.to(
            device=device,
            dtype=dtype,
        )

        base_logits = base_logits.view(
            1,
            -1,
        ).expand(
            batch_size,
            -1,
        )

        if mode == "structured_static":
            delta_logits = torch.zeros_like(
                base_logits
            )

        else:
            if param is None:
                raise ValueError(
                    "param is required for "
                    "structured_parameter_conditioned mode"
                )

            param = param.to(
                device=device,
                dtype=dtype,
            )

            delta_logits = (
                self.conditioned_delta_logits(
                    param
                )
            )

        strengths = torch.sigmoid(
            base_logits + delta_logits
        )

        return strengths, delta_logits

    @staticmethod
    def _family_rms(
        contribution: torch.Tensor,
    ) -> torch.Tensor:
        """
        contribution:
            [B, F, C, H, W]

        returns:
            [B]
        """

        return torch.sqrt(
            torch.mean(
                contribution.square(),
                dim=(1, 2, 3, 4),
            )
            + 1e-12
        )

    def forward(
        self,
        x: torch.Tensor,
        param: torch.Tensor | None = None,
        mode: str = "structured_static",
        return_diagnostics: bool = False,
    ):
        if x.ndim != 5:
            raise ValueError(
                "M8CStructuredCouplingBlock expects "
                "[B, F, C, H, W], "
                f"got {tuple(x.shape)}"
            )

        (
            batch_size,
            fields,
            channels,
            height,
            width,
        ) = x.shape

        if fields != self.num_fields:
            raise ValueError(
                f"Expected num_fields={self.num_fields}, "
                f"got {fields}"
            )

        if channels != self.channels:
            raise ValueError(
                f"Expected channels={self.channels}, "
                f"got {channels}"
            )

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown mode={mode}. "
                f"Expected one of {sorted(self.VALID_MODES)}"
            )

        buoyancy = x[:, self.BUOYANCY_INDEX]
        u_x = x[:, self.UX_INDEX]
        u_y = x[:, self.UY_INDEX]
        pressure = x[:, self.PRESSURE_INDEX]

        zero = torch.zeros_like(
            buoyancy
        )

        strengths, delta_logits = (
            self.effective_strengths(
                batch_size=batch_size,
                device=x.device,
                dtype=x.dtype,
                mode=mode,
                param=param,
            )
        )

        # ---------------------------------------------------------
        # Buoyancy family:
        # buoyancy -> u_y
        # ---------------------------------------------------------
        buoyancy_to_uy = self.buoyancy_branch(
            buoyancy
        )

        buoyancy_contribution = torch.stack(
            [
                zero,
                zero,
                buoyancy_to_uy,
                zero,
            ],
            dim=1,
        )

        # ---------------------------------------------------------
        # Advection family:
        #
        # [b, u_x, u_y] -> b
        # [u_x, u_y]    -> [u_x, u_y]
        # ---------------------------------------------------------
        adv_b_input = torch.cat(
            [
                buoyancy,
                u_x,
                u_y,
            ],
            dim=1,
        )

        adv_b_output = (
            self.advection_buoyancy_branch(
                adv_b_input
            )
        )

        velocity_input = torch.cat(
            [
                u_x,
                u_y,
            ],
            dim=1,
        )

        adv_u_output = (
            self.advection_velocity_branch(
                velocity_input
            )
        )

        adv_u_output = adv_u_output.reshape(
            batch_size,
            2,
            channels,
            height,
            width,
        )

        advection_contribution = torch.stack(
            [
                adv_b_output,
                adv_u_output[:, 0],
                adv_u_output[:, 1],
                zero,
            ],
            dim=1,
        )

        # ---------------------------------------------------------
        # Pressure family:
        # pressure -> [u_x, u_y]
        # ---------------------------------------------------------
        pressure_output = self.pressure_branch(
            pressure
        )

        pressure_output = pressure_output.reshape(
            batch_size,
            2,
            channels,
            height,
            width,
        )

        pressure_contribution = torch.stack(
            [
                zero,
                pressure_output[:, 0],
                pressure_output[:, 1],
                zero,
            ],
            dim=1,
        )

        # ---------------------------------------------------------
        # Viscous family:
        # [u_x, u_y] -> [u_x, u_y]
        # ---------------------------------------------------------
        viscous_output = self.viscous_branch(
            velocity_input
        )

        viscous_output = viscous_output.reshape(
            batch_size,
            2,
            channels,
            height,
            width,
        )

        viscous_contribution = torch.stack(
            [
                zero,
                viscous_output[:, 0],
                viscous_output[:, 1],
                zero,
            ],
            dim=1,
        )

        # ---------------------------------------------------------
        # Thermal-diffusion family:
        # buoyancy -> buoyancy
        # ---------------------------------------------------------
        diffusion_output = (
            self.thermal_diffusion_branch(
                buoyancy
            )
        )

        diffusion_contribution = torch.stack(
            [
                diffusion_output,
                zero,
                zero,
                zero,
            ],
            dim=1,
        )

        raw_contributions = (
            buoyancy_contribution,
            advection_contribution,
            pressure_contribution,
            viscous_contribution,
            diffusion_contribution,
        )

        weighted_contributions = []

        for branch_idx, contribution in enumerate(
            raw_contributions
        ):
            strength = strengths[
                :,
                branch_idx,
            ].view(
                batch_size,
                1,
                1,
                1,
                1,
            )

            weighted_contributions.append(
                strength * contribution
            )

        total_contribution = torch.stack(
            weighted_contributions,
            dim=0,
        ).sum(
            dim=0,
        )

        y = x + total_contribution

        if not return_diagnostics:
            return y

        branch_rms = torch.stack(
            [
                self._family_rms(contribution)
                for contribution in weighted_contributions
            ],
            dim=1,
        )

        diagnostics: Dict[str, object] = {
            "mode": mode,
            "branch_names": self.BRANCH_NAMES,
            "field_order": self.FIELD_ORDER,
            "base_strengths": self.base_strengths(),
            "effective_strengths": strengths,
            "conditioned_delta_logits": delta_logits,
            "weighted_branch_rms": branch_rms,
            "total_contribution_rms": (
                self._family_rms(
                    total_contribution
                )
            ),
        }

        return y, diagnostics


if __name__ == "__main__":
    torch.manual_seed(7)

    batch_size = 3
    channels = 8
    height = 16
    width = 64

    x = torch.randn(
        batch_size,
        4,
        channels,
        height,
        width,
    )

    param = torch.tensor(
        [
            [6.0, -0.30103],
            [7.0, 0.0],
            [8.0, 0.30103],
        ],
        dtype=torch.float32,
    )

    block = M8CStructuredCouplingBlock(
        channels=channels,
        num_fields=4,
        hidden_channels=channels,
        condition_hidden_dim=32,
        condition_scale=1.0,
        init_strength_logit=-4.0,
    )

    block.eval()

    with torch.no_grad():
        y_static, static_info = block(
            x,
            param=param,
            mode="structured_static",
            return_diagnostics=True,
        )

        y_conditioned_zero, conditioned_zero_info = block(
            x,
            param=param,
            mode="structured_parameter_conditioned",
            return_diagnostics=True,
        )

    initial_max_diff = torch.max(
        torch.abs(
            y_static
            - y_conditioned_zero
        )
    ).item()

    initial_delta_abs_max = (
        conditioned_zero_info[
            "conditioned_delta_logits"
        ]
        .abs()
        .max()
        .item()
    )

    print("input shape:", tuple(x.shape))
    print("static output shape:", tuple(y_static.shape))
    print(
        "conditioned output shape:",
        tuple(y_conditioned_zero.shape),
    )
    print(
        "branch names:",
        static_info["branch_names"],
    )
    print(
        "base strengths:",
        static_info[
            "base_strengths"
        ].detach().cpu().tolist(),
    )
    print(
        "initial static-conditioned max diff:",
        f"{initial_max_diff:.12e}",
    )
    print(
        "initial conditioned delta abs max:",
        f"{initial_delta_abs_max:.12e}",
    )
    print(
        "initial weighted branch RMS:",
        static_info[
            "weighted_branch_rms"
        ].mean(dim=0).detach().cpu().tolist(),
    )

    assert y_static.shape == x.shape
    assert y_conditioned_zero.shape == x.shape
    assert initial_max_diff < 1e-7
    assert initial_delta_abs_max < 1e-7

    # Confirm that the limited conditioned path can affect output
    # after one correction head becomes non-zero.
    with torch.no_grad():
        block.buoyancy_conditioner[
            -1
        ].bias.fill_(0.50)

        y_conditioned_active, active_info = block(
            x,
            param=param,
            mode="structured_parameter_conditioned",
            return_diagnostics=True,
        )

    active_max_diff = torch.max(
        torch.abs(
            y_static
            - y_conditioned_active
        )
    ).item()

    print(
        "activated conditioned-static max diff:",
        f"{active_max_diff:.12e}",
    )
    print(
        "activated conditioned delta logits:",
        active_info[
            "conditioned_delta_logits"
        ].detach().cpu().tolist(),
    )

    assert active_max_diff > 0.0

    # Advection and pressure corrections must remain zero in v1.
    active_delta = active_info[
        "conditioned_delta_logits"
    ]

    assert torch.max(
        torch.abs(active_delta[:, 1])
    ).item() < 1e-7

    assert torch.max(
        torch.abs(active_delta[:, 2])
    ).item() < 1e-7

    print(
        "✅ M8CStructuredCouplingBlock smoke test passed."
    )
