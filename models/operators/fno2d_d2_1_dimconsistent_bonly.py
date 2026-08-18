import torch

from models.operators.fno2d_m10_2_bonly_rmscap import (
    M10BOnlyRMSCapFNO2d,
)


class D21DimConsistentBOnlyFNO2d(
    M10BOnlyRMSCapFNO2d
):
    """
    D2-1 formal candidate:
    dimension-consistent Path-B-only StateParam coupling.

    Compared with M10-2, the learned architecture and training
    topology are unchanged.

    Old M10-2 Path B:

        P_B_old = (-u · grad(b)) / std_b

    D2-1 canonical finite-step Path B:

        P_B_DC = dt * (-u · grad(b)) / std_b

    The canonical raw Path-B is used consistently by:
      1. the StateParam state summary,
      2. the RMSCap safety layer,
      3. the buoyancy residual injection.

    Path A remains disabled as an output residual and is retained
    only as the same compact state-summary feature used by M10-2.
    """

    def __init__(
        self,
        field_mean,
        field_std,
        *,
        dx,
        dy,
        dt,
        path_b_rms_cap,
        path_b_rms_eps=1.0e-12,
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        alpha_max=0.25,
        conditioner_hidden=32,
        freeze_m6=True,
    ):
        if dt <= 0.0:
            raise ValueError(
                "dt must be positive."
            )

        super().__init__(
            field_mean=field_mean,
            field_std=field_std,
            mode="stateparam",
            dx=dx,
            dy=dy,
            path_b_rms_cap=path_b_rms_cap,
            path_b_rms_eps=path_b_rms_eps,
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            alpha_max=alpha_max,
            conditioner_hidden=conditioner_hidden,
            freeze_m6=freeze_m6,
        )

        # Fixed, non-learnable canonical frame interval.
        # Stored in checkpoints as a persistent buffer.
        self.register_buffer(
            "canonical_dt",
            torch.tensor(
                float(dt),
                dtype=torch.float32,
            ),
            persistent=True,
        )

    def _physics_signals(
        self,
        latest_phys,
    ):
        """
        Parent implementation calculates the Path-B RATE using
        self.dx and self.dy.

        D2-1 explicitly converts that rate to one-frame increment
        representation through multiplication by canonical dt.
        """

        (
            path_a_norm,
            path_b_rate_norm,
        ) = super()._physics_signals(
            latest_phys
        )

        path_b_increment_norm = (
            path_b_rate_norm
            * self.canonical_dt.to(
                dtype=path_b_rate_norm.dtype
            )
        )

        return (
            path_a_norm,
            path_b_increment_norm,
        )

    def forward(
        self,
        x_norm,
        params=None,
        return_components=False,
    ):
        if not return_components:
            return super().forward(
                x_norm,
                params=params,
                return_components=False,
            )

        (
            output,
            components,
        ) = super().forward(
            x_norm,
            params=params,
            return_components=True,
        )

        # --------------------------------------------------------
        # D2-specific diagnostic aliases.
        #
        # Existing M10 diagnostic keys remain unchanged:
        #
        #   path_b_norm_raw
        #       = canonical finite-step Path B
        #
        #   path_b_norm_safe
        #       = canonical finite-step Path B after RMSCap
        # --------------------------------------------------------

        components[
            "path_b_increment_norm_raw"
        ] = components[
            "path_b_norm_raw"
        ]

        components[
            "path_b_increment_norm_safe"
        ] = components[
            "path_b_norm_safe"
        ]

        # Recompute the underlying rate on the SAME canonical grid.
        #
        # Calling super()._physics_signals here deliberately bypasses
        # the D2 override and retrieves the parent rate representation.
        latest_phys = self._latest_state_phys(
            x_norm
        )

        (
            _,
            path_b_rate_norm,
        ) = super()._physics_signals(
            latest_phys
        )

        components[
            "path_b_rate_norm_raw_canonical_grid"
        ] = path_b_rate_norm

        components[
            "canonical_dt"
        ] = self.canonical_dt

        return (
            output,
            components,
        )


# Explicit alias for later evaluation scripts.
D2DimConsistentBOnlyFNO2d = (
    D21DimConsistentBOnlyFNO2d
)
