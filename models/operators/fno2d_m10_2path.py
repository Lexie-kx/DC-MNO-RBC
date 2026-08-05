import torch
import torch.nn as nn

from constants import B_IDX, UX_IDX, UY_IDX, P_IDX
from models.operators.fno2d_fieldwise import FieldWiseFNO2d


class M10TwoPathFNO2d(nn.Module):
    """
    M10-0 formal 2x2 mechanism-screening model.

    Frozen audited M6 backbone
        +
    two sparse explicit physics residual paths
        +
    optional dynamic gate conditioner.

    ------------------------------------------------------------
    Factorial modes
    ------------------------------------------------------------

    static:
        global learnable base gates only

    param:
        global base gates
        + dynamic residual conditioned on [log10(Ra), log10(Pr)]

    state:
        global base gates
        + dynamic residual conditioned on latest-state summaries

    stateparam:
        global base gates
        + dynamic residual conditioned on
          [latest-state summaries, log10(Ra), log10(Pr)]

    ------------------------------------------------------------
    Fixed two-path topology
    ------------------------------------------------------------

    Path A:
        b' = b - <b>_x
        b' -> u_y delta

    Path B:
        -u · grad(b)
        -> buoyancy delta

    u_x and pressure are not directly modified.

    ------------------------------------------------------------
    Initialization contract
    ------------------------------------------------------------

    base_raw_alpha_a = base_raw_alpha_b = 0

    For dynamic modes:
        conditioner final layer is exactly zero initialized.

    Therefore every mode starts with:

        alpha_A = 0
        alpha_B = 0
        output == pure frozen M6

    exactly at initialization.

    ------------------------------------------------------------
    State summary
    ------------------------------------------------------------

    State conditioning uses ONLY the latest input state.
    No future target information is used.

    Eight compact state features are used:

        1  mean(latest normalized buoyancy)
        2  RMS(latest normalized buoyancy)
        3  mean(latest normalized u_x)
        4  RMS(latest normalized u_x)
        5  mean(latest normalized u_y)
        6  RMS(latest normalized u_y)
        7  log(1 + RMS(Path-A normalized signal))
        8  log(1 + RMS(Path-B normalized signal))

    Pressure is intentionally not used as a direct path signal.

    ------------------------------------------------------------
    Gate parameterization
    ------------------------------------------------------------

        base_raw = global learnable scalar

        dynamic_raw =
            0                                static
            tanh(conditioner(condition))     dynamic modes

        alpha =
            alpha_max * tanh(base_raw + dynamic_raw)

    Thus the final physical residual gate is bounded by alpha_max.
    """

    VALID_MODES = (
        "static",
        "param",
        "state",
        "stateparam",
    )

    STATE_DIM = 8
    PARAM_DIM = 2

    def __init__(
        self,
        field_mean,
        field_std,
        mode,
        *,
        dx,
        dy,
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        alpha_max=0.25,
        conditioner_hidden=32,
        freeze_m6=True,
    ):
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown mode={mode}. "
                f"Expected one of {self.VALID_MODES}."
            )

        if len(field_mean) != 4 or len(field_std) != 4:
            raise ValueError(
                "field_mean and field_std must each contain 4 values "
                "in [buoyancy, u_x, u_y, pressure] order."
            )

        if dx <= 0.0 or dy <= 0.0:
            raise ValueError("dx and dy must be positive.")

        if alpha_max <= 0.0:
            raise ValueError("alpha_max must be positive.")

        if conditioner_hidden <= 0:
            raise ValueError("conditioner_hidden must be positive.")

        self.mode = mode
        self.dx = float(dx)
        self.dy = float(dy)
        self.alpha_max = float(alpha_max)
        self.conditioner_hidden = int(conditioner_hidden)

        # ==========================================================
        # Audited M6 backbone
        # ==========================================================

        self.m6 = FieldWiseFNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
        )

        # ==========================================================
        # Fixed normalization statistics
        # ==========================================================

        mean = torch.as_tensor(
            field_mean,
            dtype=torch.float32,
        )

        std = torch.as_tensor(
            field_std,
            dtype=torch.float32,
        )

        if torch.any(std <= 0):
            raise ValueError(
                "All field std values must be positive."
            )

        self.register_buffer(
            "field_mean",
            mean.view(1, 4, 1, 1),
        )

        self.register_buffer(
            "field_std",
            std.view(1, 4, 1, 1),
        )

        # ==========================================================
        # Shared global base gates
        #
        # All four factorial arms have the same two learnable
        # global base gates.
        # ==========================================================

        self.base_raw_alpha_a = nn.Parameter(
            torch.zeros(())
        )

        self.base_raw_alpha_b = nn.Parameter(
            torch.zeros(())
        )

        # ==========================================================
        # Dynamic conditioner
        #
        # Static:
        #     no conditioner.
        #
        # Param:
        #     [logRa, logPr]
        #
        # State:
        #     8 state-summary features.
        #
        # StateParam:
        #     8 state-summary features + [logRa, logPr].
        #
        # The final linear layer is exactly zero initialized,
        # guaranteeing zero dynamic residual at initialization.
        # ==========================================================

        condition_dim = self._condition_dim_for_mode(mode)

        if condition_dim == 0:
            self.conditioner = None

        else:
            self.conditioner = nn.Sequential(
                nn.Linear(
                    condition_dim,
                    self.conditioner_hidden,
                ),
                nn.GELU(),
                nn.Linear(
                    self.conditioner_hidden,
                    2,
                ),
            )

            self._zero_init_conditioner_output()

        if freeze_m6:
            self.freeze_m6()

    # ==============================================================
    # Configuration
    # ==============================================================

    @classmethod
    def _condition_dim_for_mode(cls, mode):
        if mode == "static":
            return 0

        if mode == "param":
            return cls.PARAM_DIM

        if mode == "state":
            return cls.STATE_DIM

        if mode == "stateparam":
            return cls.STATE_DIM + cls.PARAM_DIM

        raise ValueError(f"Unsupported mode={mode}")

    def _zero_init_conditioner_output(self):
        """
        Keep hidden feature extraction normally initialized,
        but force dynamic gate residual to exactly zero initially.

        Do NOT zero-initialize the whole MLP, because that would
        unnecessarily suppress gradient flow through all layers.
        """
        if self.conditioner is None:
            return

        final_layer = self.conditioner[-1]

        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    # ==============================================================
    # M6 control
    # ==============================================================

    def freeze_m6(self):
        for parameter in self.m6.parameters():
            parameter.requires_grad = False

    def unfreeze_m6(self):
        for parameter in self.m6.parameters():
            parameter.requires_grad = True

    def load_m6_state_dict(self, state_dict):
        """
        Strictly load an audited bare-M6 state_dict.
        """
        self.m6.load_state_dict(
            state_dict,
            strict=True,
        )

    # ==============================================================
    # Physical-space reconstruction
    # ==============================================================

    def _latest_state_norm(self, x_norm):
        """
        Input
        -----
        x_norm:
            [B, 16, X, Y]

            4 history frames x 4 fields.

        Returns
        -------
        latest_norm:
            [B, 4, X, Y]
        """
        if x_norm.ndim != 4:
            raise ValueError(
                "Expected x_norm [B,16,X,Y], "
                f"got shape={tuple(x_norm.shape)}"
            )

        if x_norm.shape[1] != 16:
            raise ValueError(
                "Expected exactly 16 input channels, "
                f"got {x_norm.shape[1]}"
            )

        return x_norm[:, -4:, :, :]

    def _latest_state_phys(self, x_norm):
        latest_norm = self._latest_state_norm(
            x_norm
        )

        latest_phys = (
            latest_norm * self.field_std
            + self.field_mean
        )

        return latest_phys

    # ==============================================================
    # Path A
    # ==============================================================

    def _buoyancy_fluctuation(self, b):
        """
        Remove horizontal x-direction mean independently at every y.

        b:
            [B, X, Y]

        x is tensor dimension -2.

            b'(x,y)
              =
            b(x,y) - <b(.,y)>_x
        """
        horizontal_mean = b.mean(
            dim=-2,
            keepdim=True,
        )

        return b - horizontal_mean

    # ==============================================================
    # Path B derivative operators
    # ==============================================================

    def _grad_x_periodic(self, f):
        """
        Second-order central difference
        in periodic x direction.
        """
        return (
            torch.roll(
                f,
                shifts=-1,
                dims=-2,
            )
            -
            torch.roll(
                f,
                shifts=1,
                dims=-2,
            )
        ) / (2.0 * self.dx)

    def _grad_y_nonperiodic(self, f):
        """
        Non-periodic y derivative.

        Interior:
            second-order central difference

        Boundaries:
            second-order one-sided difference
        """
        if f.shape[-1] < 3:
            raise ValueError(
                "Need at least 3 y points "
                "for second-order differences."
            )

        grad = torch.empty_like(f)

        # Interior.
        grad[..., 1:-1] = (
            f[..., 2:]
            -
            f[..., :-2]
        ) / (2.0 * self.dy)

        # Lower boundary.
        grad[..., 0] = (
            -3.0 * f[..., 0]
            + 4.0 * f[..., 1]
            - f[..., 2]
        ) / (2.0 * self.dy)

        # Upper boundary.
        grad[..., -1] = (
            3.0 * f[..., -1]
            - 4.0 * f[..., -2]
            + f[..., -3]
        ) / (2.0 * self.dy)

        return grad

    # ==============================================================
    # Fixed two-path signals
    # ==============================================================

    def _physics_signals(self, latest_phys):
        """
        Returns
        -------
        path_a_norm:
            b' represented in normalized u_y-delta units.

        path_b_norm:
            -u·grad(b) represented in normalized
            buoyancy-delta units.
        """

        b = latest_phys[:, B_IDX, :, :]
        ux = latest_phys[:, UX_IDX, :, :]
        uy = latest_phys[:, UY_IDX, :, :]

        # ----------------------------------------------------------
        # Path A
        # b' -> u_y
        # ----------------------------------------------------------

        b_prime = self._buoyancy_fluctuation(
            b
        )

        uy_std = self.field_std[
            :,
            UY_IDX,
            :,
            :,
        ]

        path_a_norm = (
            b_prime / uy_std
        )

        # ----------------------------------------------------------
        # Path B
        # -u · grad(b) -> buoyancy
        # ----------------------------------------------------------

        db_dx = self._grad_x_periodic(
            b
        )

        db_dy = self._grad_y_nonperiodic(
            b
        )

        minus_u_dot_grad_b = -(
            ux * db_dx
            +
            uy * db_dy
        )

        b_std = self.field_std[
            :,
            B_IDX,
            :,
            :,
        ]

        path_b_norm = (
            minus_u_dot_grad_b / b_std
        )

        return (
            path_a_norm,
            path_b_norm,
        )

    # ==============================================================
    # State conditioner features
    # ==============================================================

    @staticmethod
    def _spatial_mean(z):
        return z.mean(
            dim=(-2, -1)
        )

    @staticmethod
    def _spatial_rms(z):
        return torch.sqrt(
            torch.mean(
                z * z,
                dim=(-2, -1),
            )
            + 1.0e-12
        )

    def _state_summary(
        self,
        x_norm,
        path_a_norm,
        path_b_norm,
    ):
        """
        Construct compact state-only information.

        Uses ONLY the latest input frame.
        """

        latest_norm = self._latest_state_norm(
            x_norm
        )

        b_norm = latest_norm[
            :,
            B_IDX,
            :,
            :,
        ]

        ux_norm = latest_norm[
            :,
            UX_IDX,
            :,
            :,
        ]

        uy_norm = latest_norm[
            :,
            UY_IDX,
            :,
            :,
        ]

        features = torch.stack(
            [
                self._spatial_mean(
                    b_norm
                ),
                self._spatial_rms(
                    b_norm
                ),
                self._spatial_mean(
                    ux_norm
                ),
                self._spatial_rms(
                    ux_norm
                ),
                self._spatial_mean(
                    uy_norm
                ),
                self._spatial_rms(
                    uy_norm
                ),
                torch.log1p(
                    self._spatial_rms(
                        path_a_norm
                    )
                ),
                torch.log1p(
                    self._spatial_rms(
                        path_b_norm
                    )
                ),
            ],
            dim=-1,
        )

        return features

    # ==============================================================
    # Parameter conditioner features
    # ==============================================================

    def _validate_params(
        self,
        params,
        batch_size,
        device,
        dtype,
    ):
        """
        params must be:

            [B, 2]
            =
            [log10(Ra), log10(Pr)]

        exactly as provided by RBCDataset(return_params=True).
        """

        if params is None:
            raise ValueError(
                f"M10 mode '{self.mode}' "
                "requires params=[log10(Ra), log10(Pr)]."
            )

        if not torch.is_tensor(params):
            params = torch.as_tensor(
                params,
                dtype=dtype,
                device=device,
            )

        else:
            params = params.to(
                device=device,
                dtype=dtype,
            )

        if params.ndim != 2:
            raise ValueError(
                "Expected params [B,2], "
                f"got shape={tuple(params.shape)}"
            )

        if params.shape[0] != batch_size:
            raise ValueError(
                "Parameter batch size does not match input. "
                f"params={params.shape[0]}, "
                f"x={batch_size}"
            )

        if params.shape[1] != self.PARAM_DIM:
            raise ValueError(
                "Expected params columns "
                "[log10(Ra), log10(Pr)], "
                f"got {params.shape[1]} columns."
            )

        return params

    # ==============================================================
    # Conditioner input
    # ==============================================================

    def _build_condition(
        self,
        x_norm,
        path_a_norm,
        path_b_norm,
        params,
    ):
        batch_size = x_norm.shape[0]

        state_summary = None
        param_features = None

        if self.mode in (
            "state",
            "stateparam",
        ):
            state_summary = self._state_summary(
                x_norm,
                path_a_norm,
                path_b_norm,
            )

        if self.mode in (
            "param",
            "stateparam",
        ):
            param_features = self._validate_params(
                params=params,
                batch_size=batch_size,
                device=x_norm.device,
                dtype=x_norm.dtype,
            )

        if self.mode == "static":
            condition = None

        elif self.mode == "param":
            condition = param_features

        elif self.mode == "state":
            condition = state_summary

        elif self.mode == "stateparam":
            condition = torch.cat(
                [
                    state_summary,
                    param_features,
                ],
                dim=-1,
            )

        else:
            raise RuntimeError(
                f"Unhandled mode={self.mode}"
            )

        return (
            condition,
            state_summary,
            param_features,
        )

    # ==============================================================
    # Dynamic gates
    # ==============================================================

    def _gate_values(
        self,
        batch_size,
        condition,
        device,
        dtype,
    ):
        """
        Return per-sample:

            alpha_a [B]
            alpha_b [B]

        Static has zero dynamic residual.

        Dynamic modes use bounded conditioner residual:
            tanh(conditioner(...))
        """

        base_raw = torch.stack(
            [
                self.base_raw_alpha_a,
                self.base_raw_alpha_b,
            ],
            dim=0,
        )

        base_raw = base_raw.view(
            1,
            2,
        ).expand(
            batch_size,
            2,
        )

        if self.conditioner is None:
            dynamic_raw = torch.zeros(
                batch_size,
                2,
                device=device,
                dtype=dtype,
            )

        else:
            if condition is None:
                raise RuntimeError(
                    "Dynamic mode received no condition."
                )

            dynamic_raw = torch.tanh(
                self.conditioner(
                    condition
                )
            )

        combined_raw = (
            base_raw
            + dynamic_raw
        )

        alpha = (
            self.alpha_max
            * torch.tanh(
                combined_raw
            )
        )

        alpha_a = alpha[:, 0]
        alpha_b = alpha[:, 1]

        return (
            alpha_a,
            alpha_b,
            base_raw,
            dynamic_raw,
            combined_raw,
        )

    # ==============================================================
    # Forward
    # ==============================================================

    def forward(
        self,
        x_norm,
        params=None,
        return_components=False,
    ):
        """
        Parameters
        ----------
        x_norm:
            normalized history
            [B,16,X,Y]

        params:
            required only for param/stateparam:
            [B,2] = [log10(Ra), log10(Pr)]

        Returns
        -------
        normalized delta
            [B,4,X,Y]
        """

        # ----------------------------------------------------------
        # Frozen audited M6 base prediction.
        # ----------------------------------------------------------

        base_delta_norm = self.m6(
            x_norm
        )

        # ----------------------------------------------------------
        # Fixed Path-A/B signals.
        # ----------------------------------------------------------

        latest_phys = self._latest_state_phys(
            x_norm
        )

        (
            path_a_norm,
            path_b_norm,
        ) = self._physics_signals(
            latest_phys
        )

        # ----------------------------------------------------------
        # Build optional dynamic condition.
        # ----------------------------------------------------------

        (
            condition,
            state_summary,
            param_features,
        ) = self._build_condition(
            x_norm=x_norm,
            path_a_norm=path_a_norm,
            path_b_norm=path_b_norm,
            params=params,
        )

        # ----------------------------------------------------------
        # Per-sample bounded gates.
        # ----------------------------------------------------------

        (
            alpha_a,
            alpha_b,
            base_raw,
            dynamic_raw,
            combined_raw,
        ) = self._gate_values(
            batch_size=x_norm.shape[0],
            condition=condition,
            device=x_norm.device,
            dtype=x_norm.dtype,
        )

        # ----------------------------------------------------------
        # Sparse residual injection.
        #
        # Path A -> only u_y
        # Path B -> only buoyancy
        #
        # u_x and pressure remain untouched directly.
        # ----------------------------------------------------------

        residual = torch.zeros_like(
            base_delta_norm
        )

        residual[
            :,
            UY_IDX,
            :,
            :,
        ] = (
            alpha_a[:, None, None]
            * path_a_norm
        )

        residual[
            :,
            B_IDX,
            :,
            :,
        ] = (
            alpha_b[:, None, None]
            * path_b_norm
        )

        output = (
            base_delta_norm
            + residual
        )

        if return_components:
            return output, {
                "mode": self.mode,
                "base_delta_norm": (
                    base_delta_norm
                ),
                "physics_residual_norm": (
                    residual
                ),
                "path_a_norm": (
                    path_a_norm
                ),
                "path_b_norm": (
                    path_b_norm
                ),
                "alpha_a": (
                    alpha_a
                ),
                "alpha_b": (
                    alpha_b
                ),
                "base_raw": (
                    base_raw
                ),
                "dynamic_raw": (
                    dynamic_raw
                ),
                "combined_raw": (
                    combined_raw
                ),
                "condition": (
                    condition
                ),
                "state_summary": (
                    state_summary
                ),
                "param_features": (
                    param_features
                ),
            }

        return output

    # ==============================================================
    # Audit helpers
    # ==============================================================

    def trainable_parameter_names(self):
        return [
            name
            for name, parameter
            in self.named_parameters()
            if parameter.requires_grad
        ]

    def frozen_parameter_names(self):
        return [
            name
            for name, parameter
            in self.named_parameters()
            if not parameter.requires_grad
        ]

    def trainable_parameter_count(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def frozen_m6_parameter_tensor_count(self):
        return sum(
            1
            for parameter
            in self.m6.parameters()
            if not parameter.requires_grad
        )


# Explicit alias for experiment naming.
M10Structured2PathFNO2d = M10TwoPathFNO2d
