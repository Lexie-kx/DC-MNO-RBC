import torch
import torch.nn as nn

from constants import B_IDX, UX_IDX, UY_IDX, P_IDX
from models.operators.fno2d_fieldwise import FieldWiseFNO2d


class M10PrePhysicsPathFNO2d(nn.Module):
    """
    M10-Pre: frozen-M6 + sparse explicit physics residual paths.

    Purpose
    -------
    Lightweight path-necessity control before formal M10-0 factorial study.

    Supported modes
    ---------------
    "m6":
        Pure frozen M6. No physics residual.

    "path_b":
        Frozen M6 + Path B only:
            -u · grad(b) -> buoyancy delta

    "path_ab":
        Frozen M6 + Path A + Path B:
            b' -> u_y delta
            -u · grad(b) -> buoyancy delta

    Physics signals are constructed from the latest physical state, while
    the residual is converted into M6 normalized-delta space before addition.

    Important
    ---------
    raw_alpha_a = raw_alpha_b = 0 at initialization.

    Therefore:
        alpha_a = alpha_b = 0
        output = pure M6 output

    exactly at initialization.
    """

    VALID_PATH_MODES = ("m6", "path_b", "path_ab")

    def __init__(
        self,
        field_mean,
        field_std,
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        path_mode="path_ab",
        dx=4.0 / 256.0,
        dy=1.0 / 64.0,
        alpha_max=0.25,
        freeze_m6=True,
    ):
        super().__init__()

        if path_mode not in self.VALID_PATH_MODES:
            raise ValueError(
                f"Unknown path_mode={path_mode}. "
                f"Expected one of {self.VALID_PATH_MODES}."
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

        self.path_mode = path_mode
        self.dx = float(dx)
        self.dy = float(dy)
        self.alpha_max = float(alpha_max)

        # ----------------------------------------------------------
        # Audited M6 backbone.
        # ----------------------------------------------------------
        self.m6 = FieldWiseFNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
        )

        # ----------------------------------------------------------
        # Fixed field normalization statistics.
        #
        # Shape:
        #   [1, 4, 1, 1]
        #
        # These are buffers, not trainable parameters.
        # ----------------------------------------------------------
        mean = torch.as_tensor(field_mean, dtype=torch.float32)
        std = torch.as_tensor(field_std, dtype=torch.float32)

        if torch.any(std <= 0):
            raise ValueError("All field std values must be positive.")

        self.register_buffer(
            "field_mean",
            mean.view(1, 4, 1, 1),
        )
        self.register_buffer(
            "field_std",
            std.view(1, 4, 1, 1),
        )

        # ----------------------------------------------------------
        # Zero-initialized bounded path gates.
        #
        # alpha = alpha_max * tanh(raw_alpha)
        #
        # raw_alpha = 0  => alpha = 0
        # ----------------------------------------------------------
        self.raw_alpha_a = nn.Parameter(torch.zeros(()))
        self.raw_alpha_b = nn.Parameter(torch.zeros(()))

        # ----------------------------------------------------------
        # Only gates that are actually used by the selected
        # M10-Pre path topology are trainable.
        #
        # m6:
        #     no trainable physics gates
        #
        # path_b:
        #     only Path B gate is trainable
        #
        # path_ab:
        #     both Path A and Path B gates are trainable
        # ----------------------------------------------------------
        if self.path_mode == "m6":
            self.raw_alpha_a.requires_grad_(False)
            self.raw_alpha_b.requires_grad_(False)

        elif self.path_mode == "path_b":
            self.raw_alpha_a.requires_grad_(False)
            self.raw_alpha_b.requires_grad_(True)

        elif self.path_mode == "path_ab":
            self.raw_alpha_a.requires_grad_(True)
            self.raw_alpha_b.requires_grad_(True)

        if freeze_m6:
            self.freeze_m6()

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
        Strictly load an audited bare-M6 state_dict into the backbone.
        """
        self.m6.load_state_dict(state_dict, strict=True)

    # ==============================================================
    # Gate values
    # ==============================================================

    def alpha_a(self):
        return self.alpha_max * torch.tanh(self.raw_alpha_a)

    def alpha_b(self):
        return self.alpha_max * torch.tanh(self.raw_alpha_b)

    # ==============================================================
    # Physical-space helpers
    # ==============================================================

    def _latest_state_phys(self, x_norm):
        """
        x_norm:
            [B, 16, X, Y]
            ordered as 4 history frames x 4 fields.

        Returns:
            latest physical state [B, 4, X, Y]
        """
        if x_norm.ndim != 4:
            raise ValueError(
                f"Expected x_norm [B,16,X,Y], got shape={tuple(x_norm.shape)}"
            )

        if x_norm.shape[1] != 16:
            raise ValueError(
                f"Expected 16 input channels, got {x_norm.shape[1]}"
            )

        latest_norm = x_norm[:, -4:, :, :]

        latest_phys = (
            latest_norm * self.field_std
            + self.field_mean
        )

        return latest_phys

    def _buoyancy_fluctuation(self, b):
        """
        Remove horizontal (x-direction) mean independently at every y.

        b:
            [B, X, Y]

        x is tensor dimension -2.

        b_prime(x,y) = b(x,y) - <b(.,y)>_x
        """
        horizontal_mean = b.mean(dim=-2, keepdim=True)
        return b - horizontal_mean

    def _grad_x_periodic(self, f):
        """
        Second-order central difference in periodic x direction.

        f:
            [B, X, Y]

        x is tensor dimension -2.
        """
        return (
            torch.roll(f, shifts=-1, dims=-2)
            - torch.roll(f, shifts=1, dims=-2)
        ) / (2.0 * self.dx)

    def _grad_y_nonperiodic(self, f):
        """
        y derivative on the non-periodic last spatial dimension.

        Interior:
            second-order central difference

        Ends:
            second-order one-sided difference

        f:
            [B, X, Y]
        """
        if f.shape[-1] < 3:
            raise ValueError(
                "Need at least 3 y points for second-order differences."
            )

        grad = torch.empty_like(f)

        # Interior points.
        grad[..., 1:-1] = (
            f[..., 2:]
            - f[..., :-2]
        ) / (2.0 * self.dy)

        # Lower end.
        grad[..., 0] = (
            -3.0 * f[..., 0]
            + 4.0 * f[..., 1]
            - f[..., 2]
        ) / (2.0 * self.dy)

        # Upper sampled end.
        grad[..., -1] = (
            3.0 * f[..., -1]
            - 4.0 * f[..., -2]
            + f[..., -3]
        ) / (2.0 * self.dy)

        return grad

    def _physics_signals(self, latest_phys):
        """
        Construct the two candidate explicit physics signals.

        Returns
        -------
        path_a_norm:
            b' represented in normalized u_y-delta units.

        path_b_norm:
            -u·grad(b) represented in normalized buoyancy-delta units.
        """
        b = latest_phys[:, B_IDX, :, :]
        ux = latest_phys[:, UX_IDX, :, :]
        uy = latest_phys[:, UY_IDX, :, :]

        # ----------------------------------------------------------
        # Path A:
        # horizontal buoyancy fluctuation -> vertical velocity
        # ----------------------------------------------------------
        b_prime = self._buoyancy_fluctuation(b)

        uy_std = self.field_std[:, UY_IDX, :, :]
        path_a_norm = b_prime / uy_std

        # ----------------------------------------------------------
        # Path B:
        # -u · grad(b) -> buoyancy
        # ----------------------------------------------------------
        db_dx = self._grad_x_periodic(b)
        db_dy = self._grad_y_nonperiodic(b)

        minus_u_dot_grad_b = -(
            ux * db_dx
            + uy * db_dy
        )

        b_std = self.field_std[:, B_IDX, :, :]
        path_b_norm = minus_u_dot_grad_b / b_std

        return path_a_norm, path_b_norm

    # ==============================================================
    # Forward
    # ==============================================================

    def forward(self, x_norm, return_components=False):
        """
        x_norm:
            normalized history [B,16,X,Y]

        output:
            normalized delta [B,4,X,Y]
        """

        # Pure audited M6 prediction.
        base_delta_norm = self.m6(x_norm)

        # "m6" control mode returns exactly the backbone.
        if self.path_mode == "m6":
            if return_components:
                zeros = torch.zeros_like(base_delta_norm)
                return base_delta_norm, {
                    "base_delta_norm": base_delta_norm,
                    "physics_residual_norm": zeros,
                    "alpha_a": self.alpha_a().detach(),
                    "alpha_b": self.alpha_b().detach(),
                    "path_a_norm": None,
                    "path_b_norm": None,
                }

            return base_delta_norm

        latest_phys = self._latest_state_phys(x_norm)

        path_a_norm, path_b_norm = self._physics_signals(
            latest_phys
        )

        residual = torch.zeros_like(base_delta_norm)

        # ----------------------------------------------------------
        # Path B is active in both path_b and path_ab.
        # Only modifies buoyancy delta.
        # ----------------------------------------------------------
        residual[:, B_IDX, :, :] = (
            self.alpha_b() * path_b_norm
        )

        # ----------------------------------------------------------
        # Path A exists only in path_ab.
        # Only modifies vertical-velocity delta.
        # ----------------------------------------------------------
        if self.path_mode == "path_ab":
            residual[:, UY_IDX, :, :] = (
                self.alpha_a() * path_a_norm
            )

        output = base_delta_norm + residual

        if return_components:
            return output, {
                "base_delta_norm": base_delta_norm,
                "physics_residual_norm": residual,
                "alpha_a": self.alpha_a(),
                "alpha_b": self.alpha_b(),
                "path_a_norm": path_a_norm,
                "path_b_norm": path_b_norm,
            }

        return output

    # ==============================================================
    # Audit helpers
    # ==============================================================

    def trainable_parameter_names(self):
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def frozen_parameter_names(self):
        return [
            name
            for name, parameter in self.named_parameters()
            if not parameter.requires_grad
        ]
