import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.blocks.field_coupling import FieldCouplingBlock


def main():
    torch.manual_seed(42)

    B = 2
    F = 4
    C = 32
    H = 16
    W = 64

    block = FieldCouplingBlock(
        channels=C,
        num_fields=F,
        hidden_channels=C,
        dropout=0.0,
        init_gate=-4.0,
        use_norm=True,
    )

    x = torch.randn(B, F, C, H, W, requires_grad=True)
    y = block(x)

    print("========== FieldCouplingBlock Shape Test ==========")
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")

    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    assert y.shape[-2:] == (H, W), "Spatial size changed."
    assert torch.isfinite(y).all(), "Output contains NaN or Inf."

    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None, "Input gradient is None."
    assert torch.isfinite(x.grad).all(), "Input gradient contains NaN or Inf."
    assert block.coupling_matrix.grad is not None, "coupling_matrix grad is None."
    assert block.residual_gate.grad is not None, "residual_gate grad is None."

    gate = block.gate_values().detach().cpu()
    coupling = block.effective_coupling_matrix().detach().cpu()

    print(f"Loss: {loss.item():.6f}")
    print(f"Gate values: {gate.tolist()}")
    print("Effective coupling matrix:")
    print(coupling)

    print("✅ Shape test passed.")
    print("✅ Backward test passed.")
    print("✅ No NaN/Inf detected.")
    print("✅ Spatial resolution unchanged.")
    print("===================================================")


if __name__ == "__main__":
    main()
