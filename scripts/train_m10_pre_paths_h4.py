import argparse
import hashlib
import json
import os
import random
import sys

# Ensure project root is importable when running:
# python scripts/train_m10_pre_paths_h4.py
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from constants import FIELD_ORDER
from datasets.rbc_dataset import RBCDataset
from training.metrics import FieldWiseRelativeL2Loss

from models.operators.fno2d_m10_pre_paths import (
    M10PrePhysicsPathFNO2d,
)

from scripts.train_m6_fieldwise_encoder_h4 import (
    MultiStepDeltaDataset,
    make_rollout_weights,
    autoregressive_multistep_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "M10-Pre H4 path-necessity screening: "
            "frozen audited M6 + sparse physics residual paths."
        )
    )

    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--stats", type=str, required=True)
    parser.add_argument("--m6_checkpoint", type=str, required=True)

    parser.add_argument(
        "--path_mode",
        type=str,
        required=True,
        choices=["path_b", "path_ab"],
    )

    parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)

    # Keep M10-Pre aligned with the audited M6 H4 training scale.
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--rollout_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--dx",
        type=float,
        default=4.0 / 256.0,
    )

    parser.add_argument(
        "--dy",
        type=float,
        default=1.0 / 64.0,
    )

    parser.add_argument(
        "--alpha_max",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/m10_pre",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # No stochastic layers exist in the frozen M6 backbone, but keep
    # CuDNN behavior explicit for reproducibility.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def build_field_stats(stats_path):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    field_mean = [
        float(stats[field]["mean"])
        for field in FIELD_ORDER
    ]

    field_std = [
        float(stats[field]["std"])
        for field in FIELD_ORDER
    ]

    return field_mean, field_std


def evaluate(
    model,
    loader,
    criterion,
    rollout_weights,
    device,
    max_batches=None,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (context_norm, y_seq_norm) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            context_norm = context_norm.to(device)
            y_seq_norm = y_seq_norm.to(device)

            loss, _ = autoregressive_multistep_loss(
                model=model,
                context_norm=context_norm,
                y_seq_norm=y_seq_norm,
                criterion=criterion,
                rollout_weights=rollout_weights,
            )

            bs = context_norm.shape[0]

            total_loss += float(loss.item()) * bs
            total_samples += bs

    if total_samples == 0:
        raise RuntimeError("Validation loader produced zero samples.")

    return total_loss / total_samples


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    epoch,
    val_loss,
    best_val_loss,
    args,
    m6_sha256,
    trainable_names,
):
    payload = {
        "experiment": "M10-Pre-PhysicsPath-H4",
        "stage": "lightweight_path_necessity_control",
        "epoch": int(epoch),
        "path_mode": args.path_mode,

        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            None
            if optimizer is None
            else optimizer.state_dict()
        ),

        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),

        "alpha_a": float(model.alpha_a().detach().cpu()),
        "alpha_b": float(model.alpha_b().detach().cpu()),

        "raw_alpha_a": float(
            model.raw_alpha_a.detach().cpu()
        ),
        "raw_alpha_b": float(
            model.raw_alpha_b.detach().cpu()
        ),

        "alpha_max": float(args.alpha_max),
        "dx": float(args.dx),
        "dy": float(args.dy),

        "split": args.split,
        "stats": args.stats,

        "m6_checkpoint": args.m6_checkpoint,
        "m6_checkpoint_sha256": m6_sha256,

        "rollout_steps": int(args.rollout_steps),
        "rollout_weights": [
            float(x)
            for x in make_rollout_weights(
                args.rollout_steps
            ).tolist()
        ],

        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "seed": int(args.seed),

        "trainable_parameter_names": trainable_names,

        "backbone_frozen": True,
        "prediction_type": "normalized_delta",
        "loss": (
            "weighted H4 free-autoregressive "
            "FieldWiseRelativeL2Loss"
        ),
    }

    torch.save(payload, path)


def main():
    args = parse_args()

    if args.rollout_steps != 4:
        raise ValueError(
            "M10-Pre protocol is locked to rollout_steps=4."
        )

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("🚀 [M10-Pre Physics Path H4]")
    print("📌 Stage: lightweight path-necessity control")
    print("📌 Device:", device)
    print("📌 Path mode:", args.path_mode)
    print("📌 Seed:", args.seed)
    print("📌 LR:", args.lr)
    print("📌 Epochs:", args.epochs)
    print("📌 Batch size:", args.batch_size)
    print("📌 H4 rollout weights: [1.0, 0.8, 0.6, 0.4]")
    print("📌 Frozen audited M6 backbone")
    print("📌 No Parameter conditioner")
    print("📌 No State conditioner")
    print("📌 No PDE loss")

    with open(args.split, "r", encoding="utf-8") as f:
        split = json.load(f)

    field_mean, field_std = build_field_stats(
        args.stats
    )

    train_base = RBCDataset(
        split_config=split["train"],
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=args.rollout_steps,
    )

    val_base = RBCDataset(
        split_config=split["val"],
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=args.rollout_steps,
    )

    train_dataset = MultiStepDeltaDataset(
        train_base,
        context_length=4,
    )

    val_dataset = MultiStepDeltaDataset(
        val_base,
        context_length=4,
    )

    # Fresh generator with the same seed gives path_b and path_ab
    # the same epoch-1 shuffled order when runs use the same seed.
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = M10PrePhysicsPathFNO2d(
        field_mean=field_mean,
        field_std=field_std,
        path_mode=args.path_mode,
        dx=args.dx,
        dy=args.dy,
        alpha_max=args.alpha_max,
        freeze_m6=True,
    ).to(device)

    m6_checkpoint = torch.load(
        args.m6_checkpoint,
        map_location=device,
    )

    m6_state = m6_checkpoint["model_state_dict"]

    model.load_m6_state_dict(m6_state)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    expected = {
        "path_b": ["raw_alpha_b"],
        "path_ab": ["raw_alpha_a", "raw_alpha_b"],
    }[args.path_mode]

    if trainable_names != expected:
        raise RuntimeError(
            "Unexpected trainable parameters.\n"
            f"Expected: {expected}\n"
            f"Actual:   {trainable_names}"
        )

    frozen_m6_tensors = sum(
        1
        for parameter in model.m6.parameters()
        if not parameter.requires_grad
    )

    total_m6_tensors = sum(
        1
        for _ in model.m6.parameters()
    )

    if frozen_m6_tensors != total_m6_tensors:
        raise RuntimeError(
            "M6 backbone is not fully frozen."
        )

    print()
    print("========== PARAMETER AUDIT ==========")
    print("TRAINABLE:", trainable_names)
    print(
        "FROZEN_M6_PARAMETER_TENSORS:",
        frozen_m6_tensors,
    )

    m6_sha = sha256_file(args.m6_checkpoint)

    print("M6_CHECKPOINT_SHA256:", m6_sha)

    criterion = FieldWiseRelativeL2Loss()

    rollout_weights = make_rollout_weights(
        args.rollout_steps
    ).to(device)

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.lr,
    )

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True,
    )

    best_path = os.path.join(
        args.checkpoint_dir,
        f"{args.run_name}_best.pth",
    )

    last_path = os.path.join(
        args.checkpoint_dir,
        f"{args.run_name}_last.pth",
    )

    # ----------------------------------------------------------
    # Epoch 0:
    # zero-init path model must be exactly pure M6.
    #
    # Crucially, epoch 0 participates in best-checkpoint
    # selection. If training never beats pure M6, the experiment
    # correctly returns the zero-path baseline as the best model.
    # ----------------------------------------------------------
    initial_val = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        rollout_weights=rollout_weights,
        device=device,
        max_batches=args.max_val_batches,
    )

    best_val_loss = initial_val
    best_epoch = 0

    save_checkpoint(
        best_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        val_loss=initial_val,
        best_val_loss=best_val_loss,
        args=args,
        m6_sha256=m6_sha,
        trainable_names=trainable_names,
    )

    print()
    print("========== EPOCH 0 / PURE-M6 START ==========")
    print(f"VAL_LOSS={initial_val:.12f}")
    print(
        f"alpha_A={model.alpha_a().item():+.8e} "
        f"alpha_B={model.alpha_b().item():+.8e}"
    )
    print("BEST_EPOCH=0")

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()

        # Backbone is frozen; keep it explicitly in eval mode.
        model.m6.eval()

        train_loss_sum = 0.0
        train_samples = 0

        for batch_idx, (context_norm, y_seq_norm) in enumerate(
            train_loader
        ):
            if (
                args.max_train_batches is not None
                and batch_idx >= args.max_train_batches
            ):
                break

            context_norm = context_norm.to(device)
            y_seq_norm = y_seq_norm.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, _ = autoregressive_multistep_loss(
                model=model,
                context_norm=context_norm,
                y_seq_norm=y_seq_norm,
                criterion=criterion,
                rollout_weights=rollout_weights,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()

            bs = context_norm.shape[0]

            train_loss_sum += float(loss.item()) * bs
            train_samples += bs

        if train_samples == 0:
            raise RuntimeError(
                "Training loader produced zero samples."
            )

        train_loss = train_loss_sum / train_samples

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            rollout_weights=rollout_weights,
            device=device,
            max_batches=args.max_val_batches,
        )

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch

            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                args=args,
                m6_sha256=m6_sha,
                trainable_names=trainable_names,
            )

        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            args=args,
            m6_sha256=m6_sha,
            trainable_names=trainable_names,
        )

        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] "
            f"train={train_loss:.12f} "
            f"val={val_loss:.12f} "
            f"alpha_A={model.alpha_a().item():+.8e} "
            f"alpha_B={model.alpha_b().item():+.8e} "
            f"best_epoch={best_epoch}"
            + ("  ✅ BEST" if improved else "")
        )

    print()
    print("========== M10-PRE COMPLETE ==========")
    print("PATH_MODE:", args.path_mode)
    print("INITIAL_VAL:", f"{initial_val:.12f}")
    print("BEST_VAL:", f"{best_val_loss:.12f}")
    print("BEST_EPOCH:", best_epoch)
    print("BEST_CHECKPOINT:", best_path)
    print("LAST_CHECKPOINT:", last_path)


if __name__ == "__main__":
    main()
