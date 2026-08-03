"""
Train M8-Gate-O5-H4: gate-only lightweight ablation.

Protocol
--------
- Initialize from an M8-A-O5 checkpoint.
- Freeze all existing M8-A parameters.
- Train only module_gate.weight and module_gate.bias.
- Keep static FieldCoupling.
- Keep ParameterToken.
- No PDE loss.
- H4 autoregressive delta training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from datasets.rbc_dataset import RBCDataset
from models.operators.fno2d_m8_module_gate import (
    M8ModuleGateFNO2d,
)
from training.metrics import FieldWiseRelativeL2Loss


class MultiStepDeltaParamDataset(Dataset):
    """Convert RBCDataset samples to context/y-sequence/parameter."""

    def __init__(
        self,
        base_dataset,
        context_length: int = 4,
    ) -> None:
        self.base_dataset = base_dataset
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _parse_sample(self, sample):
        x_norm = None
        y_seq_norm = None
        param = None

        if isinstance(sample, dict):
            for key in [
                "x_norm",
                "x",
                "context",
                "context_norm",
            ]:
                if key in sample:
                    x_norm = sample[key]
                    break

            for key in [
                "y_seq_norm",
                "y_seq",
                "target_sequence",
                "sequence",
            ]:
                if key in sample:
                    y_seq_norm = sample[key]
                    break

            for key in [
                "param",
                "params",
                "parameter",
                "parameters",
            ]:
                if key in sample:
                    param = sample[key]
                    break

        elif isinstance(sample, (tuple, list)):
            for obj in sample:
                if torch.is_tensor(obj):
                    if (
                        obj.ndim == 3
                        and obj.shape[0]
                        == self.context_length * 4
                    ):
                        x_norm = obj

                    elif (
                        obj.ndim == 4
                        and obj.shape[1] == 4
                    ):
                        y_seq_norm = obj

                    elif (
                        obj.ndim == 1
                        and obj.numel() == 2
                    ):
                        param = obj

                elif (
                    isinstance(obj, (tuple, list))
                    and len(obj) == 2
                ):
                    try:
                        candidate = torch.tensor(
                            obj,
                            dtype=torch.float32,
                        )

                        if (
                            candidate.ndim == 1
                            and candidate.numel() == 2
                        ):
                            param = candidate

                    except Exception:
                        pass

        else:
            raise TypeError(
                "Unsupported sample type: "
                f"{type(sample)}"
            )

        if (
            x_norm is None
            or y_seq_norm is None
            or param is None
        ):
            raise RuntimeError(
                "无法解析 x_norm / y_seq_norm / param。"
            )

        if not torch.is_tensor(param):
            param = torch.tensor(
                param,
                dtype=torch.float32,
            )

        return (
            x_norm,
            y_seq_norm,
            param.float(),
        )

    def __getitem__(self, index):
        sample = self.base_dataset[index]

        (
            x_norm,
            y_seq_norm,
            param,
        ) = self._parse_sample(sample)

        channels, height, width = x_norm.shape

        expected_channels = self.context_length * 4

        if channels != expected_channels:
            raise RuntimeError(
                f"x_norm channels={channels}, "
                f"expected={expected_channels}"
            )

        context_norm = x_norm.view(
            self.context_length,
            4,
            height,
            width,
        )

        return (
            context_norm,
            y_seq_norm,
            param,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train M8-Gate-O5-H4 with only three "
            "module-gate parameters trainable."
        )
    )

    parser.add_argument("--split", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--init_ckpt", required=True)
    parser.add_argument("--run_name", required=True)

    parser.add_argument(
        "--ckpt_dir",
        default="checkpoints/tuning/m8_gate_o5",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gate_lr",
        type=float,
        default=6e-4,
        help="O5 learning rate for the three gate parameters.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--eta_min",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--scheduler_t_max",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--rollout_weights",
        default="1.0,0.8,0.6,0.4",
    )

    parser.add_argument(
        "--gate_scale",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
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
        "--dry_run",
        action="store_true",
        help="Run training/validation without saving checkpoints.",
    )

    return parser.parse_args()


def resolve_path(
    project_root: str,
    path: str,
) -> str:
    if os.path.isabs(path):
        return path

    return os.path.abspath(
        os.path.join(project_root, path)
    )


def make_rollout_weights(
    text: str,
    rollout_steps: int,
) -> torch.Tensor:
    values = [
        float(value.strip())
        for value in text.split(",")
        if value.strip()
    ]

    if len(values) != rollout_steps:
        raise ValueError(
            "rollout_weights数量必须等于rollout_steps："
            f"{values} vs {rollout_steps}"
        )

    if any(
        not math.isfinite(value) or value <= 0
        for value in values
    ):
        raise ValueError(
            f"rollout_weights必须是有限正数：{values}"
        )

    return torch.tensor(
        values,
        dtype=torch.float32,
    )


def make_scheduler(
    optimizer,
    initial_lr: float,
    eta_min: float,
    t_max: int,
):
    if initial_lr <= 0:
        raise ValueError("initial_lr must be positive")

    if eta_min < 0 or eta_min > initial_lr:
        raise ValueError(
            "eta_min must be in [0, initial_lr]"
        )

    if t_max <= 0:
        raise ValueError("t_max must be positive")

    minimum_ratio = eta_min / initial_lr

    def lr_lambda(epoch):
        progress = (
            min(max(epoch, 0), t_max)
            / float(t_max)
        )

        return (
            minimum_ratio
            + 0.5
            * (1.0 - minimum_ratio)
            * (
                1.0
                + math.cos(math.pi * progress)
            )
        )

    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


def autoregressive_multistep_loss(
    model,
    context_norm,
    y_seq_norm,
    param,
    criterion,
    rollout_weights,
):
    batch_size, context_length, channels, height, width = (
        context_norm.shape
    )

    rollout_steps = y_seq_norm.shape[1]

    if context_length != 4 or channels != 4:
        raise RuntimeError(
            "Expected context [B,4,4,H,W], "
            f"got {tuple(context_norm.shape)}"
        )

    if rollout_steps != len(rollout_weights):
        raise RuntimeError(
            "rollout step count mismatch"
        )

    context = context_norm
    total_loss = 0.0
    step_losses = []

    for step in range(rollout_steps):
        model_input = context.reshape(
            batch_size,
            context_length * channels,
            height,
            width,
        )

        pred_delta_norm = model(
            model_input,
            param,
        )

        current_state_norm = context[:, -1]
        pred_next_norm = (
            current_state_norm
            + pred_delta_norm
        )

        gt_next_norm = y_seq_norm[:, step]

        step_loss = criterion(
            pred_next_norm,
            gt_next_norm,
        )

        total_loss = (
            total_loss
            + rollout_weights[step] * step_loss
        )

        step_losses.append(
            step_loss.detach()
        )

        context = torch.cat(
            [
                context[:, 1:],
                pred_next_norm.unsqueeze(1),
            ],
            dim=1,
        )

    total_loss = (
        total_loss / rollout_weights.sum()
    )

    return total_loss, step_losses


def get_git_commit(
    project_root: str,
) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    except Exception:
        return "unknown"


def make_gate_probe() -> torch.Tensor:
    log_pr_low = -0.3010299956639812
    log_pr_high = 0.3010299956639812

    return torch.tensor(
        [
            [6.0, log_pr_low],
            [6.0, log_pr_high],
            [8.0, log_pr_low],
            [8.0, log_pr_high],
        ],
        dtype=torch.float32,
    )


def summarize_gate(
    model,
    param_probe,
):
    model_device = next(
        model.parameters()
    ).device

    model_dtype = next(
        model.parameters()
    ).dtype

    probe = param_probe.to(
        device=model_device,
        dtype=model_dtype,
    )

    with torch.no_grad():
        (
            gate_score,
            token_weight,
            coupling_weight,
        ) = model.module_weights(probe)

    gate_linear_weight = (
        model.module_gate.weight
        .detach()
        .cpu()
        .reshape(-1)
        .tolist()
    )

    gate_linear_bias = float(
        model.module_gate.bias
        .detach()
        .cpu()
        .item()
    )

    return {
        "score": gate_score.detach().cpu().tolist(),
        "token_weight": (
            token_weight.detach().cpu().tolist()
        ),
        "coupling_weight": (
            coupling_weight.detach().cpu().tolist()
        ),
        "linear_weight": gate_linear_weight,
        "linear_bias": gate_linear_bias,
    }


def main():
    args = parse_args()

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    split_path = resolve_path(
        project_root,
        args.split,
    )

    stats_path = resolve_path(
        project_root,
        args.stats,
    )

    init_ckpt_path = resolve_path(
        project_root,
        args.init_ckpt,
    )

    ckpt_dir = resolve_path(
        project_root,
        args.ckpt_dir,
    )

    if not os.path.exists(split_path):
        raise FileNotFoundError(split_path)

    if not os.path.exists(stats_path):
        raise FileNotFoundError(stats_path)

    if not os.path.exists(init_ckpt_path):
        raise FileNotFoundError(init_ckpt_path)

    if args.rollout_steps != 4:
        raise ValueError(
            "当前Gate-only实验必须使用rollout_steps=4"
        )

    rollout_weights_cpu = make_rollout_weights(
        args.rollout_weights,
        args.rollout_steps,
    )

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as file:
        split_config = json.load(file)

    train_base_dataset = RBCDataset(
        split_config=split_config["train"],
        normalize=True,
        stats_path=stats_path,
        return_sequence=True,
        target_steps=args.rollout_steps,
        return_params=True,
    )

    val_base_dataset = RBCDataset(
        split_config=split_config["val"],
        normalize=True,
        stats_path=stats_path,
        return_sequence=True,
        target_steps=args.rollout_steps,
        return_params=True,
    )

    train_dataset = MultiStepDeltaParamDataset(
        train_base_dataset,
        context_length=4,
    )

    val_dataset = MultiStepDeltaParamDataset(
        val_base_dataset,
        context_length=4,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(
        init_ckpt_path,
        map_location="cpu",
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    checkpoint_model_config = checkpoint.get(
        "model_config",
        {},
    )

    checkpoint_mode = checkpoint_model_config.get(
        "coupling_mode",
        "static",
    )

    if checkpoint_mode != "static":
        raise RuntimeError(
            "Gate-only必须从M8-A static checkpoint初始化，"
            f"当前coupling_mode={checkpoint_mode}"
        )

    model_kwargs = {
        "in_channels": 16,
        "out_channels": 4,
        "modes1": 16,
        "modes2": 16,
        "width": 32,
        "context_length": 4,
        "num_fields": 4,
        "field_width": None,
        "coupling_hidden_channels": 8,
        "coupling_dropout": 0.0,
        "coupling_init_gate": -4.0,
        "coupling_use_norm": True,
        "coupling_param_hidden_dim": 64,
        "coupling_condition_scale": 0.10,
        "token_hidden_dim": 64,
        "alpha_token": 1.0,
        "gate_scale": args.gate_scale,
    }

    supported_checkpoint_keys = {
        "in_channels",
        "out_channels",
        "modes1",
        "modes2",
        "width",
        "context_length",
        "num_fields",
        "field_width",
        "coupling_hidden_channels",
        "coupling_dropout",
        "coupling_init_gate",
        "coupling_use_norm",
        "coupling_param_hidden_dim",
        "coupling_condition_scale",
        "token_hidden_dim",
        "alpha_token",
    }

    for key in supported_checkpoint_keys:
        if key in checkpoint_model_config:
            model_kwargs[key] = (
                checkpoint_model_config[key]
            )

    model = M8ModuleGateFNO2d(
        **model_kwargs,
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    expected_missing = {
        "module_gate.weight",
        "module_gate.bias",
    }

    if set(missing_keys) != expected_missing:
        raise RuntimeError(
            "出现非预期missing keys："
            f"{missing_keys}"
        )

    if unexpected_keys:
        raise RuntimeError(
            "出现unexpected keys："
            f"{unexpected_keys}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in model.module_gate.parameters():
        parameter.requires_grad_(True)

    trainable_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    expected_trainable_names = {
        "module_gate.weight",
        "module_gate.bias",
    }

    if set(trainable_names) != expected_trainable_names:
        raise RuntimeError(
            "Gate-only冻结失败："
            f"{trainable_names}"
        )

    if trainable_parameter_count != 3:
        raise RuntimeError(
            "Gate-only可训练参数应为3，"
            f"实际={trainable_parameter_count}"
        )

    model = model.to(device)

    criterion = FieldWiseRelativeL2Loss()

    gate_parameters = [
        parameter
        for parameter in model.module_gate.parameters()
        if parameter.requires_grad
    ]

    optimizer = optim.AdamW(
        gate_parameters,
        lr=args.gate_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = make_scheduler(
        optimizer=optimizer,
        initial_lr=args.gate_lr,
        eta_min=args.eta_min,
        t_max=args.scheduler_t_max,
    )

    rollout_weights = rollout_weights_cpu.to(device)
    param_probe = make_gate_probe()

    print(
        "🚀 [M8-Gate-O5-H4 | Gate-only] "
        f"启动训练 | 设备: {device}"
    )
    print("👉 实验定位: 轻量消融版")
    print("👉 Base: trained M8-A-O5 checkpoint")
    print("👉 Frozen: all M8-A parameters")
    print("👉 Trainable: module_gate only")
    print("👉 Static 12-edge FieldCoupling")
    print("👉 Existing ParameterToken")
    print("👉 No PDE loss")
    print("👉 H4 autoregressive delta training")
    print(f"👉 Gate scale: {args.gate_scale}")
    print(f"👉 Gate LR: {args.gate_lr:.2e}")
    print(f"👉 Trainable names: {trainable_names}")
    print(
        "👉 Trainable parameter count: "
        f"{trainable_parameter_count}"
    )
    print(f"📌 Split: {split_path}")
    print(f"📌 Stats: {stats_path}")
    print(f"📌 Init checkpoint: {init_ckpt_path}")
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    print(f"📌 Dry run: {args.dry_run}")

    initial_summary = summarize_gate(
        model,
        param_probe,
    )

    print("🔎 Initial gate summary:")
    print(
        "   linear weight:",
        initial_summary["linear_weight"],
    )
    print(
        "   linear bias:",
        initial_summary["linear_bias"],
    )
    print(
        "   scores:",
        initial_summary["score"],
    )
    print(
        "   token weights:",
        initial_summary["token_weight"],
    )
    print(
        "   coupling weights:",
        initial_summary["coupling_weight"],
    )

    best_val_loss = float("inf")
    top3_checkpoints = []
    start_time = time.time()

    if not args.dry_run:
        os.makedirs(
            ckpt_dir,
            exist_ok=True,
        )

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_num_samples = 0
        train_num_batches = 0
        train_grad_norm_sum = 0.0

        train_step_sum = torch.zeros(
            args.rollout_steps,
            dtype=torch.float64,
        )

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):
            (
                context_norm,
                y_seq_norm,
                param,
            ) = batch

            context_norm = context_norm.to(device)
            y_seq_norm = y_seq_norm.to(device)
            param = param.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, step_losses = (
                autoregressive_multistep_loss(
                    model=model,
                    context_norm=context_norm,
                    y_seq_norm=y_seq_norm,
                    param=param,
                    criterion=criterion,
                    rollout_weights=rollout_weights,
                )
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "训练loss出现NaN或Inf"
                )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                gate_parameters,
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = context_norm.shape[0]

            train_loss_sum += (
                float(loss.item()) * batch_size
            )
            train_num_samples += batch_size
            train_num_batches += 1
            train_grad_norm_sum += float(
                grad_norm.item()
            )

            for step, step_loss in enumerate(
                step_losses
            ):
                train_step_sum[step] += (
                    float(step_loss.item())
                    * batch_size
                )

            if (
                args.max_train_batches is not None
                and batch_index
                >= args.max_train_batches
            ):
                break

        train_loss = (
            train_loss_sum
            / max(train_num_samples, 1)
        )

        train_grad_norm = (
            train_grad_norm_sum
            / max(train_num_batches, 1)
        )

        train_step_losses = (
            train_step_sum
            / max(train_num_samples, 1)
        )

        model.eval()

        val_loss_sum = 0.0
        val_num_samples = 0

        val_step_sum = torch.zeros(
            args.rollout_steps,
            dtype=torch.float64,
        )

        with torch.no_grad():
            for batch_index, batch in enumerate(
                val_loader,
                start=1,
            ):
                (
                    context_norm,
                    y_seq_norm,
                    param,
                ) = batch

                context_norm = context_norm.to(device)
                y_seq_norm = y_seq_norm.to(device)
                param = param.to(device)

                loss, step_losses = (
                    autoregressive_multistep_loss(
                        model=model,
                        context_norm=context_norm,
                        y_seq_norm=y_seq_norm,
                        param=param,
                        criterion=criterion,
                        rollout_weights=rollout_weights,
                    )
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "验证loss出现NaN或Inf"
                    )

                batch_size = context_norm.shape[0]

                val_loss_sum += (
                    float(loss.item()) * batch_size
                )
                val_num_samples += batch_size

                for step, step_loss in enumerate(
                    step_losses
                ):
                    val_step_sum[step] += (
                        float(step_loss.item())
                        * batch_size
                    )

                if (
                    args.max_val_batches is not None
                    and batch_index
                    >= args.max_val_batches
                ):
                    break

        val_loss = (
            val_loss_sum
            / max(val_num_samples, 1)
        )

        val_step_losses = (
            val_step_sum
            / max(val_num_samples, 1)
        )

        gate_summary = summarize_gate(
            model,
            param_probe,
        )

        val_step_message = " | ".join(
            [
                f"Val t+{step + 1}: "
                f"{val_step_losses[step]:.6f}"
                for step in range(
                    args.rollout_steps
                )
            ]
        )

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"{val_step_message} | "
            f"Grad: {train_grad_norm:.6e} | "
            f"GateW: "
            f"{gate_summary['linear_weight']} | "
            f"GateB: "
            f"{gate_summary['linear_bias']:.6f} | "
            f"Scores: "
            f"{[round(v, 5) for v in gate_summary['score']]} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if not args.dry_run:
            checkpoint_payload = {
                "checkpoint_format_version": 1,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "scheduler_state_dict": (
                    scheduler.state_dict()
                ),
                "model_config": model_kwargs,
                "train_config": {
                    "experiment": (
                        "M8-Gate-O5-H4-GateOnly"
                    ),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "gate_lr": args.gate_lr,
                    "weight_decay": (
                        args.weight_decay
                    ),
                    "eta_min": args.eta_min,
                    "scheduler_t_max": (
                        args.scheduler_t_max
                    ),
                    "rollout_steps": (
                        args.rollout_steps
                    ),
                    "rollout_weights": (
                        rollout_weights_cpu.tolist()
                    ),
                    "gate_scale": args.gate_scale,
                    "seed": args.seed,
                    "trainable_parameter_count": (
                        trainable_parameter_count
                    ),
                    "frozen_base": True,
                },
                "data_config": {
                    "split_path": split_path,
                    "stats_path": stats_path,
                    "init_checkpoint": (
                        init_ckpt_path
                    ),
                },
                "gate_summary": gate_summary,
                "val_loss": val_loss,
                "git_commit": get_git_commit(
                    project_root
                ),
                "run_name": args.run_name,
            }

            qualifies_for_top3 = (
                len(top3_checkpoints) < 3
                or val_loss
                < top3_checkpoints[-1]["val_loss"]
            )

            if qualifies_for_top3:
                top3_path = os.path.join(
                    ckpt_dir,
                    (
                        f"{args.run_name}"
                        f"_valtop_epoch_{epoch:02d}.pth"
                    ),
                )

                torch.save(
                    checkpoint_payload,
                    top3_path,
                )

                top3_checkpoints.append(
                    {
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "path": top3_path,
                    }
                )

                top3_checkpoints.sort(
                    key=lambda item: (
                        item["val_loss"],
                        item["epoch"],
                    )
                )

                while len(top3_checkpoints) > 3:
                    removed = top3_checkpoints.pop()

                    if os.path.exists(
                        removed["path"]
                    ):
                        os.remove(
                            removed["path"]
                        )

            if epoch % 5 == 0:
                epoch_path = os.path.join(
                    ckpt_dir,
                    (
                        f"{args.run_name}"
                        f"_epoch_{epoch:02d}.pth"
                    ),
                )

                torch.save(
                    checkpoint_payload,
                    epoch_path,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                best_path = os.path.join(
                    ckpt_dir,
                    f"{args.run_name}_best.pth",
                )

                checkpoint_payload[
                    "best_val_loss"
                ] = best_val_loss

                torch.save(
                    checkpoint_payload,
                    best_path,
                )

                print(
                    "   🌟 New best: "
                    f"{best_path}"
                )

        best_val_loss = min(
            best_val_loss,
            val_loss,
        )

        scheduler.step()

    elapsed_minutes = (
        time.time() - start_time
    ) / 60.0

    final_summary = summarize_gate(
        model,
        param_probe,
    )

    print()
    print("✅ M8-Gate Gate-only运行完成")
    print(
        f"⏱️ 总耗时: {elapsed_minutes:.2f}分钟"
    )
    print(
        "📌 Final gate weight:",
        final_summary["linear_weight"],
    )
    print(
        "📌 Final gate bias:",
        final_summary["linear_bias"],
    )
    print(
        "📌 Final gate scores:",
        final_summary["score"],
    )

    if args.dry_run:
        print(
            "📌 Dry run模式：未保存checkpoint"
        )


if __name__ == "__main__":
    main()
