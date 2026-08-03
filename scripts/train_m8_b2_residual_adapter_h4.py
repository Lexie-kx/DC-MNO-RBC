import os
import sys
import json
import time
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from datasets.rbc_dataset import RBCDataset
from training.metrics import FieldWiseRelativeL2Loss

from models.operators.fno2d_m8_full_conditioned import (
    M8FullConditionedFNO2d,
)
from models.operators.fno2d_m8_b2_residual import (
    M8B2ResidualAdapterFNO2d,
)

from scripts.train_m8_full_conditioned_h4 import (
    MultiStepDeltaParamDataset,
    make_rollout_weights,
    autoregressive_multistep_loss,
    make_shared_cosine_scheduler,
    get_git_commit,
)


TRAINABLE_PREFIXES = (
    "field_coupling.param_conditioner",
    "field_coupling.adapter_gate",
)

EXPECTED_NEW_KEYS = {
    "field_coupling.adapter_gate.0.weight",
    "field_coupling.adapter_gate.0.bias",
    "field_coupling.adapter_gate.2.weight",
    "field_coupling.adapter_gate.2.bias",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train fair M8-A2/M8-B2 residual coupling adapters "
            "from the same frozen M8-A checkpoint."
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--stats",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--init_ckpt",
        type=str,
        required=True,
        help="Trained M8-A-O5 checkpoint.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--adapter_mode",
        type=str,
        required=True,
        choices=[
            "static_adapter",
            "parameter_adapter",
        ],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--adapter_lr",
        type=float,
        default=6e-4,
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
        default=10,
    )

    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--rollout_weights",
        type=str,
        default="1.0,0.8,0.6,0.4",
    )

    parser.add_argument(
        "--adapter_gate_hidden_dim",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--adapter_gate_init_bias",
        type=float,
        default=-2.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        "--no_wandb",
        action="store_true",
    )

    return parser.parse_args()


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path

    return os.path.abspath(
        os.path.join(project_root, path)
    )


def set_deterministic_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_datasets(
    split_path,
    stats_path,
    rollout_steps,
):
    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as file:
        split_config = json.load(file)

    train_base = RBCDataset(
        split_config=split_config["train"],
        normalize=True,
        stats_path=stats_path,
        return_sequence=True,
        target_steps=rollout_steps,
        return_params=True,
    )

    val_base = RBCDataset(
        split_config=split_config["val"],
        normalize=True,
        stats_path=stats_path,
        return_sequence=True,
        target_steps=rollout_steps,
        return_params=True,
    )

    train_dataset = MultiStepDeltaParamDataset(
        train_base,
        context_length=4,
    )

    val_dataset = MultiStepDeltaParamDataset(
        val_base,
        context_length=4,
    )

    return train_dataset, val_dataset


def load_models(
    checkpoint_path,
    adapter_mode,
    adapter_gate_hidden_dim,
    adapter_gate_init_bias,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        not isinstance(checkpoint, dict)
        or "model_state_dict" not in checkpoint
        or "model_config" not in checkpoint
    ):
        raise RuntimeError(
            "❌ init_ckpt 必须是包含 model_state_dict "
            "和 model_config 的 M8-A checkpoint"
        )

    state_dict = checkpoint["model_state_dict"]
    base_config = dict(checkpoint["model_config"])

    if base_config.get("coupling_mode") != "static":
        raise RuntimeError(
            "❌ M8-B2 必须从 M8-A static checkpoint 初始化，"
            f"当前 mode={base_config.get('coupling_mode')}"
        )

    teacher = M8FullConditionedFNO2d(
        **base_config
    ).to(device)

    teacher.load_state_dict(
        state_dict,
        strict=True,
    )
    teacher.eval()

    common_config = dict(base_config)
    common_config.pop("coupling_mode", None)

    model = M8B2ResidualAdapterFNO2d(
        **common_config,
        adapter_mode=adapter_mode,
        adapter_gate_hidden_dim=(
            adapter_gate_hidden_dim
        ),
        adapter_gate_init_bias=(
            adapter_gate_init_bias
        ),
    ).to(device)

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print("🔎 M8-A → M8-B2 checkpoint loading")
    print(f"   missing_keys: {missing_keys}")
    print(f"   unexpected_keys: {unexpected_keys}")

    if set(missing_keys) != EXPECTED_NEW_KEYS:
        raise RuntimeError(
            "❌ missing_keys 不符合预期："
            f"{missing_keys}"
        )

    if unexpected_keys:
        raise RuntimeError(
            "❌ unexpected_keys 不为空："
            f"{unexpected_keys}"
        )

    return teacher, model, base_config


def freeze_base_and_enable_adapter(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable_names = []
    trainable_parameters = []

    for name, parameter in model.named_parameters():
        if any(
            name == prefix
            or name.startswith(prefix + ".")
            for prefix in TRAINABLE_PREFIXES
        ):
            parameter.requires_grad_(True)
            trainable_names.append(name)
            trainable_parameters.append(parameter)

    if not trainable_parameters:
        raise RuntimeError(
            "❌ 没有找到可训练 adapter 参数"
        )

    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not any(
            name == prefix
            or name.startswith(prefix + ".")
            for prefix in TRAINABLE_PREFIXES
        )
    ]

    if unexpected_trainable:
        raise RuntimeError(
            "❌ 出现非预期可训练参数："
            f"{unexpected_trainable}"
        )

    return trainable_names, trainable_parameters


def verify_initial_alignment(
    teacher,
    model,
    sample_context,
    sample_param,
    device,
):
    teacher.eval()
    model.eval()

    context = sample_context[:2].to(device)
    param = sample_param[:2].to(device)

    batch_size, steps, fields, height, width = (
        context.shape
    )

    model_input = context.reshape(
        batch_size,
        steps * fields,
        height,
        width,
    )

    with torch.no_grad():
        teacher_output = teacher(
            model_input,
            param,
            coupling_mode="static",
        )

        adapter_output = model(
            model_input,
            param,
        )

        initial_delta = (
            model.field_coupling
            .conditioned_delta_matrix(param)
        )

    output_diff = (
        teacher_output - adapter_output
    ).abs().max().item()

    delta_max = (
        initial_delta.abs().max().item()
    )

    print("🔎 Frozen-base initial alignment")
    print(
        "   Adapter - M8-A max diff: "
        f"{output_diff:.12e}"
    )
    print(
        "   Initial residual delta abs max: "
        f"{delta_max:.12e}"
    )

    if output_diff >= 1e-7:
        raise RuntimeError(
            "❌ Adapter 初始输出与 M8-A 不一致"
        )

    if delta_max >= 1e-7:
        raise RuntimeError(
            "❌ Adapter 初始动态残差不为零"
        )

    print(
        "✅ Adapter 初始输出与 M8-A 严格一致"
    )


def summarize_adapter(model, param_probe):
    model.eval()

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    param_probe = param_probe.to(
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        gate = (
            model.field_coupling
            .adapter_gate_values(param_probe)
        )

        delta = (
            model.field_coupling
            .conditioned_delta_matrix(param_probe)
        )

    return {
        "adapter_gate_mean": (
            gate.mean().item()
        ),
        "adapter_gate_min": (
            gate.min().item()
        ),
        "adapter_gate_max": (
            gate.max().item()
        ),
        "residual_delta_abs_mean": (
            delta.abs().mean().item()
        ),
        "residual_delta_abs_max": (
            delta.abs().max().item()
        ),
        "condition_diversity": (
            delta.std(
                dim=0,
                unbiased=False,
            ).mean().item()
        ),
    }


def run_epoch(
    model,
    loader,
    criterion,
    rollout_weights,
    device,
    optimizer=None,
    trainable_parameters=None,
    max_batches=None,
):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    total_grad_norm = 0.0
    num_batches = 0

    step_loss_sum = torch.zeros(
        len(rollout_weights),
        dtype=torch.float64,
    )

    context_manager = (
        torch.enable_grad()
        if is_training
        else torch.no_grad()
    )

    with context_manager:
        for batch_index, (
            context_norm,
            y_seq_norm,
            param,
        ) in enumerate(loader, start=1):
            context_norm = context_norm.to(device)
            y_seq_norm = y_seq_norm.to(device)
            param = param.to(device)

            if is_training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            loss, step_losses = (
                autoregressive_multistep_loss(
                    model=model,
                    context_norm=context_norm,
                    y_seq_norm=y_seq_norm,
                    param=param,
                    criterion=criterion,
                    rollout_weights=(
                        rollout_weights
                    ),
                )
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "❌ 检测到 NaN/Inf loss"
                )

            if is_training:
                loss.backward()

                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=1.0,
                    )
                )

                optimizer.step()

                total_grad_norm += (
                    grad_norm.item()
                )

            batch_size = context_norm.shape[0]

            total_loss += (
                loss.item() * batch_size
            )
            total_samples += batch_size
            num_batches += 1

            for step_index, step_loss in enumerate(
                step_losses
            ):
                step_loss_sum[step_index] += (
                    step_loss.item()
                    * batch_size
                )

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

    average_loss = (
        total_loss / max(total_samples, 1)
    )

    average_step_losses = (
        step_loss_sum / max(total_samples, 1)
    )

    average_grad_norm = (
        total_grad_norm / max(num_batches, 1)
    )

    return (
        average_loss,
        average_step_losses,
        average_grad_norm,
    )


def make_checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    val_loss,
    best_val_loss,
    args,
    base_model_config,
    trainable_names,
    split_path,
    stats_path,
    init_ckpt,
    git_commit,
):
    model_config = dict(base_model_config)
    model_config.pop("coupling_mode", None)

    model_config.update(
        {
            "adapter_mode": args.adapter_mode,
            "adapter_gate_hidden_dim": (
                args.adapter_gate_hidden_dim
            ),
            "adapter_gate_init_bias": (
                args.adapter_gate_init_bias
            ),
        }
    )

    return {
        "checkpoint_format_version": 3,
        "model_class": (
            "M8B2ResidualAdapterFNO2d"
        ),
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
        ),
        "model_config": model_config,
        "base_model_config": (
            base_model_config
        ),
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "experiment": (
            "M8-A2-StaticResidual-Control"
            if args.adapter_mode
            == "static_adapter"
            else
            "M8-B2-ParameterResidualAdapter"
        ),
        "experiment_role": (
            "fair frozen-base control"
            if args.adapter_mode
            == "static_adapter"
            else
            "formal structural candidate"
        ),
        "adapter_mode": args.adapter_mode,
        "adapter_training_stage": (
            "frozen_m8_a_base"
        ),
        "trainable_parameter_names": (
            trainable_names
        ),
        "train_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "adapter_lr": (
                args.adapter_lr
            ),
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
                args.rollout_weights
            ),
            "seed": args.seed,
            "grad_clip": 1.0,
        },
        "data_config": {
            "split_path": split_path,
            "stats_path": stats_path,
            "init_ckpt": init_ckpt,
        },
        "git_commit": git_commit,
    }


def main():
    args = parse_args()

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )

    split_path = resolve_path(
        project_root,
        args.split,
    )
    stats_path = resolve_path(
        project_root,
        args.stats,
    )
    init_ckpt = resolve_path(
        project_root,
        args.init_ckpt,
    )
    ckpt_dir = resolve_path(
        project_root,
        args.ckpt_dir,
    )

    for path in [
        split_path,
        stats_path,
        init_ckpt,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ 文件不存在: {path}"
            )

    if args.rollout_steps != 4:
        raise ValueError(
            "❌ 当前 B2 公平实验必须保持 "
            "rollout_steps=4"
        )

    os.makedirs(
        ckpt_dir,
        exist_ok=True,
    )

    set_deterministic_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    role = (
        "M8-A2-StaticResidual-Control"
        if args.adapter_mode
        == "static_adapter"
        else
        "M8-B2-ParameterResidualAdapter"
    )

    print(
        f"🚀 [{role}] 启动训练 | "
        f"设备: {device}"
    )
    print(
        "👉 实验定位: 正式候选结构公平验证"
    )
    print(
        "👉 Base: trained M8-A-O5 checkpoint"
    )
    print(
        "👉 Frozen: complete M8-A backbone "
        "and static coupling"
    )
    print(
        "👉 Trainable: param_conditioner "
        "+ adapter_gate only"
    )
    print(
        f"👉 Adapter mode: "
        f"{args.adapter_mode}"
    )
    print(
        "👉 No PDE loss"
    )
    print(
        f"👉 Random seed: {args.seed}"
    )
    print(
        f"📌 Init checkpoint: {init_ckpt}"
    )

    train_dataset, val_dataset = (
        build_datasets(
            split_path=split_path,
            stats_path=stats_path,
            rollout_steps=(
                args.rollout_steps
            ),
        )
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(
        f"📊 Train samples: "
        f"{len(train_dataset)}"
    )
    print(
        f"📊 Val samples:   "
        f"{len(val_dataset)}"
    )

    sample_context, _, sample_param = next(
        iter(train_loader)
    )

    teacher, model, base_model_config = (
        load_models(
            checkpoint_path=init_ckpt,
            adapter_mode=args.adapter_mode,
            adapter_gate_hidden_dim=(
                args.adapter_gate_hidden_dim
            ),
            adapter_gate_init_bias=(
                args.adapter_gate_init_bias
            ),
            device=device,
        )
    )

    trainable_names, trainable_parameters = (
        freeze_base_and_enable_adapter(model)
    )

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    print(
        f"🧮 Total params: "
        f"{total_params:,}"
    )
    print(
        f"🧮 Trainable adapter params: "
        f"{trainable_count:,}"
    )
    print("🧩 Trainable parameter tensors:")

    for name in trainable_names:
        print(f"   {name}")

    verify_initial_alignment(
        teacher=teacher,
        model=model,
        sample_context=sample_context,
        sample_param=sample_param,
        device=device,
    )

    del teacher

    criterion = FieldWiseRelativeL2Loss()

    rollout_weights_cpu = (
        make_rollout_weights(
            args.rollout_steps,
            args.rollout_weights,
        )
    )

    rollout_weights = (
        rollout_weights_cpu.to(device)
    )

    optimizer = optim.AdamW(
        [
            {
                "name": "adapter_only",
                "params": (
                    trainable_parameters
                ),
                "lr": args.adapter_lr,
                "weight_decay": (
                    args.weight_decay
                ),
                "initial_lr": (
                    args.adapter_lr
                ),
            }
        ]
    )

    scheduler = make_shared_cosine_scheduler(
        optimizer=optimizer,
        base_lr=args.adapter_lr,
        eta_min=args.eta_min,
        t_max=args.scheduler_t_max,
    )

    param_probe = torch.tensor(
        [
            [6.0, -0.30103],
            [6.0, 0.0],
            [7.0, -0.30103],
            [7.0, 0.0],
            [8.0, 0.30103],
        ],
        dtype=torch.float32,
    )

    initial_summary = summarize_adapter(
        model,
        param_probe,
    )

    print("🔎 Initial adapter summary:")

    for key, value in initial_summary.items():
        print(f"   {key}: {value:.8f}")

    best_val_loss = float("inf")
    top3 = []

    best_path = os.path.join(
        ckpt_dir,
        f"{args.run_name}_best.pth",
    )

    git_commit = get_git_commit(
        project_root
    )

    start_time = time.time()

    print(
        "\n🔥 开始冻结基础模型的 "
        "H4 adapter 训练..."
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        (
            train_loss,
            train_step_losses,
            grad_norm,
        ) = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            rollout_weights=(
                rollout_weights
            ),
            device=device,
            optimizer=optimizer,
            trainable_parameters=(
                trainable_parameters
            ),
            max_batches=(
                args.max_train_batches
            ),
        )

        (
            val_loss,
            val_step_losses,
            _,
        ) = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            rollout_weights=(
                rollout_weights
            ),
            device=device,
            optimizer=None,
            trainable_parameters=None,
            max_batches=(
                args.max_val_batches
            ),
        )

        adapter_summary = summarize_adapter(
            model,
            param_probe,
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        step_message = " | ".join(
            [
                (
                    f"Val t+{index + 1}: "
                    f"{value:.6f}"
                )
                for index, value
                in enumerate(
                    val_step_losses.tolist()
                )
            ]
        )

        print(
            f"Epoch [{epoch:03d}/"
            f"{args.epochs}] | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"{step_message} | "
            f"Grad Norm: {grad_norm:.4f} | "
            f"AdapterGateMean: "
            f"{adapter_summary['adapter_gate_mean']:.5f} | "
            f"ResidualDeltaMax: "
            f"{adapter_summary['residual_delta_abs_max']:.6f} | "
            f"ConditionDiversity: "
            f"{adapter_summary['condition_diversity']:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        is_new_best = (
            val_loss < best_val_loss
        )

        if is_new_best:
            best_val_loss = val_loss

        payload = make_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            args=args,
            base_model_config=(
                base_model_config
            ),
            trainable_names=(
                trainable_names
            ),
            split_path=split_path,
            stats_path=stats_path,
            init_ckpt=init_ckpt,
            git_commit=git_commit,
        )

        if is_new_best:
            torch.save(
                payload,
                best_path,
            )
            print(
                "   🌟 [New Best] "
                f"{best_path}"
            )

        top3_path = os.path.join(
            ckpt_dir,
            (
                f"{args.run_name}"
                f"_valtop_epoch_{epoch:02d}.pth"
            ),
        )

        top3.append(
            {
                "epoch": epoch,
                "val": val_loss,
                "path": top3_path,
            }
        )

        top3 = sorted(
            top3,
            key=lambda item: item["val"],
        )

        if len(top3) <= 3:
            torch.save(
                payload,
                top3_path,
            )
        else:
            removed = top3.pop(-1)

            if removed["epoch"] == epoch:
                pass
            else:
                torch.save(
                    payload,
                    top3_path,
                )

                if os.path.exists(
                    removed["path"]
                ):
                    os.remove(
                        removed["path"]
                    )

        print("   [Top-3 当前排名]")

        for rank, item in enumerate(
            top3,
            start=1,
        ):
            print(
                f"      #{rank}: "
                f"epoch={item['epoch']} | "
                f"val={item['val']:.6f}"
            )

        if epoch in {
            5,
            10,
            15,
            20,
        }:
            periodic_path = os.path.join(
                ckpt_dir,
                (
                    f"{args.run_name}"
                    f"_epoch_{epoch:02d}.pth"
                ),
            )

            torch.save(
                payload,
                periodic_path,
            )

            print(
                "   [*] 保存周期 checkpoint: "
                f"{periodic_path}"
            )

        scheduler.step()

    elapsed_minutes = (
        time.time() - start_time
    ) / 60.0

    print("\n✅ Adapter 训练完成")
    print(
        f"📌 Adapter mode: "
        f"{args.adapter_mode}"
    )
    print(
        f"📌 Best checkpoint: "
        f"{best_path}"
    )
    print(
        f"📌 Best validation: "
        f"{best_val_loss:.6f}"
    )
    print(
        f"⏱️ 总耗时: "
        f"{elapsed_minutes:.2f} 分钟"
    )


if __name__ == "__main__":
    main()
