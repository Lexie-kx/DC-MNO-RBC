import os
import sys
import json
import math
import subprocess
import time
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.rbc_dataset import RBCDataset
from models.operators.fno2d_m8_full_conditioned import M8FullConditionedFNO2d
from training.metrics import FieldWiseRelativeL2Loss


class MultiStepDeltaParamDataset(Dataset):
    """
    M8 FullStatic / ParamConditionedCoupling H4 dataset wrapper.

    base_dataset must return normalized context / y_seq / param.

    Expected final output:
        context_norm: [4, 4, H, W]
        y_seq_norm:  [S, 4, H, W]
        param:       [2] = [log10(Ra), log10(Pr)]

    Training protocol:
        model predicts delta_norm:
            pred_next_norm = current_state_norm + pred_delta_norm

        multi-step loss is computed between pred_next_norm and gt_next_norm.
    """

    def __init__(self, base_dataset, context_length=4):
        self.base_dataset = base_dataset
        self.context_length = context_length

    def __len__(self):
        return len(self.base_dataset)

    def _parse_sample(self, sample):
        x_norm = None
        y_seq_norm = None
        param = None

        if isinstance(sample, dict):
            for key in ["x_norm", "x", "context", "context_norm"]:
                if key in sample:
                    x_norm = sample[key]
                    break

            for key in ["y_seq_norm", "y_seq", "target_sequence", "sequence"]:
                if key in sample:
                    y_seq_norm = sample[key]
                    break

            for key in ["param", "params", "parameter", "parameters"]:
                if key in sample:
                    param = sample[key]
                    break

        elif isinstance(sample, (tuple, list)):
            for obj in sample:
                if torch.is_tensor(obj):
                    if obj.ndim == 3 and obj.shape[0] == self.context_length * 4:
                        x_norm = obj
                    elif obj.ndim == 4 and obj.shape[1] == 4:
                        y_seq_norm = obj
                    elif obj.ndim == 1 and obj.numel() == 2:
                        param = obj

                elif isinstance(obj, (tuple, list)) and len(obj) == 2:
                    try:
                        maybe_param = torch.tensor(obj, dtype=torch.float32)
                        if maybe_param.ndim == 1 and maybe_param.numel() == 2:
                            param = maybe_param
                    except Exception:
                        pass

        else:
            raise TypeError(
                f"Unsupported sample type from RBCDataset: {type(sample)}"
            )

        if x_norm is None or y_seq_norm is None or param is None:
            raise RuntimeError(
                "❌ 无法解析 RBCDataset 返回值中的 x_norm / y_seq_norm / param。\n"
                "请确认 RBCDataset 设置了 return_sequence=True 且 return_params=True。\n"
                f"sample type: {type(sample)}\n"
                f"sample repr: {repr(sample)[:500]}"
            )

        if not torch.is_tensor(param):
            param = torch.tensor(param, dtype=torch.float32)

        return x_norm, y_seq_norm, param.float()

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        x_norm, y_seq_norm, param = self._parse_sample(sample)

        c_total, h, w = x_norm.shape
        assert c_total == self.context_length * 4, (
            f"❌ x_norm channels should be {self.context_length * 4}, got {c_total}"
        )

        context_norm = x_norm.view(self.context_length, 4, h, w)

        return context_norm, y_seq_norm, param


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train M8-A FullStatic-H4 or M8-B "
            "ParamConditionedCoupling-H4 using Delta-H4 training."
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        default="data/splits/unseen_pr_split.json",
        help="Path to split json file, relative to project root or absolute path."
    )

    parser.add_argument(
        "--stats",
        type=str,
        default="data/stats/rbc_field_stats_unseen_pr.json",
        help="Path to field stats json file, relative to project root or absolute path."
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="m8_full_static_h4_unseen_pr",
        help="Run name for checkpoint and wandb."
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints/cross_param",
        help="Checkpoint directory, relative to project root or absolute path."
    )

    parser.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help=(
            "M6-FieldWiseEncoder-H4 checkpoint used to initialize shared weights. "
            "Only param_token.* and field_coupling.* may be missing."
        )
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
        help="M8 Full/Conditioned H4 training uses substantial rollout memory. Start with 4."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Base learning rate eta0.",
    )

    parser.add_argument(
        "--backbone_lr_mult",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--new_common_lr_mult",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dynamic_lr_mult",
        type=float,
        default=1.0,
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
        help="Cosine scheduler cycle length, independent of epochs.",
    )

    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=4,
        help="Number of autoregressive training steps. For this script, use 4."
    )

    parser.add_argument(
        "--rollout_weights",
        type=str,
        default="1.0,0.8,0.6,0.4",
        help=(
            "Comma-separated multi-step loss weights. "
            "The number of values must equal rollout_steps."
        ),
    )

    parser.add_argument(
        "--token_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension of ParameterToken MLP."
    )

    parser.add_argument(
        "--alpha_token",
        type=float,
        default=1.0,
        help="Scale applied to ParameterToken before FNO-block injection.",
    )

    parser.add_argument(
        "--coupling_mode",
        type=str,
        default="static",
        choices=["static", "parameter_conditioned"],
        help=(
            "static: M8-A FullStatic control; "
            "parameter_conditioned: M8-B formal candidate."
        ),
    )

    parser.add_argument(
        "--coupling_param_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension of conditioned-coupling MLP.",
    )

    parser.add_argument(
        "--coupling_condition_scale",
        type=float,
        default=0.10,
        help="Maximum scale of dynamic coupling correction.",
    )

    parser.add_argument(
        "--coupling_init_gate",
        type=float,
        default=-4.0,
        help="Initial residual coupling-gate logit.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed shared by M8-A and M8-B.",
    )

    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Debug only: limit number of training batches per epoch."
    )

    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Debug only: limit number of validation batches per epoch."
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging."
    )

    return parser.parse_args()


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


def make_rollout_weights(
    rollout_steps,
    rollout_weights_text=None,
):
    """Build and validate multi-step loss weights."""
    if rollout_weights_text is None:
        if rollout_steps == 4:
            values = [1.0, 0.8, 0.6, 0.4]
        else:
            values = torch.linspace(
                1.0,
                0.4,
                steps=rollout_steps,
                dtype=torch.float32,
            ).tolist()
    else:
        values = [
            float(value.strip())
            for value in rollout_weights_text.split(",")
            if value.strip()
        ]

    if len(values) != rollout_steps:
        raise ValueError(
            "rollout_weights 数量必须等于 rollout_steps："
            f"得到 {len(values)} 个权重，"
            f"rollout_steps={rollout_steps}"
        )

    if any(
        (not math.isfinite(value)) or value <= 0
        for value in values
    ):
        raise ValueError(
            f"rollout_weights 必须全部为有限正数，得到 {values}"
        )

    return torch.tensor(values, dtype=torch.float32)


def _name_matches_prefixes(name, prefixes):
    return any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in prefixes
    )


def build_optimizer_param_groups(
    model,
    base_lr,
    weight_decay,
    backbone_lr_mult,
    new_common_lr_mult,
    dynamic_lr_mult,
    coupling_mode,
):
    backbone_prefixes = (
        "field_encoders",
        "fusion",
        "conv0",
        "conv1",
        "conv2",
        "conv3",
        "w0",
        "w1",
        "w2",
        "w3",
        "mlp0",
        "mlp1",
    )

    new_common_prefixes = (
        "param_token",
        "field_coupling.norm",
        "field_coupling.pre_proj",
        "field_coupling.post_proj",
        "field_coupling.coupling_matrix",
        "field_coupling.residual_gate",
    )

    dynamic_prefix = "field_coupling.param_conditioner"

    grouped_params = {
        "backbone": [],
        "new_common": [],
        "dynamic_only": [],
    }
    grouped_names = {
        "backbone": [],
        "new_common": [],
        "dynamic_only": [],
    }

    frozen_dynamic_names = []
    unmatched = []
    multi_matched = []

    for name, param in model.named_parameters():
        if coupling_mode == "static" and (
            name == dynamic_prefix or name.startswith(dynamic_prefix + ".")
        ):
            param.requires_grad_(False)
            frozen_dynamic_names.append(name)
            continue

        if not param.requires_grad:
            continue

        matched_groups = []

        if _name_matches_prefixes(name, backbone_prefixes):
            matched_groups.append("backbone")

        if _name_matches_prefixes(name, new_common_prefixes):
            matched_groups.append("new_common")

        if name == dynamic_prefix or name.startswith(dynamic_prefix + "."):
            matched_groups.append("dynamic_only")

        if len(matched_groups) == 0:
            unmatched.append(name)
            continue

        if len(matched_groups) > 1:
            multi_matched.append((name, matched_groups))
            continue

        group_name = matched_groups[0]
        grouped_params[group_name].append(param)
        grouped_names[group_name].append(name)

    if unmatched:
        raise RuntimeError(
            "❌ 以下参数没有被分到任何 optimizer group:\n"
            + "\n".join(unmatched)
        )

    if multi_matched:
        raise RuntimeError(
            "❌ 以下参数被分到了多个 optimizer group:\n"
            + "\n".join(
                [f"{name}: {groups}" for name, groups in multi_matched]
            )
        )

    lr_mult_map = {
        "backbone": backbone_lr_mult,
        "new_common": new_common_lr_mult,
        "dynamic_only": dynamic_lr_mult,
    }

    active_groups = ["backbone", "new_common"]
    if coupling_mode == "parameter_conditioned":
        active_groups.append("dynamic_only")

    optimizer_param_groups = []
    summary = {}

    for group_name in active_groups:
        params = grouped_params[group_name]
        if len(params) == 0:
            raise RuntimeError(
                f"❌ optimizer group 为空：{group_name}"
            )

        group_lr = base_lr * lr_mult_map[group_name]

        optimizer_param_groups.append(
            {
                "name": group_name,
                "params": params,
                "lr": group_lr,
                "weight_decay": weight_decay,
                "initial_lr": group_lr,
            }
        )

        summary[group_name] = {
            "num_tensors": len(params),
            "num_params": sum(p.numel() for p in params),
            "lr": group_lr,
            "names": grouped_names[group_name],
        }

    return optimizer_param_groups, summary, frozen_dynamic_names


def make_shared_cosine_scheduler(
    optimizer,
    base_lr,
    eta_min,
    t_max,
):
    if base_lr <= 0:
        raise ValueError(f"base_lr 必须大于 0，得到 {base_lr}")

    if eta_min < 0:
        raise ValueError(f"eta_min 不能小于 0，得到 {eta_min}")

    if eta_min > base_lr:
        raise ValueError(
            f"eta_min 不能大于 base_lr：eta_min={eta_min}, base_lr={base_lr}"
        )

    if t_max <= 0:
        raise ValueError(f"t_max 必须大于 0，得到 {t_max}")

    min_ratio = eta_min / base_lr

    def lr_lambda(epoch):
        progress = min(max(epoch, 0), t_max) / float(t_max)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (
            1.0 + math.cos(math.pi * progress)
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
    """
    context_norm: [B, T=4, C=4, H, W]
    y_seq_norm:  [B, S=4, C=4, H, W]
    param:       [B, 2] = [log10(Ra), log10(Pr)]

    Each rollout step:
        1. flatten context -> [B, 16, H, W]
        2. model(model_input, param) outputs pred_delta_norm -> [B, 4, H, W]
        3. pred_next_norm = current_state_norm + pred_delta_norm
        4. compute FieldWiseRelativeL2Loss(pred_next_norm, gt_next_norm)
        5. append pred_next_norm back to context
    """
    batch_size, context_len, channels, h, w = context_norm.shape
    rollout_steps = y_seq_norm.shape[1]

    assert context_len == 4
    assert channels == 4
    assert rollout_steps == len(rollout_weights)

    if param.ndim != 2 or param.shape[1] != 2:
        raise ValueError(
            f"param should be [B, 2] = [log10(Ra), log10(Pr)], got {param.shape}"
        )

    loss_total = 0.0
    step_losses = []

    context = context_norm

    for step in range(rollout_steps):
        model_input = context.reshape(batch_size, context_len * channels, h, w)

        pred_delta_norm = model(model_input, param)

        current_state_norm = context[:, -1, :, :, :]
        pred_next_norm = current_state_norm + pred_delta_norm

        gt_next_norm = y_seq_norm[:, step, :, :, :]

        loss_step = criterion(pred_next_norm, gt_next_norm)

        loss_total = loss_total + rollout_weights[step] * loss_step
        step_losses.append(loss_step.detach())

        context = torch.cat(
            [context[:, 1:, :, :, :], pred_next_norm.unsqueeze(1)],
            dim=1
        )

    loss_total = loss_total / rollout_weights.sum()

    return loss_total, step_losses


def summarize_m8_modules(model, param_probe):
    """
    Summarize ParameterToken and field-coupling modules.

    In static mode, the conditioned branch is disabled and its
    zero-initialized parameters should remain zero.

    In parameter_conditioned mode, the conditioned correction
    starts from zero and should become non-zero during training.
    """
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    param_probe = param_probe.to(
        device=model_device,
        dtype=model_dtype,
    )

    with torch.no_grad():
        tokens = model.param_token(param_probe)

        gate = model.field_coupling.gate_values().to(
            device=model_device,
            dtype=model_dtype,
        )

        base_matrix = (
            model.field_coupling.base_coupling_matrix()
            .to(device=model_device, dtype=model_dtype)
        )

        conditioned_delta = (
            model.field_coupling.conditioned_delta_matrix(
                param_probe
            )
        )

        effective_matrix = (
            base_matrix.unsqueeze(0)
            + conditioned_delta
        )

    return {
        "token_abs_mean": float(
            tokens.abs().mean().item()
        ),
        "token_abs_max": float(
            tokens.abs().max().item()
        ),
        "token_mean": float(
            tokens.mean().item()
        ),
        "coupling_gate_mean": float(
            gate.mean().item()
        ),
        "coupling_gate_max": float(
            gate.max().item()
        ),
        "base_coupling_abs_mean": float(
            base_matrix.abs().mean().item()
        ),
        "base_coupling_abs_max": float(
            base_matrix.abs().max().item()
        ),
        "conditioned_delta_abs_mean": float(
            conditioned_delta.abs().mean().item()
        ),
        "conditioned_delta_abs_max": float(
            conditioned_delta.abs().max().item()
        ),
        "effective_coupling_abs_mean": float(
            effective_matrix.abs().mean().item()
        ),
    }



def get_git_commit(project_root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as error:
        print(f"⚠️ 无法读取 Git commit: {error}")
        return "unknown"


def make_fixed_param_probe():
    return torch.tensor(
        [
            [6.0, -0.3010299956639812],
            [6.0,  0.0],
            [7.0, -0.3010299956639812],
            [7.0,  0.0],
        ],
        dtype=torch.float32,
    )


def main():
    args = parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    BACKBONE_LR_MULT = args.backbone_lr_mult
    NEW_COMMON_LR_MULT = args.new_common_lr_mult
    DYNAMIC_LR_MULT = args.dynamic_lr_mult
    EPOCHS = args.epochs
    WEIGHT_DECAY = args.weight_decay
    ETA_MIN = args.eta_min
    SCHEDULER_T_MAX = args.scheduler_t_max
    RUN_NAME = args.run_name
    ROLLOUT_STEPS = args.rollout_steps
    TOKEN_HIDDEN_DIM = args.token_hidden_dim
    ALPHA_TOKEN = args.alpha_token
    COUPLING_MODE = args.coupling_mode
    COUPLING_PARAM_HIDDEN_DIM = args.coupling_param_hidden_dim
    COUPLING_CONDITION_SCALE = args.coupling_condition_scale
    COUPLING_INIT_GATE = args.coupling_init_gate
    SEED = args.seed

    if COUPLING_MODE == "static":
        EXPERIMENT_NAME = "M8-A-FullStatic-H4"
        EXPERIMENT_ROLE = "组合对照实验"
    else:
        EXPERIMENT_NAME = "M8-B-ParamConditionedCoupling-H4"
        EXPERIMENT_ROLE = "升级版正式候选架构"

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)

    rollout_weights_cpu = make_rollout_weights(
        ROLLOUT_STEPS,
        args.rollout_weights,
    )

    lr_multipliers = {
        "backbone": BACKBONE_LR_MULT,
        "new_common": NEW_COMMON_LR_MULT,
        "dynamic_only": DYNAMIC_LR_MULT,
    }

    if any(value <= 0 for value in lr_multipliers.values()):
        raise ValueError(
            f"学习率倍率必须全部大于 0，得到 {lr_multipliers}"
        )

    if SCHEDULER_T_MAX <= 0:
        raise ValueError(
            f"scheduler_t_max 必须大于 0，得到 {SCHEDULER_T_MAX}"
        )

    if ALPHA_TOKEN < 0:
        raise ValueError(
            f"alpha_token 必须非负，得到 {ALPHA_TOKEN}"
        )

    # 分组 optimizer 已接入；倍率参数现在可以安全生效。

    if ROLLOUT_STEPS != 4:
        print(f"⚠️ Script name is train_m8_full_conditioned_h4.py, but rollout_steps={ROLLOUT_STEPS}")

    SPLIT_PATH = resolve_path(project_root, args.split)
    STATS_PATH = resolve_path(project_root, args.stats)
    CKPT_DIR = resolve_path(project_root, args.ckpt_dir)
    INIT_CKPT = resolve_path(project_root, args.init_ckpt) if args.init_ckpt else None

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.join(project_root, "outputs", "logs"), exist_ok=True)

    GIT_COMMIT = get_git_commit(project_root)

    BEST_SAVE_PATH = os.path.join(CKPT_DIR, f"{RUN_NAME}_best.pth")

    print(f"🚀 [{EXPERIMENT_NAME}] 启动训练 | 设备: {DEVICE}")
    print(f"👉 实验定位: {EXPERIMENT_ROLE}")
    print("👉 Base: M6-FieldWiseEncoder-H4 warm start")
    print("👉 Shared: FieldWiseEncoder + FieldCoupling + ParameterToken")
    print(f"👉 Coupling mode: {COUPLING_MODE}")

    if COUPLING_MODE == "static":
        print("👉 M8-A: fixed trainable field coupling")
        print("👉 Parameter-conditioned correction disabled")
    else:
        print("👉 M8-B: Ra/Pr-conditioned coupling correction")
        print("👉 Conditional correction starts from zero")

    print("👉 ParameterToken: existing additive block tokens")
    print("👉 Prediction type: normalized delta")
    print("👉 Training type: H4 autoregressive multi-step")
    print("👉 No PDE loss")
    print(f"👉 Rollout train steps: {ROLLOUT_STEPS}")
    print(f"👉 Rollout weights: {rollout_weights_cpu.tolist()}")
    print(f"👉 Alpha token: {ALPHA_TOKEN}")
    print(f"👉 Scheduler T_max: {SCHEDULER_T_MAX}")
    print(f"👉 LR multipliers: {lr_multipliers}")
    print(f"👉 Random seed: {SEED}")
    print(f"📌 Split: {SPLIT_PATH}")
    print(f"📌 Stats: {STATS_PATH}")
    print(f"📌 Run name: {RUN_NAME}")
    print(f"📌 Init checkpoint: {INIT_CKPT}")
    print(f"📌 Best checkpoint: {BEST_SAVE_PATH}")
    print(f"📌 Git commit: {GIT_COMMIT}")

    if INIT_CKPT is None:
        print(
            "⚠️ Warning: --init_ckpt is None. "
            "For a fair M8-A/M8-B comparison, both runs must use "
            "the same M6-FieldWiseEncoder-H4 checkpoint."
        )

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"❌ 找不到 split 文件: {SPLIT_PATH}")

    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(
            f"❌ 找不到统计量文件: {STATS_PATH}，请先运行 scripts/compute_field_stats.py"
        )

    if INIT_CKPT is not None and not os.path.exists(INIT_CKPT):
        raise FileNotFoundError(f"❌ 找不到初始化 checkpoint: {INIT_CKPT}")

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    wandb.init(
        project="DC-MNO",
        name=RUN_NAME,
        config={
            "experiment": EXPERIMENT_NAME,
            "experiment_role": EXPERIMENT_ROLE,
            "architecture": (
                "FieldWiseEncoder + FieldCoupling "
                "+ ParameterToken"
            ),
            "base_model": "M6-FieldWiseEncoder-H4",

            "coupling_mode": COUPLING_MODE,
            "field_coupling": True,
            "parameter_conditioned_coupling": (
                COUPLING_MODE == "parameter_conditioned"
            ),
            "coupling_param_hidden_dim": (
                COUPLING_PARAM_HIDDEN_DIM
            ),
            "coupling_condition_scale": (
                COUPLING_CONDITION_SCALE
            ),
            "coupling_init_gate": COUPLING_INIT_GATE,

            "parameter_token": True,
            "parameter_token_type": (
                "additive_token_per_fno_block"
            ),
            "parameter_format": (
                "[log10(Ra), log10(Pr)]"
            ),
            "token_hidden_dim": TOKEN_HIDDEN_DIM,
            "alpha_token": ALPHA_TOKEN,

            "prediction_type": "delta",
            "training_type": "H4 autoregressive",
            "pde_loss": False,
            "normalization": "Field-wise mean/std",
            "delta_definition": (
                "pred_next_norm = current_state_norm "
                "+ pred_delta_norm"
            ),
            "loss_function": (
                "Weighted multi-step "
                "FieldWiseRelativeL2Loss"
            ),

            "rollout_steps": ROLLOUT_STEPS,
            "rollout_weights": (
                rollout_weights_cpu.tolist()
            ),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "backbone_lr_mult": BACKBONE_LR_MULT,
            "new_common_lr_mult": NEW_COMMON_LR_MULT,
            "dynamic_lr_mult": DYNAMIC_LR_MULT,
            "weight_decay": WEIGHT_DECAY,
            "eta_min": ETA_MIN,
            "scheduler_t_max": SCHEDULER_T_MAX,
            "seed": SEED,

            "split": SPLIT_PATH,
            "stats_path": STATS_PATH,
            "run_name": RUN_NAME,
            "init_ckpt": INIT_CKPT,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
        },
    )

    with open(SPLIT_PATH, 'r', encoding='utf-8') as f:
        split_config = json.load(f)

    print("📦 正在加载数据集：return_sequence=True, target_steps=4, return_params=True")

    train_base_dataset = RBCDataset(
        split_config=split_config["train"],
        normalize=True,
        stats_path=STATS_PATH,
        return_sequence=True,
        target_steps=ROLLOUT_STEPS,
        return_params=True,
    )

    val_base_dataset = RBCDataset(
        split_config=split_config["val"],
        normalize=True,
        stats_path=STATS_PATH,
        return_sequence=True,
        target_steps=ROLLOUT_STEPS,
        return_params=True,
    )

    train_dataset = MultiStepDeltaParamDataset(train_base_dataset, context_length=4)
    val_dataset = MultiStepDeltaParamDataset(val_base_dataset, context_length=4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples:   {len(val_dataset)}")

    sample_context, sample_y_seq, sample_param = next(iter(train_loader))
    print(f"✅ Context shape = {sample_context.shape}，应为 [B, 4, 4, H, W]")
    print(f"✅ Y_seq shape   = {sample_y_seq.shape}，应为 [B, {ROLLOUT_STEPS}, 4, H, W]")
    print(f"✅ Param shape   = {sample_param.shape}，应为 [B, 2]")
    print(f"👉 示例 param = [log10(Ra), log10(Pr)] = {sample_param[0].tolist()}")
    print(f"👉 Context mean: {sample_context.mean().item():.6f}")
    print(f"👉 Context std:  {sample_context.std().item():.6f}")
    print(f"👉 Y_seq mean:   {sample_y_seq.mean().item():.6f}")
    print(f"👉 Y_seq std:    {sample_y_seq.std().item():.6f}")

    param_probe = make_fixed_param_probe()

    print("📌 Fixed param_probe:")
    for probe_index, probe_value in enumerate(param_probe.tolist()):
        print(
            f"   probe[{probe_index}] = "
            f"[log10(Ra), log10(Pr)] = {probe_value}"
        )

    model = M8FullConditionedFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        context_length=4,
        num_fields=4,
        coupling_mode=COUPLING_MODE,
        coupling_hidden_channels=8,
        coupling_dropout=0.0,
        coupling_init_gate=COUPLING_INIT_GATE,
        coupling_use_norm=True,
        coupling_param_hidden_dim=COUPLING_PARAM_HIDDEN_DIM,
        coupling_condition_scale=COUPLING_CONDITION_SCALE,
        token_hidden_dim=TOKEN_HIDDEN_DIM,
        alpha_token=ALPHA_TOKEN,
    ).to(DEVICE)

    if INIT_CKPT is not None:
        print(f"🔁 正在从 M6-FieldWiseEncoder-H4 checkpoint 初始化 M8 共享权重: {INIT_CKPT}")
        ckpt = torch.load(INIT_CKPT, map_location=DEVICE)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print("✅ 初始化权重加载完成（strict=False）")
        print(f"   missing_keys: {missing_keys}")
        print(f"   unexpected_keys: {unexpected_keys}")

        if unexpected_keys:
            raise RuntimeError(
                "❌ Unexpected keys are not allowed when loading M6 FieldWise checkpoint into M7-ParamTokenOnly. "
                f"Got: {unexpected_keys}"
            )

        allowed_missing_prefixes = (
            "param_token.",
            "field_coupling.",
        )

        bad_missing = [
            key
            for key in missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]

        if bad_missing:
            raise RuntimeError(
                "❌ Only param_token.* and field_coupling.* may be "
                "missing when loading M6 -> M8. "
                f"Bad missing keys: {bad_missing}"
            )

        print(
            "   说明：ParameterToken 与 FieldCoupling 是 M8 新模块；"
            "其他 M6 FieldWise 权重应完整加载。"
        )
        print(
            f"✅ M6 共享权重加载成功，将训练 {EXPERIMENT_NAME}"
        )

    # --------------------------------------------------------
    # M8-A / M8-B initial alignment.
    #
    # The conditioned correction head is zero-initialized,
    # so both modes must initially produce the same output.
    # --------------------------------------------------------
    # 使用同一个固定 context 复制到所有参数探针，
    # 避免诊断 batch size 依赖训练 DataLoader 的 batch_size。
    probe_context = sample_context[:1].repeat(
        param_probe.shape[0],
        1,
        1,
        1,
        1,
    ).to(DEVICE)

    probe_param = param_probe.to(DEVICE)

    if probe_context.shape[0] != probe_param.shape[0]:
        raise RuntimeError(
            "probe_context 与 probe_param batch 不一致："
            f"context={probe_context.shape[0]}, "
            f"param={probe_param.shape[0]}"
        )

    (
        probe_batch,
        probe_t,
        probe_c,
        probe_h,
        probe_w,
    ) = probe_context.shape

    probe_x = probe_context.reshape(
        probe_batch,
        probe_t * probe_c,
        probe_h,
        probe_w,
    )

    model.eval()

    with torch.no_grad():
        initial_static = model(
            probe_x,
            probe_param,
            coupling_mode="static",
        )

        initial_conditioned = model(
            probe_x,
            probe_param,
            coupling_mode="parameter_conditioned",
        )

        initial_mode_max_diff = (
            initial_static - initial_conditioned
        ).abs().max().item()

        initial_mode_mean_diff = (
            initial_static - initial_conditioned
        ).abs().mean().item()

    print("🔎 Initial M8 mode alignment:")
    print(
        "   static-conditioned max diff: "
        f"{initial_mode_max_diff:.12e}"
    )
    print(
        "   static-conditioned mean diff: "
        f"{initial_mode_mean_diff:.12e}"
    )

    if initial_mode_max_diff >= 1e-7:
        raise RuntimeError(
            "❌ M8 initial mode alignment failed: "
            f"max diff={initial_mode_max_diff}"
        )

    print(
        "✅ M8 static / parameter_conditioned "
        "initial alignment passed"
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧮 Total params: {total_params:,}")
    print(f"🧮 Trainable params: {trainable_params:,}")

    m8_summary = summarize_m8_modules(
        model,
        param_probe,
    )

    print("🔎 Initial M8 module summary:")
    for key, value in m8_summary.items():
        print(f"   {key}: {value:.6f}")

    criterion = FieldWiseRelativeL2Loss()

    optimizer_param_groups, optimizer_group_summary, frozen_dynamic_names = (
        build_optimizer_param_groups(
            model=model,
            base_lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            backbone_lr_mult=BACKBONE_LR_MULT,
            new_common_lr_mult=NEW_COMMON_LR_MULT,
            dynamic_lr_mult=DYNAMIC_LR_MULT,
            coupling_mode=COUPLING_MODE,
        )
    )

    if frozen_dynamic_names:
        print(
            "🧊 Static 模式已冻结 dynamic-only 参数张量数: "
            f"{len(frozen_dynamic_names)}"
        )

    print("🧩 Optimizer parameter groups:")
    for group_name, info in optimizer_group_summary.items():
        print(
            f"   {group_name}: "
            f"tensors={info['num_tensors']} | "
            f"params={info['num_params']:,} | "
            f"lr={info['lr']:.2e}"
        )

    optimizer = optim.AdamW(optimizer_param_groups)

    scheduler = make_shared_cosine_scheduler(
        optimizer=optimizer,
        base_lr=LEARNING_RATE,
        eta_min=ETA_MIN,
        t_max=SCHEDULER_T_MAX,
    )

    rollout_weights = rollout_weights_cpu.to(DEVICE)

    best_val_loss = float("inf")
    top3_checkpoints = []
    start_time = time.time()

    print(
        f"\n🔥 开始 {EXPERIMENT_NAME} "
        "多步增量自回归训练..."
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss = 0.0
        total_grad_norm = 0.0
        num_train = 0
        num_batches = 0
        train_step_loss_sum = torch.zeros(ROLLOUT_STEPS, dtype=torch.float64)

        for batch_idx, (batch_context_norm, batch_y_seq_norm, batch_param) in enumerate(train_loader, start=1):
            batch_context_norm = batch_context_norm.to(DEVICE)
            batch_y_seq_norm = batch_y_seq_norm.to(DEVICE)
            batch_param = batch_param.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            loss, step_losses = autoregressive_multistep_loss(
                model=model,
                context_norm=batch_context_norm,
                y_seq_norm=batch_y_seq_norm,
                param=batch_param,
                criterion=criterion,
                rollout_weights=rollout_weights,
            )

            if not torch.isfinite(loss):
                print("⚠️ 检测到 NaN/Inf loss，跳过当前 batch")
                continue

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_size = batch_context_norm.size(0)
            train_loss += loss.item() * batch_size
            total_grad_norm += grad_norm.item()
            num_train += batch_size
            num_batches += 1

            for s, loss_s in enumerate(step_losses):
                train_step_loss_sum[s] += float(loss_s.item()) * batch_size

            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

        train_loss = train_loss / max(num_train, 1)
        avg_grad_norm = total_grad_norm / max(num_batches, 1)
        train_step_losses = train_step_loss_sum / max(num_train, 1)

        model.eval()
        val_loss = 0.0
        num_val = 0
        val_step_loss_sum = torch.zeros(ROLLOUT_STEPS, dtype=torch.float64)

        with torch.no_grad():
            for batch_idx, (batch_context_norm, batch_y_seq_norm, batch_param) in enumerate(val_loader, start=1):
                batch_context_norm = batch_context_norm.to(DEVICE)
                batch_y_seq_norm = batch_y_seq_norm.to(DEVICE)
                batch_param = batch_param.to(DEVICE)

                loss, step_losses = autoregressive_multistep_loss(
                    model=model,
                    context_norm=batch_context_norm,
                    y_seq_norm=batch_y_seq_norm,
                    param=batch_param,
                    criterion=criterion,
                    rollout_weights=rollout_weights,
                )

                if not torch.isfinite(loss):
                    continue

                batch_size = batch_context_norm.size(0)
                val_loss += loss.item() * batch_size
                num_val += batch_size

                for s, loss_s in enumerate(step_losses):
                    val_step_loss_sum[s] += float(loss_s.item()) * batch_size

                if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                    break

        val_loss = val_loss / max(num_val, 1)
        val_step_losses = val_step_loss_sum / max(num_val, 1)

        current_lr_dict = {
            group.get("name", f"group_{idx}"): group["lr"]
            for idx, group in enumerate(optimizer.param_groups)
        }
        lr_msg = " | ".join(
            [
                f"{name} LR: {lr:.2e}"
                for name, lr in current_lr_dict.items()
            ]
        )
        current_best = min(best_val_loss, val_loss)
        m8_summary = summarize_m8_modules(
            model,
            param_probe,
        )

        step_msg = " | ".join(
            [f"Val t+{i+1}: {val_step_losses[i]:.6f}" for i in range(ROLLOUT_STEPS)]
        )

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"Train MStep RelL2: {train_loss:.6f} | "
            f"Val MStep RelL2: {val_loss:.6f} | "
            f"{step_msg} | "
            f"Grad Norm: {avg_grad_norm:.4f} | "
            f"TokenAbsMean: "
            f"{m8_summary['token_abs_mean']:.5f} | "
            f"GateMean: "
            f"{m8_summary['coupling_gate_mean']:.5f} | "
            f"CondDeltaAbsMax: "
            f"{m8_summary['conditioned_delta_abs_max']:.6f} | "
            f"{lr_msg}"
        )

        log_dict = {
            "epoch": epoch,
            "Train MultiStep Rel-L2": train_loss,
            "Val MultiStep Rel-L2": val_loss,
            "Grad Norm": avg_grad_norm,
            "Learning Rate / backbone": current_lr_dict.get("backbone", float("nan")),
            "Learning Rate / new_common": current_lr_dict.get("new_common", float("nan")),
            "Learning Rate / dynamic_only": current_lr_dict.get("dynamic_only", float("nan")),
            "Best Val MultiStep Rel-L2": current_best,
            "ParameterToken abs mean": (
                m8_summary["token_abs_mean"]
            ),
            "ParameterToken abs max": (
                m8_summary["token_abs_max"]
            ),
            "ParameterToken mean": (
                m8_summary["token_mean"]
            ),
            "Coupling gate mean": (
                m8_summary["coupling_gate_mean"]
            ),
            "Coupling gate max": (
                m8_summary["coupling_gate_max"]
            ),
            "Base coupling abs mean": (
                m8_summary["base_coupling_abs_mean"]
            ),
            "Base coupling abs max": (
                m8_summary["base_coupling_abs_max"]
            ),
            "Conditioned delta abs mean": (
                m8_summary[
                    "conditioned_delta_abs_mean"
                ]
            ),
            "Conditioned delta abs max": (
                m8_summary[
                    "conditioned_delta_abs_max"
                ]
            ),
        }

        for i in range(ROLLOUT_STEPS):
            log_dict[f"Train step t+{i+1} Rel-L2"] = float(train_step_losses[i])
            log_dict[f"Val step t+{i+1} Rel-L2"] = float(val_step_losses[i])

        wandb.log(log_dict)

        model_config = {
            "in_channels": 16,
            "out_channels": 4,
            "modes1": 16,
            "modes2": 16,
            "width": 32,
            "context_length": 4,
            "num_fields": 4,
            "field_width": None,
            "coupling_mode": COUPLING_MODE,
            "coupling_hidden_channels": 8,
            "coupling_dropout": 0.0,
            "coupling_init_gate": COUPLING_INIT_GATE,
            "coupling_use_norm": True,
            "coupling_param_hidden_dim": COUPLING_PARAM_HIDDEN_DIM,
            "coupling_condition_scale": COUPLING_CONDITION_SCALE,
            "token_hidden_dim": TOKEN_HIDDEN_DIM,
            "alpha_token": ALPHA_TOKEN,
        }

        train_config = {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "base_lr": LEARNING_RATE,
            "backbone_lr_mult": BACKBONE_LR_MULT,
            "new_common_lr_mult": NEW_COMMON_LR_MULT,
            "dynamic_lr_mult": DYNAMIC_LR_MULT,
            "weight_decay": WEIGHT_DECAY,
            "eta_min": ETA_MIN,
            "scheduler_type": "shared_cosine_multiplier",
            "scheduler_t_max": SCHEDULER_T_MAX,
            "rollout_steps": ROLLOUT_STEPS,
            "rollout_weights": rollout_weights.detach().cpu().tolist(),
            "seed": SEED,
            "grad_clip": 1.0,
        }

        data_config = {
            "split_path": SPLIT_PATH,
            "stats_path": STATS_PATH,
            "train_key": "train",
            "validation_key": "val",
            "context_length": 4,
            "field_order": [
                "buoyancy",
                "u_x",
                "u_y",
                "pressure",
            ],
            "init_ckpt": INIT_CKPT,
        }

        checkpoint_payload = {
            "checkpoint_format_version": 2,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "model_config": model_config,
            "train_config": train_config,
            "data_config": data_config,
            "git_commit": GIT_COMMIT,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,

            "experiment": EXPERIMENT_NAME,
            "experiment_role": EXPERIMENT_ROLE,
            "base_model": "M6-FieldWiseEncoder-H4",

            "prediction_type": (
                "multi_step_delta_prediction"
            ),
            "normalization": "field-wise mean/std",
            "task": (
                "autoregressive multi-step "
                "delta prediction"
            ),
            "delta_definition": (
                "pred_next_norm = current_state_norm "
                "+ pred_delta_norm"
            ),
            "loss_function": (
                "Weighted multi-step "
                "FieldWiseRelativeL2Loss "
                "on normalized state"
            ),

            "parameter_token": {
                "enabled": True,
                "type": (
                    "additive_token_per_fno_block"
                ),
                "token_hidden_dim": TOKEN_HIDDEN_DIM,
                "alpha_token": ALPHA_TOKEN,
                "num_layers": 4,
                "token_abs_mean": (
                    m8_summary["token_abs_mean"]
                ),
                "token_abs_max": (
                    m8_summary["token_abs_max"]
                ),
            },

            "field_coupling": {
                "enabled": True,
                "mode": COUPLING_MODE,
                "parameter_conditioned": (
                    COUPLING_MODE
                    == "parameter_conditioned"
                ),
                "param_hidden_dim": (
                    COUPLING_PARAM_HIDDEN_DIM
                ),
                "condition_scale": (
                    COUPLING_CONDITION_SCALE
                ),
                "initial_gate_logit": (
                    COUPLING_INIT_GATE
                ),
                "gate_mean": (
                    m8_summary["coupling_gate_mean"]
                ),
                "base_coupling_abs_mean": (
                    m8_summary[
                        "base_coupling_abs_mean"
                    ]
                ),
                "conditioned_delta_abs_mean": (
                    m8_summary[
                        "conditioned_delta_abs_mean"
                    ]
                ),
                "conditioned_delta_abs_max": (
                    m8_summary[
                        "conditioned_delta_abs_max"
                    ]
                ),
            },

            "pde_loss": False,
            "rollout_steps": ROLLOUT_STEPS,
            "rollout_weights": (
                rollout_weights.detach()
                .cpu()
                .tolist()
            ),

            "split_path": SPLIT_PATH,
            "stats_path": STATS_PATH,
            "run_name": RUN_NAME,
            "init_ckpt": INIT_CKPT,
            "seed": SEED,

            "initial_static_conditioned_max_diff": (
                initial_mode_max_diff
            ),
            "initial_static_conditioned_mean_diff": (
                initial_mode_mean_diff
            ),

            "group_learning_rates": current_lr_dict,

            "training_protocol": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "backbone_lr_mult": BACKBONE_LR_MULT,
                "new_common_lr_mult": NEW_COMMON_LR_MULT,
                "dynamic_lr_mult": DYNAMIC_LR_MULT,
                "weight_decay": WEIGHT_DECAY,
                "eta_min": ETA_MIN,
                "scheduler_t_max": SCHEDULER_T_MAX,
                "grad_clip": 1.0,
                "max_train_batches": (
                    args.max_train_batches
                ),
                "max_val_batches": (
                    args.max_val_batches
                ),
            },
        }

        # 所有本轮保存文件都记录更新后的历史最优 validation loss。
        checkpoint_payload["best_val_loss"] = min(
            best_val_loss,
            val_loss,
        )

        # H4 validation loss Top-3 候选池。
        qualifies_for_top3 = (
            len(top3_checkpoints) < 3
            or val_loss < top3_checkpoints[-1]["val_loss"]
        )

        if qualifies_for_top3:
            top3_save_path = os.path.join(
                CKPT_DIR,
                f"{RUN_NAME}_valtop_epoch_{epoch:02d}.pth",
            )

            torch.save(
                checkpoint_payload,
                top3_save_path,
            )

            top3_checkpoints.append(
                {
                    "epoch": int(epoch),
                    "val_loss": float(val_loss),
                    "path": top3_save_path,
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

                if os.path.exists(removed["path"]):
                    os.remove(removed["path"])

                print(
                    "   [Top-3 移除] "
                    f"epoch={removed['epoch']} | "
                    f"val={removed['val_loss']:.6f}"
                )

            print("   [Top-3 当前排名]")

            for rank, record in enumerate(
                top3_checkpoints,
                start=1,
            ):
                print(
                    f"      #{rank}: "
                    f"epoch={record['epoch']} | "
                    f"val={record['val_loss']:.6f}"
                )

        # 固定周期 checkpoint：epoch 5 / 10 / 15 / 20。
        if epoch % 5 == 0:
            epoch_save_path = os.path.join(
                CKPT_DIR,
                f"{RUN_NAME}_epoch_{epoch:02d}.pth"
            )
            torch.save(checkpoint_payload, epoch_save_path)
            print(f"   [*] 保存周期 checkpoint: {epoch_save_path}")

        # 保留原有 best checkpoint，兼容既有训练和评估流程。
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_payload["best_val_loss"] = best_val_loss
            torch.save(checkpoint_payload, BEST_SAVE_PATH)
            print(f"  🌟 [New Best] 权重已保存至: {BEST_SAVE_PATH}")

        scheduler.step()

    total_time = time.time() - start_time

    print(f"\n✅ {EXPERIMENT_NAME} 训练完成!")
    print(f"📌 Coupling mode: {COUPLING_MODE}")
    print(f"📌 最优模型: {BEST_SAVE_PATH}")

    print("📌 最终 H4 validation Top-3:")

    for rank, record in enumerate(
        top3_checkpoints,
        start=1,
    ):
        print(
            f"   #{rank}: "
            f"epoch={record['epoch']} | "
            f"val={record['val_loss']:.6f} | "
            f"path={record['path']}"
        )

    print(f"⏱️ 总耗时: {total_time / 60:.2f} 分钟")

    final_summary = summarize_m8_modules(
        model,
        param_probe,
    )

    print("🔎 Final M8 module summary:")
    for key, value in final_summary.items():
        print(f"   {key}: {value:.6f}")

    wandb.finish()


if __name__ == "__main__":
    main()
