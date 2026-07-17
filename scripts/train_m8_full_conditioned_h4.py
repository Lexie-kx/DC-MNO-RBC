import os
import sys
import json
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
        "--rollout_steps",
        type=int,
        default=4,
        help="Number of autoregressive training steps. For this script, use 4."
    )

    parser.add_argument(
        "--token_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension of ParameterToken MLP."
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


def make_rollout_weights(rollout_steps):
    """
    H4 default weights:
        t+1: 1.0
        t+2: 0.8
        t+3: 0.6
        t+4: 0.4
    """
    if rollout_steps == 4:
        return torch.tensor([1.0, 0.8, 0.6, 0.4], dtype=torch.float32)

    return torch.linspace(1.0, 0.4, steps=rollout_steps, dtype=torch.float32)


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


def main():
    args = parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    WEIGHT_DECAY = args.weight_decay
    ETA_MIN = args.eta_min
    RUN_NAME = args.run_name
    ROLLOUT_STEPS = args.rollout_steps
    TOKEN_HIDDEN_DIM = args.token_hidden_dim
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

    if ROLLOUT_STEPS != 4:
        print(f"⚠️ Script name is train_m8_full_conditioned_h4.py, but rollout_steps={ROLLOUT_STEPS}")

    SPLIT_PATH = resolve_path(project_root, args.split)
    STATS_PATH = resolve_path(project_root, args.stats)
    CKPT_DIR = resolve_path(project_root, args.ckpt_dir)
    INIT_CKPT = resolve_path(project_root, args.init_ckpt) if args.init_ckpt else None

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.join(project_root, "outputs", "logs"), exist_ok=True)

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
    print(f"👉 Random seed: {SEED}")
    print(f"📌 Split: {SPLIT_PATH}")
    print(f"📌 Stats: {STATS_PATH}")
    print(f"📌 Run name: {RUN_NAME}")
    print(f"📌 Init checkpoint: {INIT_CKPT}")
    print(f"📌 Best checkpoint: {BEST_SAVE_PATH}")

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
                make_rollout_weights(
                    ROLLOUT_STEPS
                ).tolist()
            ),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "eta_min": ETA_MIN,
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

    param_probe = sample_param[: min(4, sample_param.shape[0])].clone()

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
    probe_context = sample_context[
        : param_probe.shape[0]
    ].to(DEVICE)

    probe_param = param_probe.to(DEVICE)

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

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=ETA_MIN
    )

    rollout_weights = make_rollout_weights(ROLLOUT_STEPS).to(DEVICE)

    best_val_loss = float('inf')
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

        current_lr = optimizer.param_groups[0]['lr']
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
            f"LR: {current_lr:.2e}"
        )

        log_dict = {
            "epoch": epoch,
            "Train MultiStep Rel-L2": train_loss,
            "Val MultiStep Rel-L2": val_loss,
            "Grad Norm": avg_grad_norm,
            "Learning Rate": current_lr,
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

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
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

            "training_protocol": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "eta_min": ETA_MIN,
                "grad_clip": 1.0,
                "max_train_batches": (
                    args.max_train_batches
                ),
                "max_val_batches": (
                    args.max_val_batches
                ),
            },
        }

        if epoch % 5 == 0:
            epoch_save_path = os.path.join(
                CKPT_DIR,
                f"{RUN_NAME}_epoch_{epoch:02d}.pth"
            )
            torch.save(checkpoint_payload, epoch_save_path)
            print(f"   [*] 保存周期 checkpoint: {epoch_save_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_payload['best_val_loss'] = best_val_loss
            torch.save(checkpoint_payload, BEST_SAVE_PATH)
            print(f"  🌟 [New Best] 权重已保存至: {BEST_SAVE_PATH}")

        scheduler.step()

    total_time = time.time() - start_time

    print(f"\n✅ {EXPERIMENT_NAME} 训练完成!")
    print(f"📌 Coupling mode: {COUPLING_MODE}")
    print(f"📌 最优模型: {BEST_SAVE_PATH}")
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
