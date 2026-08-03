import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.rbc_dataset import RBCDataset
from models.operators.fno2d_fieldwise_m7 import M7FieldCouplingFNO2d
from training.metrics import FieldWiseRelativeL2Loss


class MultiStepDeltaDataset(Dataset):
    """
    M7-FieldCoupling-H4 dataset wrapper.

    base_dataset must return:
        x_norm:     [16, H, W]
        y_norm:     [4, H, W], old interface, unused here
        y_seq_norm: [S, 4, H, W]

    This wrapper returns:
        context_norm: [4, 4, H, W]
        y_seq_norm:  [S, 4, H, W]

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

    def __getitem__(self, idx):
        x_norm, y_norm, y_seq_norm = self.base_dataset[idx]

        c_total, h, w = x_norm.shape
        assert c_total == self.context_length * 4, (
            f"❌ x_norm channels should be {self.context_length * 4}, got {c_total}"
        )

        context_norm = x_norm.view(self.context_length, 4, h, w)

        return context_norm, y_seq_norm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train M7-FieldCoupling-H4: FieldWiseEncoder plus explicit "
            "field-to-field coupling, no ParameterToken, no PDE loss."
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
        default="m7_fieldcoupling_h4_unseen_pr_o5",
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
            "Only field_coupling.* should be missing with strict=False."
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
        help="M7 field coupling rollout training uses similar memory as M6 field-wise training. Start with 4."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=6e-4,
        help="O5 initial learning rate. Default: 6e-4."
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization and DataLoader shuffle."
    )

    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help=(
            "Allow overwriting an existing best checkpoint. "
            "Default behavior is to stop for safety."
        )
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
    criterion,
    rollout_weights,
):
    """
    context_norm: [B, T=4, C=4, H, W]
    y_seq_norm:  [B, S=4, C=4, H, W]

    Each rollout step:
        1. flatten context -> [B, 16, H, W]
        2. model outputs pred_delta_norm -> [B, 4, H, W]
        3. pred_next_norm = current_state_norm + pred_delta_norm
        4. compute FieldWiseRelativeL2Loss(pred_next_norm, gt_next_norm)
        5. append pred_next_norm back to context
    """
    batch_size, context_len, channels, h, w = context_norm.shape
    rollout_steps = y_seq_norm.shape[1]

    assert context_len == 4
    assert channels == 4
    assert rollout_steps == len(rollout_weights)

    loss_total = 0.0
    step_losses = []

    context = context_norm

    for step in range(rollout_steps):
        model_input = context.reshape(batch_size, context_len * channels, h, w)

        pred_delta_norm = model(model_input)

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


def summarize_coupling(model):
    gate = model.field_coupling.gate_values().detach().cpu()
    matrix = model.field_coupling.effective_coupling_matrix().detach().cpu()

    return {
        "gate_mean": float(gate.mean().item()),
        "gate_max": float(gate.max().item()),
        "gate_min": float(gate.min().item()),
        "matrix_abs_mean": float(matrix.abs().mean().item()),
        "matrix_abs_max": float(matrix.abs().max().item()),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    WEIGHT_DECAY = args.weight_decay
    ETA_MIN = args.eta_min
    RUN_NAME = args.run_name
    ROLLOUT_STEPS = args.rollout_steps
    SEED = args.seed

    if ROLLOUT_STEPS != 4:
        print(f"⚠️ Script name is train_m7_fieldcoupling_h4.py, but rollout_steps={ROLLOUT_STEPS}")

    SPLIT_PATH = resolve_path(project_root, args.split)
    STATS_PATH = resolve_path(project_root, args.stats)
    CKPT_DIR = resolve_path(project_root, args.ckpt_dir)
    INIT_CKPT = resolve_path(project_root, args.init_ckpt) if args.init_ckpt else None

    os.makedirs(CKPT_DIR, exist_ok=True)

    BEST_SAVE_PATH = os.path.join(CKPT_DIR, f"{RUN_NAME}_best.pth")

    if os.path.exists(BEST_SAVE_PATH) and not args.allow_overwrite:
        raise FileExistsError(
            "❌ 检测到同名 best checkpoint，已停止以防覆盖：\n"
            f"   {BEST_SAVE_PATH}\n"
            "请更换 --run_name。只有明确需要覆盖时才使用 --allow_overwrite。"
        )

    print(f"🚀 [M7-FieldCoupling-O5-H4] 启动训练 | 设备: {DEVICE}")
    print("👉 Base: M6-FieldWiseEncoder-H4 warm start")
    print("👉 Upgrade: explicit field-to-field coupling before fusion")
    print("👉 No ParameterToken")
    print("👉 No PDE loss")
    print("👉 Input: 4 frames × 4 fields = [B, 16, H, W]")
    print("👉 Output per step: 4 delta fields")
    print(f"👉 Rollout train steps: {ROLLOUT_STEPS}")
    print("👉 Loss: weighted multi-step FieldWiseRelativeL2Loss on predicted normalized state")
    print(f"📌 Split: {SPLIT_PATH}")
    print(f"📌 Stats: {STATS_PATH}")
    print(f"📌 Run name: {RUN_NAME}")
    print(f"📌 Init checkpoint: {INIT_CKPT}")
    print(f"📌 Best checkpoint: {BEST_SAVE_PATH}")
    print(f"📌 Random seed: {SEED}")
    print(f"📌 O5 initial learning rate: {LEARNING_RATE:.2e}")
    print("📌 Existing checkpoint protection: enabled")

    if INIT_CKPT is None:
        print("⚠️ Warning: --init_ckpt is None. For fair M7 vs FieldWise-continue comparison, use M6-FieldWiseEncoder-H4 checkpoint.")

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
            "experiment_type": "m7_fieldcoupling_o5_h4",
            "architecture": "FieldWiseEncoder + explicit FieldCouplingBlock",
            "base_model": "M6-FieldWiseEncoder-H4",
            "upgrade": "explicit 4x4 field-to-field coupling before fusion",
            "parameter_token": False,
            "pde_loss": False,
            "normalization": "Field-wise mean/std",
            "task": "multi-step delta prediction",
            "delta_definition": "pred_next_norm = current_state_norm + pred_delta_norm",
            "loss_function": "Weighted multi-step FieldWiseRelativeL2Loss on normalized state",
            "rollout_steps": ROLLOUT_STEPS,
            "rollout_weights": make_rollout_weights(ROLLOUT_STEPS).tolist(),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "eta_min": ETA_MIN,
            "grad_clip": 1.0,
            "split": SPLIT_PATH,
            "stats_path": STATS_PATH,
            "run_name": RUN_NAME,
            "init_ckpt": INIT_CKPT,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
        }
    )

    with open(SPLIT_PATH, 'r', encoding='utf-8') as f:
        split_config = json.load(f)

    print("📦 正在加载数据集：return_sequence=True, target_steps=4")

    train_base_dataset = RBCDataset(
        split_config=split_config["train"],
        normalize=True,
        stats_path=STATS_PATH,
        return_sequence=True,
        target_steps=ROLLOUT_STEPS,
    )

    val_base_dataset = RBCDataset(
        split_config=split_config["val"],
        normalize=True,
        stats_path=STATS_PATH,
        return_sequence=True,
        target_steps=ROLLOUT_STEPS,
    )

    train_dataset = MultiStepDeltaDataset(train_base_dataset, context_length=4)
    val_dataset = MultiStepDeltaDataset(val_base_dataset, context_length=4)

    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=train_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples:   {len(val_dataset)}")

    sample_context, sample_y_seq = next(iter(train_loader))
    print(f"✅ Context shape = {sample_context.shape}，应为 [B, 4, 4, H, W]")
    print(f"✅ Y_seq shape   = {sample_y_seq.shape}，应为 [B, {ROLLOUT_STEPS}, 4, H, W]")
    print(f"👉 Context mean: {sample_context.mean().item():.6f}")
    print(f"👉 Context std:  {sample_context.std().item():.6f}")
    print(f"👉 Y_seq mean:   {sample_y_seq.mean().item():.6f}")
    print(f"👉 Y_seq std:    {sample_y_seq.std().item():.6f}")

    model = M7FieldCouplingFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        context_length=4,
        num_fields=4,
        coupling_hidden_channels=8,
        coupling_dropout=0.0,
        coupling_init_gate=-4.0,
        coupling_use_norm=True,
    ).to(DEVICE)

    if INIT_CKPT is not None:
        print(f"🔁 正在从 M6-FieldWiseEncoder-H4 checkpoint 初始化 M7-FieldCoupling-H4 共享权重: {INIT_CKPT}")
        ckpt = torch.load(INIT_CKPT, map_location=DEVICE)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print("✅ 初始化权重加载完成（strict=False）")
        print(f"   missing_keys: {missing_keys}")
        print(f"   unexpected_keys: {unexpected_keys}")

        if unexpected_keys:
            raise RuntimeError(
                "❌ Unexpected keys are not allowed when loading M6 FieldWise checkpoint into M7. "
                f"Got: {unexpected_keys}"
            )

        bad_missing = [k for k in missing_keys if not k.startswith("field_coupling.")]
        if bad_missing:
            raise RuntimeError(
                "❌ Only field_coupling.* keys should be missing when loading M6 -> M7. "
                f"Bad missing keys: {bad_missing}"
            )

        print("   说明：field_coupling 是 M7 新模块；其他 M6 FieldWise 权重应完整加载。")
        print("✅ M6-FieldWiseEncoder-H4 共享权重加载成功，将训练 M7-FieldCoupling-H4")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧮 Total params: {total_params:,}")
    print(f"🧮 Trainable params: {trainable_params:,}")

    coupling_summary = summarize_coupling(model)
    print("🔎 Initial coupling summary:")
    for k, v in coupling_summary.items():
        print(f"   {k}: {v:.6f}")

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
    top_checkpoints = []
    start_time = time.time()

    print("\n🔥 开始 M7-FieldCoupling-H4 多步自回归训练...")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss = 0.0
        total_grad_norm = 0.0
        num_train = 0
        num_batches = 0
        train_step_loss_sum = torch.zeros(ROLLOUT_STEPS, dtype=torch.float64)

        for batch_idx, (batch_context_norm, batch_y_seq_norm) in enumerate(train_loader, start=1):
            batch_context_norm = batch_context_norm.to(DEVICE)
            batch_y_seq_norm = batch_y_seq_norm.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            loss, step_losses = autoregressive_multistep_loss(
                model=model,
                context_norm=batch_context_norm,
                y_seq_norm=batch_y_seq_norm,
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
            for batch_idx, (batch_context_norm, batch_y_seq_norm) in enumerate(val_loader, start=1):
                batch_context_norm = batch_context_norm.to(DEVICE)
                batch_y_seq_norm = batch_y_seq_norm.to(DEVICE)

                loss, step_losses = autoregressive_multistep_loss(
                    model=model,
                    context_norm=batch_context_norm,
                    y_seq_norm=batch_y_seq_norm,
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
        coupling_summary = summarize_coupling(model)

        step_msg = " | ".join(
            [f"Val t+{i+1}: {val_step_losses[i]:.6f}" for i in range(ROLLOUT_STEPS)]
        )

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"Train MStep RelL2: {train_loss:.6f} | "
            f"Val MStep RelL2: {val_loss:.6f} | "
            f"{step_msg} | "
            f"Grad Norm: {avg_grad_norm:.4f} | "
            f"GateMean: {coupling_summary['gate_mean']:.5f} | "
            f"CAbsMean: {coupling_summary['matrix_abs_mean']:.5f} | "
            f"LR: {current_lr:.2e}"
        )

        log_dict = {
            "epoch": epoch,
            "Train MultiStep Rel-L2": train_loss,
            "Val MultiStep Rel-L2": val_loss,
            "Grad Norm": avg_grad_norm,
            "Learning Rate": current_lr,
            "Best Val MultiStep Rel-L2": current_best,
            "Coupling gate mean": coupling_summary["gate_mean"],
            "Coupling gate max": coupling_summary["gate_max"],
            "Coupling gate min": coupling_summary["gate_min"],
            "Coupling matrix abs mean": coupling_summary["matrix_abs_mean"],
            "Coupling matrix abs max": coupling_summary["matrix_abs_max"],
        }

        for i in range(ROLLOUT_STEPS):
            log_dict[f"Train step t+{i+1} Rel-L2"] = float(train_step_losses[i])
            log_dict[f"Val step t+{i+1} Rel-L2"] = float(val_step_losses[i])

        wandb.log(log_dict)

        is_new_best = val_loss < best_val_loss
        if is_new_best:
            best_val_loss = val_loss

        checkpoint_payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'experiment': 'M7-FieldCoupling-O5-H4',
            'base_model': 'M6-FieldWiseEncoder-H4',
            'prediction_type': 'multi_step_delta_prediction',
            'normalization': 'field-wise mean/std',
            'task': 'autoregressive multi-step delta prediction',
            'delta_definition': 'pred_next_norm = current_state_norm + pred_delta_norm',
            'loss_function': 'Weighted multi-step FieldWiseRelativeL2Loss on normalized state',
            'parameter_token': False,
            'pde_loss': False,
            'field_coupling': {
                'type': 'explicit_4x4_field_to_field_coupling',
                'insert_position': 'after_field_encoders_before_fusion',
                'coupling_hidden_channels': 8,
                'coupling_dropout': 0.0,
                'coupling_init_gate': -4.0,
                'coupling_use_norm': True,
                'coupling_summary': coupling_summary,
            },
            'rollout_steps': ROLLOUT_STEPS,
            'rollout_weights': rollout_weights.detach().cpu().tolist(),
            'split_path': SPLIT_PATH,
            'stats_path': STATS_PATH,
            'run_name': RUN_NAME,
            'init_ckpt': INIT_CKPT,
            'training_protocol': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'seed': SEED,
                'weight_decay': WEIGHT_DECAY,
                'scheduler': 'CosineAnnealingLR',
                'eta_min': ETA_MIN,
                'grad_clip': 1.0,
                'max_train_batches': args.max_train_batches,
                'max_val_batches': args.max_val_batches,
            }
        }

        if epoch % 5 == 0:
            epoch_save_path = os.path.join(
                CKPT_DIR,
                f"{RUN_NAME}_epoch_{epoch:02d}.pth"
            )
            torch.save(checkpoint_payload, epoch_save_path)
            print(f"   [*] 保存周期 checkpoint: {epoch_save_path}")

        if is_new_best:
            torch.save(checkpoint_payload, BEST_SAVE_PATH)
            print(f"  🌟 [New Best] 权重已保存至: {BEST_SAVE_PATH}")

        top_path = os.path.join(
            CKPT_DIR,
            f"{RUN_NAME}_top_candidate_epoch_{epoch:02d}.pth"
        )
        torch.save(checkpoint_payload, top_path)

        top_checkpoints.append(
            {
                "epoch": epoch,
                "val_loss": float(val_loss),
                "path": top_path,
            }
        )
        top_checkpoints.sort(key=lambda item: item["val_loss"])

        while len(top_checkpoints) > 3:
            removed = top_checkpoints.pop()
            if os.path.exists(removed["path"]):
                os.remove(removed["path"])
            print(
                f"   [Top-3 移除] epoch={removed['epoch']} | "
                f"val={removed['val_loss']:.6f}"
            )

        print("   [Top-3 当前排名]")
        for rank, item in enumerate(top_checkpoints, start=1):
            print(
                f"      #{rank}: epoch={item['epoch']} | "
                f"val={item['val_loss']:.6f}"
            )

        scheduler.step()

    total_time = time.time() - start_time

    print(f"\n✅ M7-FieldCoupling-O5-H4 训练完成!")
    print(f"📌 最优模型: {BEST_SAVE_PATH}")
    print(f"⏱️ 总耗时: {total_time / 60:.2f} 分钟")

    final_summary = summarize_coupling(model)
    print("🔎 Final coupling summary:")
    for k, v in final_summary.items():
        print(f"   {k}: {v:.6f}")

    wandb.finish()


if __name__ == "__main__":
    main()
