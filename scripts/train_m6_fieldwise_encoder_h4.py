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
from models.operators.fno2d_fieldwise import FieldWiseFNO2d
from training.metrics import FieldWiseRelativeL2Loss


class MultiStepDeltaDataset(Dataset):
    """
    M6-FieldWiseEncoder-H4 dataset wrapper.

    base_dataset 必须返回:
        x_norm:     [16, H, W]
        y_norm:     [4, H, W]，旧接口保留，但这里不用
        y_seq_norm: [S, 4, H, W]

    本 wrapper 返回:
        context_norm: [4, 4, H, W]
        y_seq_norm:  [S, 4, H, W]

    训练时模型仍然预测 delta_norm:
        pred_next_norm = current_state_norm + pred_delta_norm

    但 multi-step loss 对 pred_next_norm 和 gt_next_norm 计算，
    这样可以直接约束 autoregressive rollout 轨迹。
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
            f"❌ x_norm 通道数应为 {self.context_length * 4}，但得到 {c_total}"
        )

        context_norm = x_norm.view(self.context_length, 4, h, w)

        return context_norm, y_seq_norm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train M6-FieldWiseEncoder-H4: multi-step autoregressive field-wise Delta-FNO."
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
        default="m6_fieldwise_encoder_h4_unseen_pr",
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
        help="Optional M5-Delta-H4 checkpoint used to initialize shared FNO blocks for M6-FieldWiseEncoder-H4."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="M6 field-wise rollout training uses more memory than one-step training. Start with 4."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--eta_min",
        type=float,
        default=1e-5
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

    return parser.parse_args()


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


def make_rollout_weights(rollout_steps):
    """
    H4 默认权重:
        t+1: 1.0
        t+2: 0.8
        t+3: 0.6
        t+4: 0.4
    如果以后扩展到其他 steps，再自动线性衰减。
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

    每一步:
        1. flatten context -> [B, 16, H, W]
        2. model 输出 pred_delta_norm -> [B, 4, H, W]
        3. pred_next_norm = current_state_norm + pred_delta_norm
        4. 对 pred_next_norm 和 gt_next_norm 算 FieldWiseRelativeL2Loss
        5. 用 pred_next_norm 回填 context
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

    if ROLLOUT_STEPS != 4:
        print(f"⚠️ 当前脚本名是 train_m6_fieldwise_encoder_h4.py，但 rollout_steps={ROLLOUT_STEPS}")

    SPLIT_PATH = resolve_path(project_root, args.split)
    STATS_PATH = resolve_path(project_root, args.stats)
    CKPT_DIR = resolve_path(project_root, args.ckpt_dir)
    INIT_CKPT = resolve_path(project_root, args.init_ckpt) if args.init_ckpt else None

    os.makedirs(CKPT_DIR, exist_ok=True)

    BEST_SAVE_PATH = os.path.join(CKPT_DIR, f"{RUN_NAME}_best.pth")

    print(f"🚀 [M6-FieldWiseEncoder-H4] 启动训练 | 设备: {DEVICE}")
    print("👉 Base: M5-Delta-H4 warm start + new FieldWiseEncoder")
    print("👉 Upgrade: field-wise encoder + multi-step autoregressive rollout training")
    print("👉 Input: 4 frames × 4 fields = [B, 16, H, W]")
    print("👉 Output per step: 4 delta fields")
    print(f"👉 Rollout train steps: {ROLLOUT_STEPS}")
    print("👉 Loss: weighted multi-step FieldWiseRelativeL2Loss on predicted normalized state")
    print(f"📌 Split: {SPLIT_PATH}")
    print(f"📌 Stats: {STATS_PATH}")
    print(f"📌 Run name: {RUN_NAME}")
    print(f"📌 Init checkpoint: {INIT_CKPT}")
    print(f"📌 Best checkpoint: {BEST_SAVE_PATH}")

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"❌ 找不到 split 文件: {SPLIT_PATH}")

    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(
            f"❌ 找不到统计量文件: {STATS_PATH}，请先运行 scripts/compute_field_stats.py"
        )

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    wandb.init(
        project="DC-MNO",
        name=RUN_NAME,
        config={
            "experiment_type": "m5_multistep_delta_prediction",
            "architecture": "Plain FNO",
            "base_model": "M3-Delta",
            "upgrade": "multi-step autoregressive training",
            "normalization": "Field-wise mean/std",
            "task": "multi-step delta prediction",
            "delta_definition": "pred_next_norm = current_state_norm + pred_delta_norm",
            "loss_function": "Weighted multi-step FieldWiseRelativeL2Loss on normalized state",
            "rollout_steps": ROLLOUT_STEPS,
            "rollout_weights": make_rollout_weights(ROLLOUT_STEPS).tolist(),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples:   {len(val_dataset)}")

    sample_context, sample_y_seq = next(iter(train_loader))
    print(f"✅ Context shape = {sample_context.shape}，应为 [B, 4, 4, 256, 64]")
    print(f"✅ Y_seq shape   = {sample_y_seq.shape}，应为 [B, {ROLLOUT_STEPS}, 4, 256, 64]")
    print(f"👉 Context mean: {sample_context.mean().item():.6f}")
    print(f"👉 Context std:  {sample_context.std().item():.6f}")
    print(f"👉 Y_seq mean:   {sample_y_seq.mean().item():.6f}")
    print(f"👉 Y_seq std:    {sample_y_seq.std().item():.6f}")

    model = FieldWiseFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32
    ).to(DEVICE)

    if INIT_CKPT is not None:
        if not os.path.exists(INIT_CKPT):
            raise FileNotFoundError(f"❌ 找不到初始化 checkpoint: {INIT_CKPT}")

        print(f"🔁 正在从 M5-Delta-H4 checkpoint 初始化 M6-FieldWiseEncoder 共享权重: {INIT_CKPT}")
        ckpt = torch.load(INIT_CKPT, map_location=DEVICE)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("✅ 初始化权重加载完成（strict=False）")
        print(f"   missing_keys: {missing_keys}")
        print(f"   unexpected_keys: {unexpected_keys}")
        print("   说明：field_encoders / fusion 是新模块；PlainFNO 的 p 层会作为 unexpected_keys 被跳过。")
        print("✅ M5-Delta-H4 共享权重加载成功，将训练 M6-FieldWiseEncoder-H4")

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

    print("\n🔥 开始 M6-FieldWiseEncoder-H4 多步自回归训练...")

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

        step_msg = " | ".join(
            [f"Val t+{i+1}: {val_step_losses[i]:.6f}" for i in range(ROLLOUT_STEPS)]
        )

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"Train MStep RelL2: {train_loss:.6f} | "
            f"Val MStep RelL2: {val_loss:.6f} | "
            f"{step_msg} | "
            f"Grad Norm: {avg_grad_norm:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        log_dict = {
            "epoch": epoch,
            "Train MultiStep Rel-L2": train_loss,
            "Val MultiStep Rel-L2": val_loss,
            "Grad Norm": avg_grad_norm,
            "Learning Rate": current_lr,
            "Best Val MultiStep Rel-L2": current_best,
        }

        for i in range(ROLLOUT_STEPS):
            log_dict[f"Train step t+{i+1} Rel-L2"] = float(train_step_losses[i])
            log_dict[f"Val step t+{i+1} Rel-L2"] = float(val_step_losses[i])

        wandb.log(log_dict)

        checkpoint_payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'experiment': 'M6-FieldWiseEncoder-H4',
            'base_model': 'M3-Delta',
            'prediction_type': 'multi_step_delta_prediction',
            'normalization': 'field-wise mean/std',
            'task': 'autoregressive multi-step delta prediction',
            'delta_definition': 'pred_next_norm = current_state_norm + pred_delta_norm',
            'loss_function': 'Weighted multi-step FieldWiseRelativeL2Loss on normalized state',
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_payload['best_val_loss'] = best_val_loss
            torch.save(checkpoint_payload, BEST_SAVE_PATH)
            print(f"  🌟 [New Best] 权重已保存至: {BEST_SAVE_PATH}")

        scheduler.step()

    total_time = time.time() - start_time

    print(f"\n✅ M6-FieldWiseEncoder-H4 训练完成!")
    print(f"📌 最优模型: {BEST_SAVE_PATH}")
    print(f"⏱️ 总耗时: {total_time / 60:.2f} 分钟")

    wandb.finish()


if __name__ == "__main__":
    main()
