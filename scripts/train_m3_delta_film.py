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
from models.operators.fno2d_film import FiLMFNO2d
from training.metrics import FieldWiseRelativeL2Loss


class DeltaParamDataset(Dataset):
    """
    M3-Delta + FiLM

    base_dataset 返回:
        x_norm: [16, H, W]
        y_norm: [4, H, W]
        param:  [2] = [log10(Ra), log10(Pr)]

    target:
        delta_norm = y_norm - x_last_norm

    返回:
        x_norm:     [16, H, W]
        delta_norm: [4, H, W]
        param:      [2]
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x_norm, y_norm, param = self.base_dataset[idx]

        if x_norm.shape[0] != 16:
            raise ValueError(
                f"原始输入通道应为 16，但现在是 {x_norm.shape[0]}"
            )

        if param.shape[0] != 2:
            raise ValueError(
                f"param 应为 [log10(Ra), log10(Pr)]，但现在 shape={param.shape}"
            )

        x_last_norm = x_norm[-4:, :, :]
        delta_norm = y_norm - x_last_norm

        return x_norm, delta_norm, param


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train M3-Delta-FiLM model with configurable split/stats."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="data/splits/iid_split.json",
        help="Path to split json file, relative to project root or absolute path."
    )

    parser.add_argument(
        "--stats",
        type=str,
        default="data/stats/rbc_field_stats.json",
        help="Path to field stats json file, relative to project root or absolute path."
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="m3_delta_film_iid",
        help="Run name for checkpoint and wandb."
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints/cross_param",
        help="Checkpoint directory, relative to project root or absolute path."
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-5)
    parser.add_argument("--film_hidden_dim", type=int, default=64)

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

    SPLIT_PATH = resolve_path(project_root, args.split)
    STATS_PATH = resolve_path(project_root, args.stats)
    CKPT_DIR = resolve_path(project_root, args.ckpt_dir)

    os.makedirs(CKPT_DIR, exist_ok=True)

    BEST_SAVE_PATH = os.path.join(CKPT_DIR, f"{RUN_NAME}_best.pth")

    print(f"🚀 [M3-Delta-FiLM] 启动训练 | 设备: {DEVICE}")
    print("👉 Base: M3 = Field-wise Normalized FNO")
    print("👉 Task: predict delta = X_{t+1} - X_t")
    print("👉 Parameter conditioning: FiLM")
    print("👉 Input channels: 16")
    print("👉 Param: [log10(Ra), log10(Pr)]")
    print("👉 FiLM internally uses: log10(Ra), log10(Pr), log10(nu), log10(kappa)")
    print("👉 Output channels: 4 delta fields")
    print("👉 Loss: FieldWiseRelativeL2Loss on normalized delta")
    print(f"📌 Split: {SPLIT_PATH}")
    print(f"📌 Stats: {STATS_PATH}")
    print(f"📌 Run name: {RUN_NAME}")
    print(f"📌 Best checkpoint: {BEST_SAVE_PATH}")

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    wandb.init(
        project="DC-MNO",
        name=RUN_NAME,
        config={
            "experiment_type": "cross_parameter_delta_prediction_film",
            "architecture": "FiLM-conditioned FNO",
            "normalization": "Field-wise mean/std",
            "task": "delta prediction",
            "delta_definition": "delta_norm = y_norm - x_last_norm",
            "parameter_conditioning": "FiLM",
            "param_input": ["log10_Ra", "log10_Pr"],
            "film_internal_param": [
                "log10_Ra",
                "log10_Pr",
                "log10_nu",
                "log10_kappa",
            ],
            "input_channels": 16,
            "output_channels": 4,
            "film_hidden_dim": args.film_hidden_dim,
            "loss_function": "FieldWiseRelativeL2Loss on delta",
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
        }
    )

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"找不到 split 文件: {SPLIT_PATH}")

    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(f"找不到统计量文件: {STATS_PATH}")

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split_config = json.load(f)

    print("📦 正在加载数据集：Field-wise Normalization + Delta target + FiLM params")

    train_base_dataset = RBCDataset(
        split_config=split_config["train"],
        normalize=True,
        stats_path=STATS_PATH,
        return_params=True,
    )

    val_base_dataset = RBCDataset(
        split_config=split_config["val"],
        normalize=True,
        stats_path=STATS_PATH,
        return_params=True,
    )

    train_dataset = DeltaParamDataset(train_base_dataset)
    val_dataset = DeltaParamDataset(val_base_dataset)

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

    sample_x, sample_delta, sample_param = next(iter(train_loader))
    print(f"✅ FiLM 输入检查: X shape = {sample_x.shape}，应为 [B, 16, 256, 64]")
    print(f"✅ Delta 目标检查: delta shape = {sample_delta.shape}，应为 [B, 4, 256, 64]")
    print(f"✅ Param 检查: param shape = {sample_param.shape}，应为 [B, 2]")
    print(f"👉 示例 param = [log10(Ra), log10(Pr)] = {sample_param[0].tolist()}")
    print(f"👉 Delta target mean: {sample_delta.mean().item():.6f}")
    print(f"👉 Delta target std:  {sample_delta.std().item():.6f}")

    model = FiLMFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
        film_hidden_dim=args.film_hidden_dim,
    ).to(DEVICE)

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

    best_val_loss = float("inf")
    start_time = time.time()

    print("\n🔥 开始 M3-Delta-FiLM 训练...")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss = 0.0
        total_grad_norm = 0.0
        num_train = 0
        num_batches = 0

        for batch_x_norm, batch_delta_norm, batch_param in train_loader:
            batch_x_norm = batch_x_norm.to(DEVICE)
            batch_delta_norm = batch_delta_norm.to(DEVICE)
            batch_param = batch_param.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            pred_delta_norm = model(batch_x_norm, batch_param)
            loss = criterion(pred_delta_norm, batch_delta_norm)

            if not torch.isfinite(loss):
                print("⚠️ 检测到 NaN/Inf loss，跳过当前 batch")
                continue

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_size = batch_x_norm.size(0)
            train_loss += loss.item() * batch_size
            total_grad_norm += grad_norm.item()
            num_train += batch_size
            num_batches += 1

        train_loss = train_loss / max(num_train, 1)
        avg_grad_norm = total_grad_norm / max(num_batches, 1)

        model.eval()
        val_loss = 0.0
        num_val = 0

        with torch.no_grad():
            for batch_x_norm, batch_delta_norm, batch_param in val_loader:
                batch_x_norm = batch_x_norm.to(DEVICE)
                batch_delta_norm = batch_delta_norm.to(DEVICE)
                batch_param = batch_param.to(DEVICE)

                pred_delta_norm = model(batch_x_norm, batch_param)
                loss = criterion(pred_delta_norm, batch_delta_norm)

                if not torch.isfinite(loss):
                    continue

                batch_size = batch_x_norm.size(0)
                val_loss += loss.item() * batch_size
                num_val += batch_size

        val_loss = val_loss / max(num_val, 1)

        current_lr = optimizer.param_groups[0]["lr"]
        current_best = min(best_val_loss, val_loss)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"Train Delta RelL2: {train_loss:.6f} | "
            f"Val Delta RelL2: {val_loss:.6f} | "
            f"Grad Norm: {avg_grad_norm:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        wandb.log({
            "epoch": epoch,
            "Train Delta Rel-L2": train_loss,
            "Val Delta Rel-L2": val_loss,
            "Grad Norm": avg_grad_norm,
            "Learning Rate": current_lr,
            "Best Val Delta Rel-L2": current_best,
        })

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "experiment": "M3-Delta-FiLM",
            "prediction_type": "delta_prediction",
            "normalization": "field-wise mean/std",
            "task": "delta prediction",
            "delta_definition": "delta_norm = y_norm - x_last_norm",
            "parameter_conditioning": "FiLM",
            "param_input": ["log10_Ra", "log10_Pr"],
            "film_internal_param": [
                "log10_Ra",
                "log10_Pr",
                "log10_nu",
                "log10_kappa",
            ],
            "input_channels": 16,
            "output_channels": 4,
            "film_hidden_dim": args.film_hidden_dim,
            "loss_function": "FieldWiseRelativeL2Loss",
            "split_path": SPLIT_PATH,
            "stats_path": STATS_PATH,
            "run_name": RUN_NAME,
            "training_protocol": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "scheduler": "CosineAnnealingLR",
                "eta_min": ETA_MIN,
                "grad_clip": 1.0,
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
            checkpoint_payload["best_val_loss"] = best_val_loss

            torch.save(checkpoint_payload, BEST_SAVE_PATH)
            print(f"  🌟 [New Best] 权重已保存至: {BEST_SAVE_PATH}")

        scheduler.step()

    total_time = time.time() - start_time

    print("\n✅ M3-Delta-FiLM 训练完成!")
    print(f"📌 最优模型: {BEST_SAVE_PATH}")
    print(f"⏱️ 总耗时: {total_time / 60:.2f} 分钟")

    wandb.finish()


if __name__ == "__main__":
    main()
