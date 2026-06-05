import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.rbc_dataset import RBCDataset
from models.operators.fno2d import PlainFNO2d


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_grad_norm = 0.0
    valid_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        if not torch.isfinite(loss):
            print("⚠️ Non-finite loss detected. Skipping batch.")
            continue

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()
        total_grad_norm += grad_norm.item()
        valid_batches += 1

    if valid_batches == 0:
        return 0.0, 0.0

    return total_loss / valid_batches, total_grad_norm / valid_batches


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    valid_batches = 0

    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        if not torch.isfinite(loss):
            continue

        total_loss += loss.item()
        valid_batches += 1

    if valid_batches == 0:
        return float("inf")

    return total_loss / valid_batches


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 16
    epochs = 50
    lr = 3e-4
    weight_decay = 1e-4
    eta_min = 1e-5

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    split_file = os.path.join(root, "data", "splits", "iid_split.json")
    stats_file = os.path.join(root, "data", "stats", "rbc_field_stats.json")
    ckpt_dir = os.path.join(root, "checkpoints", "controlled")

    os.makedirs(ckpt_dir, exist_ok=True)

    best_ckpt_path = os.path.join(
        ckpt_dir,
        "m3_mse_controlled_best.pth"
    )

    if not os.path.exists(stats_file):
        raise FileNotFoundError(
            f"找不到 {stats_file}，请先运行 python scripts/compute_field_stats.py"
        )

    with open(split_file, "r", encoding="utf-8") as f:
        split_config = json.load(f)

    print("🚀 启动 [M3-MSE-Controlled] Field-wise Normalized FNO 训练")
    print(f"Device: {device}")
    print(f"Stats: {stats_file}")

    wandb.init(
        project="DC-MNO",
        name="M3-MSE-Controlled-LR3e4-Cosine-Clip1",
        config={
            "experiment_type": "controlled_phase1",
            "architecture": "Plain FNO",
            "normalization": "Field-wise mean/std",
            "loss_function": "MSE",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "scheduler": "CosineAnnealingLR",
            "eta_min": eta_min,
            "grad_clip": 1.0,
            "split": "iid_split.json",
            "stats_file": stats_file,
        }
    )

    train_dataset = RBCDataset(
        split_config=split_config["train"],
        normalize=True,
        stats_path=stats_file,
    )

    val_dataset = RBCDataset(
        split_config=split_config["val"],
        normalize=True,
        stats_path=stats_file,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = PlainFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=eta_min,
    )

    best_val_loss = float("inf")
    best_save_path = ""

    print("\n🔥 开始 M3-MSE-Controlled 训练...")
    for epoch in range(1, epochs + 1):
        train_loss, grad_norm = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        current_lr = optimizer.param_groups[0]["lr"]
        current_best = min(best_val_loss, val_loss)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train MSE: {train_loss:.6e} | "
            f"Val MSE: {val_loss:.6e} | "
            f"Grad Norm: {grad_norm:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        wandb.log({
            "epoch": epoch,
            "Train MSE": train_loss,
            "Val MSE": val_loss,
            "Grad Norm": grad_norm,
            "Learning Rate": current_lr,
            "Best Val MSE": current_best,
        })

        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "experiment": "M3-MSE-Controlled Field-wise Normalized FNO",
                    "normalization": "field-wise mean/std",
                    "loss_function": "MSE",
                    "stats_file": stats_file,
                    "training_protocol": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "weight_decay": weight_decay,
                        "scheduler": "CosineAnnealingLR",
                        "eta_min": eta_min,
                        "grad_clip": 1.0,
                    }
                },
                os.path.join(
                    ckpt_dir,
                    f"m3_mse_controlled_epoch_{epoch:02d}.pth"
                )
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_save_path = best_ckpt_path

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "experiment": "M3-MSE-Controlled Field-wise Normalized FNO",
                    "normalization": "field-wise mean/std",
                    "loss_function": "MSE",
                    "stats_file": stats_file,
                    "training_protocol": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "weight_decay": weight_decay,
                        "scheduler": "CosineAnnealingLR",
                        "eta_min": eta_min,
                        "grad_clip": 1.0,
                    }
                },
                best_ckpt_path
            )

            print(f"   [+] 更新 best checkpoint: {best_ckpt_path}")

        scheduler.step()

    print(f"\n✅ M3-MSE-Controlled 训练完成！最优模型保存至: {best_save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()