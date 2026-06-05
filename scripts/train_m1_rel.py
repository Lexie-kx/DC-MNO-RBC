import os
import sys
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.rbc_dataset import RBCDataset
from models.operators.fno2d import PlainFNO2d
from training.trainer import Trainer
from training.metrics import FieldWiseRelativeL2Loss


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50

    print(f"🚀 启动 [M1-RelL2-Controlled] 训练 | 设备: {DEVICE}")

    wandb.init(
        project="DC-MNO",
        name="M1-RelL2-Controlled-LR3e4-Cosine-Clip1",
        config={
            "experiment_type": "controlled_phase1",
            "architecture": "Plain FNO",
            "normalization": "None",
            "loss_function": "FieldWise Relative L2",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "eta_min": 1e-5,
            "grad_clip": 1.0,
            "split": "iid_split.json",
        }
    )

    split_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'splits', 'iid_split.json')
    )

    with open(split_file, 'r', encoding='utf-8') as f:
        split_config = json.load(f)

    train_dataset = RBCDataset(split_config=split_config["train"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_dataset = RBCDataset(split_config=split_config["val"])
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = PlainFNO2d(
        in_channels=16,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-5
    )

    criterion = FieldWiseRelativeL2Loss()


    trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=DEVICE,
    criterion=criterion,
    scheduler=scheduler,
    save_dir=os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'controlled')
    )
)

    best_val_loss = float("inf")
    best_save_path = ""

    print("\n🔥 开始 M1-RelL2-Controlled 训练...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, grad_norm = trainer.train_one_epoch()
        val_loss = trainer.validate()

        current_lr = optimizer.param_groups[0]["lr"]
        current_best = min(best_val_loss, val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Rel-L2: {train_loss:.6f} | "
            f"Val Rel-L2: {val_loss:.6f} | "
            f"Grad Norm: {grad_norm:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        wandb.log({
            "epoch": epoch,
            "Train Rel-L2": train_loss,
            "Val Rel-L2": val_loss,
            "Grad Norm": grad_norm,
            "Learning Rate": current_lr,
            "Best Val Rel-L2": current_best,
        })

        if epoch % 5 == 0:
            trainer.save_checkpoint(
                epoch,
                val_loss,
                filename=f"m1_rel_l2_controlled_epoch_{epoch:02d}.pth"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_save_path = trainer.save_checkpoint(
                epoch,
                val_loss,
                filename="m1_rel_l2_controlled_best.pth"
            )
            print(f"   [+] 更新 best checkpoint: {best_save_path}")

        scheduler.step()

    print(f"\n✅ 训练完成！最优模型保存至: {best_save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()