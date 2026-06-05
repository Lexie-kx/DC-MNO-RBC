import os
import sys
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.rbc_dataset import RBCDataset
from models.operators.fno2d import PlainFNO2d
from training.trainer import Trainer
from training.metrics import FieldWiseRelativeL2Loss


class ParamChannelDataset(Dataset):
    """
    M2-RelL2-ParamNorm-Controlled

    把 RBCDataset 返回的 (x, y, param) 转成 (x_with_param, y)。

    原始:
        x:     [16, H, W]
        y:     [4, H, W]
        param: [2] = [log10(Ra), log10(Pr)]

    参数归一化:
        Ra = 1e6, 1e7, 1e8
        log10(Ra) = 6, 7, 8
        ra_norm = (log10(Ra) - 7.0) / 1.0
        对应 -1, 0, 1

        Pr = 0.5, 1, 2
        log10(Pr) = -0.30103, 0, 0.30103
        pr_norm = log10(Pr) / 0.30103
        对应 -1, 0, 1

    转换后:
        x_with_param: [18, H, W]
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y, param = self.base_dataset[idx]

        # param[0] = log10(Ra)
        # param[1] = log10(Pr)
        param_norm = param.clone()
        param_norm[0] = (param_norm[0] - 7.0) / 1.0
        param_norm[1] = param_norm[1] / 0.30103

        _, H, W = x.shape

        # param_norm: [2] -> [2, H, W]
        param_map = param_norm[:, None, None].expand(2, H, W)

        # [16, H, W] + [2, H, W] -> [18, H, W]
        x = torch.cat([x, param_map], dim=0)

        return x, y


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50 
    
    print(f"🚀 启动 [M2-RelL2-ParamNorm-Controlled] 训练 | 设备: {DEVICE}")
    print("👉 M2 = M1 + 归一化 Ra/Pr 参数通道")
    print("👉 输入通道: 16 + 2 = 18")
    print("👉 Normalization: None")
    print("👉 Parameter Normalization: Ra/Pr -> [-1, 0, 1]")
    print("👉 Loss: FieldWiseRelativeL2Loss")

    wandb.init(
        project="DC-MNO",
        name="M2-RelL2-ParamNorm-Controlled-LR3e4-Cosine-Clip1",
        config={
            "experiment_type": "controlled_phase1_param_ablation",
            "architecture": "Parameter-conditioned Plain FNO",
            "normalization": "None",
            "parameter_conditioning": "normalized log10(Ra), log10(Pr) as constant channels",
            "parameter_normalization": {
                "ra_norm": "(log10(Ra) - 7.0) / 1.0",
                "pr_norm": "log10(Pr) / 0.30103"
            },
            "input_channels": 18,
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

    train_base_dataset = RBCDataset(
        split_config=split_config["train"],
        normalize=False,
        return_params=True
    )

    val_base_dataset = RBCDataset(
        split_config=split_config["val"],
        normalize=False,
        return_params=True
    )

    train_dataset = ParamChannelDataset(train_base_dataset)
    val_dataset = ParamChannelDataset(val_base_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    sample_x, sample_y = next(iter(train_loader))
    print(f"✅ M2-ParamNorm 输入检查: X shape = {sample_x.shape}，应为 [B, 18, 256, 64]")
    print(f"✅ M2-ParamNorm 目标检查: Y shape = {sample_y.shape}，应为 [B, 4, 256, 64]")

    # 看一下最后两个参数通道的数值，确认已经归一化
    print(f"👉 参数通道示例: {sample_x[0, -2:, 0, 0].tolist()}，期望大致在 [-1, 0, 1]")

    model = PlainFNO2d(
        in_channels=18,
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

    print("\n🔥 开始 M2-RelL2-ParamNorm-Controlled 训练...")

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
                filename=f"m2_rel_l2_paramnorm_controlled_epoch_{epoch:02d}.pth"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_save_path = trainer.save_checkpoint(
                epoch,
                val_loss,
                filename="m2_rel_l2_paramnorm_controlled_best.pth"
            )
            print(f"   [+] 更新 best checkpoint: {best_save_path}")

        scheduler.step()

    print(f"\n✅ 训练完成！最优模型保存至: {best_save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()