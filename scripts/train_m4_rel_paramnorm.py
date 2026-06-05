import os
import sys
import json
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.rbc_dataset import RBCDataset
from models.operators.fno2d import PlainFNO2d
from training.metrics import FieldWiseRelativeL2Loss


class ParamNormChannelDataset(Dataset):
    """
    M4-RelL2-ParamNorm-Controlled

    base_dataset 返回:
        x:     [16, H, W]  已经由 RBCDataset 按 field-wise mean/std 归一化
        y:     [4, H, W]   已经由 RBCDataset 按 field-wise mean/std 归一化
        param: [2] = [log10(Ra), log10(Pr)]

    这里把 param 归一化后拼到 x 后面:

        Ra = 1e6, 1e7, 1e8
        log10(Ra) = 6, 7, 8
        ra_norm = (log10(Ra) - 7.0) / 1.0
        对应 -1, 0, 1

        Pr = 0.5, 1, 2
        log10(Pr) = -0.30103, 0, 0.30103
        pr_norm = log10(Pr) / 0.30103
        对应 -1, 0, 1

    输出:
        x_with_param: [18, H, W]
        y:            [4, H, W]
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y, param = self.base_dataset[idx]

        param_norm = param.clone()
        param_norm[0] = (param_norm[0] - 7.0) / 1.0
        param_norm[1] = param_norm[1] / 0.30103

        _, H, W = x.shape
        param_map = param_norm[:, None, None].expand(2, H, W)

        x = torch.cat([x, param_map], dim=0)

        return x, y


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    EPOCHS = 50  # 先跑 1 轮测试，确认无误后改成 50
    WEIGHT_DECAY = 1e-4
    ETA_MIN = 1e-5

    SPLIT_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'splits', 'iid_split.json')
    )

    STATS_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'stats', 'rbc_field_stats.json')
    )

    CKPT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'controlled')
    )
    os.makedirs(CKPT_DIR, exist_ok=True)

    BEST_SAVE_PATH = os.path.join(CKPT_DIR, "m4_rel_l2_paramnorm_controlled_best.pth")

    print(f"🚀 [M4-RelL2-ParamNorm-Controlled] 启动训练 | 设备: {DEVICE}")
    print("👉 M4 = M3 + normalized Ra/Pr 参数通道")
    print("👉 Field-wise normalization: ON")
    print("👉 Parameter channels: normalized log10(Ra), log10(Pr)")
    print("👉 Input channels: 16 + 2 = 18")
    print("👉 Loss: FieldWiseRelativeL2Loss")

    wandb.init(
        project="DC-MNO",
        name="M4-RelL2-ParamNorm-Controlled-LR3e4-Cosine-Clip1",
        config={
            "experiment_type": "controlled_phase1_param_ablation",
            "architecture": "Parameter-conditioned Field-wise Normalized FNO",
            "normalization": "Field-wise mean/std",
            "parameter_conditioning": "normalized log10(Ra), log10(Pr) as constant channels",
            "parameter_normalization": {
                "ra_norm": "(log10(Ra) - 7.0) / 1.0",
                "pr_norm": "log10(Pr) / 0.30103"
            },
            "input_channels": 18,
            "loss_function": "FieldWiseRelativeL2Loss",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "eta_min": ETA_MIN,
            "grad_clip": 1.0,
            "split": "iid_split.json",
            "stats_path": STATS_PATH,
        }
    )

    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(
            f"❌ 找不到统计量文件: {STATS_PATH}，请先运行 python scripts/compute_field_stats.py"
        )

    with open(SPLIT_PATH, 'r', encoding='utf-8') as f:
        split_config = json.load(f)

    print("📦 正在加载数据集：Field-wise Normalization + ParamNorm 已开启")

    train_base_dataset = RBCDataset(
        split_config=split_config["train"],
        normalize=True,
        stats_path=STATS_PATH,
        return_params=True
    )

    val_base_dataset = RBCDataset(
        split_config=split_config["val"],
        normalize=True,
        stats_path=STATS_PATH,
        return_params=True
    )

    train_dataset = ParamNormChannelDataset(train_base_dataset)
    val_dataset = ParamNormChannelDataset(val_base_dataset)

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

    sample_x, sample_y = next(iter(train_loader))
    print(f"✅ M4 输入检查: X shape = {sample_x.shape}，应为 [B, 18, 256, 64]")
    print(f"✅ M4 目标检查: Y shape = {sample_y.shape}，应为 [B, 4, 256, 64]")
    print(f"👉 参数通道示例: {sample_x[0, -2:, 0, 0].tolist()}，应大致在 [-1, 0, 1]")

    model = PlainFNO2d(
        in_channels=18,
        out_channels=4,
        modes1=16,
        modes2=16,
        width=32
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

    best_val_loss = float('inf')
    start_time = time.time()

    print("\n🔥 开始 M4-RelL2-ParamNorm-Controlled 训练...")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss = 0.0
        total_grad_norm = 0.0
        num_train = 0
        num_batches = 0

        for batch_x_norm, batch_y_norm in train_loader:
            batch_x_norm = batch_x_norm.to(DEVICE)
            batch_y_norm = batch_y_norm.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            pred_norm = model(batch_x_norm)
            loss = criterion(pred_norm, batch_y_norm)

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
            for batch_x_norm, batch_y_norm in val_loader:
                batch_x_norm = batch_x_norm.to(DEVICE)
                batch_y_norm = batch_y_norm.to(DEVICE)

                pred_norm = model(batch_x_norm)
                loss = criterion(pred_norm, batch_y_norm)

                if not torch.isfinite(loss):
                    continue

                batch_size = batch_x_norm.size(0)
                val_loss += loss.item() * batch_size
                num_val += batch_size

        val_loss = val_loss / max(num_val, 1)

        current_lr = optimizer.param_groups[0]['lr']
        current_best = min(best_val_loss, val_loss)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"Train RelL2: {train_loss:.6f} | "
            f"Val RelL2: {val_loss:.6f} | "
            f"Grad Norm: {avg_grad_norm:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        wandb.log({
            "epoch": epoch,
            "Train Rel-L2": train_loss,
            "Val Rel-L2": val_loss,
            "Grad Norm": avg_grad_norm,
            "Learning Rate": current_lr,
            "Best Val Rel-L2": current_best,
        })

        if epoch % 5 == 0:
            epoch_save_path = os.path.join(
                CKPT_DIR,
                f"m4_rel_l2_paramnorm_controlled_epoch_{epoch:02d}.pth"
            )

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'experiment': 'M4-RelL2-ParamNorm-Controlled',
                'normalization': 'field-wise mean/std',
                'parameter_conditioning': 'normalized log10(Ra), log10(Pr) constant channels',
                'loss_function': 'FieldWiseRelativeL2Loss',
                'stats_path': STATS_PATH,
                'training_protocol': {
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'weight_decay': WEIGHT_DECAY,
                    'scheduler': 'CosineAnnealingLR',
                    'eta_min': ETA_MIN,
                    'grad_clip': 1.0,
                }
            }, epoch_save_path)

            print(f"   [*] 保存周期 checkpoint: {epoch_save_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'experiment': 'M4-RelL2-ParamNorm-Controlled',
                'normalization': 'field-wise mean/std',
                'parameter_conditioning': 'normalized log10(Ra), log10(Pr) constant channels',
                'loss_function': 'FieldWiseRelativeL2Loss',
                'stats_path': STATS_PATH,
                'training_protocol': {
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'weight_decay': WEIGHT_DECAY,
                    'scheduler': 'CosineAnnealingLR',
                    'eta_min': ETA_MIN,
                    'grad_clip': 1.0,
                }
            }, BEST_SAVE_PATH)

            print(f"  🌟 [New Best] 权重已保存至: {BEST_SAVE_PATH}")

        scheduler.step()

    total_time = time.time() - start_time

    print(f"\n✅ M4-RelL2-ParamNorm-Controlled 训练完成!")
    print(f"📌 最优模型: {BEST_SAVE_PATH}")
    print(f"⏱️ 总耗时: {total_time / 60:.2f} 分钟")

    wandb.finish()


if __name__ == "__main__":
    main()