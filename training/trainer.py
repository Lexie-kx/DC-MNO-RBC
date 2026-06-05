import os
import torch
import torch.nn.functional as F

class Trainer:
    """
    通用 FNO 训练器
    支持:
    - 自定义 criterion
    - scheduler
    - gradient clipping
    - NaN/Inf 防护
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        criterion=None,
        scheduler=None,
        save_dir="checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir

        # 默认 MSE
        self.criterion = criterion if criterion is not None else F.mse_loss

        self.scheduler = scheduler

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        total_grad_norm = 0.0
        valid_batches = 0

        for batch_x, batch_y in self.train_loader:

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(batch_x)

            loss = self.criterion(pred, batch_y)

            # NaN / Inf 防护
            if not torch.isfinite(loss):
                print("⚠️ 检测到 NaN/Inf loss，跳过当前 batch")
                continue

            loss.backward()

            # Gradient Clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm.item()

            valid_batches += 1

        if valid_batches == 0:
            return 0.0, 0.0

        avg_loss = total_loss / valid_batches
        avg_grad_norm = total_grad_norm / valid_batches

        return avg_loss, avg_grad_norm

    def validate(self):
        self.model.eval()

        total_loss = 0.0
        valid_batches = 0

        with torch.no_grad():

            for batch_x, batch_y in self.val_loader:

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                pred = self.model(batch_x)

                loss = self.criterion(pred, batch_y)

                # NaN / Inf 防护
                if not torch.isfinite(loss):
                    continue

                total_loss += loss.item()
                valid_batches += 1

        if valid_batches == 0:
            return float("inf")

        avg_loss = total_loss / valid_batches

        return avg_loss

    def save_checkpoint(self, epoch, val_loss, filename="model.pth"):

        save_path = os.path.join(self.save_dir, filename)

        criterion_name = (
            self.criterion.__name__
            if hasattr(self.criterion, "__name__")
            else self.criterion.__class__.__name__
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "criterion": criterion_name,
            "val_loss": val_loss
        }

        torch.save(checkpoint, save_path)

        return save_path