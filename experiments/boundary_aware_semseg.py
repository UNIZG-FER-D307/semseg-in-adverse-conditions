import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import JaccardIndex
from utils import BoundaryAwareCrossEntropy, BoundaryAwareFocalLoss
import os
import psutil


def get_total_memory():
    current_process = psutil.Process(os.getpid())
    mem = current_process.memory_info().rss
    for child in current_process.children(recursive=True):
        mem += child.memory_info().rss
    return mem


class BoundaryAwareSemseg(pl.LightningModule):
    def __init__(
        self, model, optimizer, scheduler, ignore_id, num_classes, loss_type="FL"
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ignore_id = ignore_id
        self.num_classes = num_classes
        assert loss_type in ["FL", "CE"]
        self.loss = self._construct_loss(loss_type)

        self.iou_train = JaccardIndex(
            num_classes=num_classes + 1, ignore_index=num_classes, task="multiclass"
        )
        self.iou_val = JaccardIndex(
            num_classes=num_classes + 1, ignore_index=num_classes, task="multiclass"
        )

    def _construct_loss(self, loss_type):
        if loss_type == "FL":
            loss = BoundaryAwareFocalLoss(
                num_classes=self.num_classes, ignore_id=self.ignore_id, check_nans=True
            )
        else:
            loss = BoundaryAwareCrossEntropy(
                num_classes=self.num_classes, ignore_id=self.ignore_id
            )
        return loss

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logit = self.model(x)
        return logit

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y, boundary_alpha = batch
        y = y[:, 0].detach()
        logit = self.forward(x)
        if isinstance(logit, tuple):
            logit = logit[0]
        loss = self.loss(logit, y, boundary_alpha)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        y = y.detach().clone()
        y[y == self.ignore_id] = self.num_classes
        N, _, H, W = logit.shape
        logit = torch.cat((logit, torch.zeros(N, 1, H, W).to(logit)), dim=1)
        self.iou_train(logit, y)
        return loss

    def _pad_input(self, x):
        H, W = x.shape[-2:]
        pad_H = (128 - H % 128) // 2 if H % 128 != 0 else 0
        pad_W = (128 - W % 128) // 2 if W % 128 != 0 else 0
        x = torch.nn.functional.pad(x, (pad_W, pad_W, pad_H, pad_H), "constant", 0.0)
        return x

    def _unpad_output(self, x, desired_H, desired_W):
        H, W = x.shape[-2:]
        dH = (H - desired_H) // 2
        dW = (W - desired_W) // 2

        x = x[:, :, dH:-dH, :] if dH > 0 else x
        x = x[:, :, :, dW:-dW] if dW > 0 else x
        return x

    def validation_step(self, batch, batch_idx):
        x, y, boundary_alpha = batch
        H, W = x.shape[-2:]
        x = self._pad_input(x)
        y = y[:, 0]
        logit = self.forward(x)
        if isinstance(logit, tuple):
            logit = logit[0]
        logit = self._unpad_output(logit, H, W)
        val_loss = self.loss(logit, y, boundary_alpha)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        y[y == self.ignore_id] = self.num_classes
        N, _, H, W = logit.shape
        logit = torch.cat((logit, torch.zeros(N, 1, H, W).to(logit)), dim=1)
        self.iou_val(logit, y)
        return val_loss

    def on_train_epoch_end(self):
        miou = self.iou_train.compute().item()
        self.log("train_mIoU", round(miou * 100.0, 2))
        self.print("Training mIoU", round(miou * 100.0, 2))
        self.iou_train.reset()

    def on_validation_epoch_end(self):
        miou = self.iou_val.compute().item()
        self.log("val_mIoU", round(miou * 100.0, 2))
        self.print("Validation mIoU", round(miou * 100.0, 2))
        self.iou_val.reset()

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler
        if scheduler != None:
            return [optimizer], [scheduler]
        return optimizer
