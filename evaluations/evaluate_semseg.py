import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import CalibrationError, JaccardIndex

from data.datasets.cityscapes.cityscapes_labels import create_id_to_name


class SemsegEvaluator(pl.LightningModule):
    def __init__(
        self,
        model,
        ignore_id,
        num_classes,
        multiscale=False,
        flip=False,
        compute_ece=False,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.iou_val = JaccardIndex(
            num_classes=num_classes + 1,
            ignore_index=num_classes,
            average="none",
            task="multiclass",
        )
        self.ece = CalibrationError(
            n_bins=11,
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_id,
        )
        self.scales = [1.0, 1.25, 1.5, 1.75, 2]
        self.stats = torch.load("./stats.pt")
        # self.scales = [1.75]
        self.multiscale = multiscale
        self.compute_ece = compute_ece
        self.flip = flip

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logit = self.model(x)
        return logit

    def _pad_input(self, x):
        H, W = x.shape[-2:]
        pad_H = (128 - H % 128) // 2 if H % 128 != 0 else 0
        pad_W = (128 - W % 128) // 2 if W % 128 != 0 else 0
        if pad_W == 0 and pad_H == 0:
            return x
        x = torch.nn.functional.pad(x, (pad_W, pad_W, pad_H, pad_H), "constant", 0.0)
        return x

    def _unpad_output(self, x, desired_H, desired_W):
        H, W = x.shape[-2:]
        dH = (H - desired_H) // 2
        dW = (W - desired_W) // 2

        x = x[:, :, dH:-dH, :] if dH > 0 else x
        x = x[:, :, :, dW:-dW] if dW > 0 else x
        return x

    def _multiscale_test_step(self, x_):
        N, _, H, W = x_.shape
        final_logits = torch.zeros(N, self.num_classes, H, W).to(x_)
        for i, scale in enumerate(self.scales):
            x = F.interpolate(
                x_, scale_factor=scale, mode="bilinear", align_corners=False
            )
            dh, dw = x.shape[-2:]
            x = self._pad_input(x)
            logit = self.forward(x)
            if self.flip:
                x_flip = torch.flip(x, [-1])
                flipped_logit = self.forward(x_flip)
                logit += torch.flip(flipped_logit, [-1])
                logit = logit / 2.0
            logit = self._unpad_output(logit, dh, dw)
            logit = F.interpolate(
                logit, size=(H, W), mode="bilinear", align_corners=False
            )
            # logit = logit * self.stats[i].view(1, 19, 1, 1).to(logit)
            final_logits += logit
        return final_logits / len(self.scales)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, 0]
        if self.multiscale:
            logit = self._multiscale_test_step(x)
        else:
            init_H, init_W = x.shape[-2:]
            x = self._pad_input(x)
            logit = self.forward(x)
            logit = self._unpad_output(logit, init_H, init_W)
        # FIXME
        y[y == self.ignore_id] = self.num_classes
        N, _, H, W = logit.shape
        logit_ = torch.cat((logit, torch.zeros(N, 1, H, W).to(logit)), dim=1)
        self.iou_val(logit_, y)
        if self.compute_ece:
            probs = logit.softmax(1)
            invalid_mask = (
                F.one_hot(y, self.num_classes + 1)
                .permute(0, 3, 1, 2)[:, -1]
                .unsqueeze(1)
                .repeat(1, self.num_classes, 1, 1)
            )
            # probs = probs * (1 - invalid_mask)
            probs = torch.cat(
                (probs, torch.ones(N, 1, H, W).to(probs) * invalid_mask[:, :1]), dim=1
            )
            self.ece.update(probs, y)

    def on_test_epoch_end(self):
        iou = self.iou_val.compute()
        miou = iou[:-1].mean().item()
        self.log("test_mIoU", round(miou * 100.0, 2))
        self.print("Test mIoU", round(miou * 100.0, 2))
        # self._plot_confmat()
        id2n = create_id_to_name()
        for i, c_iou in enumerate(iou.cpu().numpy().tolist()[:-1]):
            self.print(id2n[i], round(c_iou * 100.0, 2))

        self.iou_val.reset()
        if self.compute_ece:
            ece = self.ece.compute().item()
            self.log("Test ECE", round(ece * 100.0, 2))
            self.print("Test ECE", round(ece * 100.0, 2))
            self.ece.reset()
