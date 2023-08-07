import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import os
from data import colorize_labels
from torchvision.utils import save_image
import numpy as np
from PIL import Image


class PseudolabelGenerator(pl.LightningModule):
    def __init__(
        self,
        model,
        ignore_id,
        num_classes,
        multiscale=True,
        save_output=True,
        output_dir="./generation_outputs",
        min_conf=0.99,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.scales = [1.0, 1.25, 1.5, 1.75, 2]
        # self.scales = [0.75, 1.0, 1.5, 2]
        self.multiscale = multiscale
        self.save_output = save_output
        self.output_dir = output_dir
        self.minimal_confidence = min_conf

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
            logit = self._unpad_output(logit, dh, dw)
            logit = F.interpolate(
                logit, size=(H, W), mode="bilinear", align_corners=False
            )
            # logit = logit * self.stats[i].view(1, 19, 1, 1).to(logit)
            final_logits += logit
        return final_logits / len(self.scales)

    def _save_output(self, y, conf, name):
        name = name.split("/")[-1]
        # name = '/'.join(name.split('/')[-4:]) # cadcd
        name = name.replace(".png", "_pseudolabel.png").replace(
            ".jpg", "_pseudolabel.png"
        )
        classes = os.path.join(self.output_dir, "pseudolabelTrainIds", name)
        colors = os.path.join(self.output_dir, "colorized", name)
        os.makedirs(os.path.dirname(classes), exist_ok=True)
        os.makedirs(os.path.dirname(colors), exist_ok=True)

        conf = conf[0]
        y = y[0]
        y[conf < self.minimal_confidence] = 255  # v1

        ids = y.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(ids)
        img.save(classes)
        y[y == 255] = 19
        colorized_y = colorize_labels(y.cpu())
        save_image(colorized_y, colors)

    def test_step(self, batch, batch_idx):
        x, name = batch
        name = name[0]
        with torch.no_grad():
            if self.multiscale:
                logit = self._multiscale_test_step(x)
            else:
                init_H, init_W = x.shape[-2:]
                x = self._pad_input(x)
                logit = self.forward(x)
                logit = self._unpad_output(logit, init_H, init_W)

        y = logit.max(1)[1]
        conf = torch.softmax(logit, dim=1).max(1)[0]
        if self.save_output:
            self._save_output(y, conf, name)
