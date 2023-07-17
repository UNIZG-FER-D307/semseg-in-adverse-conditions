import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms import GaussianBlur
import numpy as np

class CreateRandomPatch(nn.Module):

    def __init__(self, class_count:int=3, kernel_h:int=16, kernel_w:int=16, stride:int=4, size_min:int=16, size_max:int=256, blur:GaussianBlur=GaussianBlur(7, 4.), shuffle_patches=False):
        super(CreateRandomPatch, self).__init__()

        self.class_count = class_count
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride
        self.size_min = size_min
        self.size_max = size_max
        self.blur = blur
        self.shuffle_patches = shuffle_patches

    def _sample_shape(self):
        sizes = [i for i in range(16, self.size_max, 16)]
        w = np.random.choice(sizes)
        h = np.random.choice(sizes)
        return (h, w)

    def _pad(self, x):
        H, W = x.shape[-2:]
        pad_H = (self.size_max - H) // 2 if H != self.size_max else 0
        pad_W = (self.size_max - W) // 2 if W != self.size_max else 0
        x = torch.nn.functional.pad(x, (pad_W, pad_W, pad_H, pad_H), 'constant', 0.)
        return x

    def forward(self, data):
        im, lbl = data
        im = im.unsqueeze(0)
        lbl = lbl.unsqueeze(0)
        ood_h, ood_w = self._sample_shape()
        assert ood_h >= self.size_min and ood_h <= self.size_max
        assert ood_w >= self.size_min and ood_w <= self.size_max

        patches = F.unfold(input=im, kernel_size=(self.kernel_h, self.kernel_w), stride=self.stride)
        patches_lbl = F.unfold(input=lbl.float(), kernel_size=(self.kernel_h, self.kernel_w), stride=self.stride)
        lbls = torch.mode(patches_lbl, 1)[0].unsqueeze(1)

        cls = lbls.unique()[:-1]
        sel_cls = cls[torch.randperm(cls.shape[0])[:self.class_count]]

        # ood_h, ood_w = 256, 256
        numel = F.unfold(input=torch.randn(1, 3, ood_h, ood_w), kernel_size=(self.kernel_h, self.kernel_w), stride=self.stride).shape[-1]
        per_class_samples = numel // self.class_count
        acc = []
        for c in sel_cls:
            cp_ = []
            class_patches = patches[:, :, lbls.squeeze() == c]
            cp = class_patches[:, :, torch.randperm(class_patches.shape[-1])][:, :, :per_class_samples]
            cp_.append(cp)
            cp = torch.cat(cp_, dim=-1)
            while cp.shape[-1] < per_class_samples:
                cp = torch.cat((class_patches[:, :, torch.randperm(class_patches.shape[-1])], cp), dim=-1)
            cp = cp[:, :, :per_class_samples]
            acc.append(cp)
        out = torch.cat(acc, dim=-1)
        if out.shape[-1] != numel:
            out = torch.cat((patches[:, :, torch.randperm(patches.shape[-1])][:, :, :(numel - out.shape[-1])], out), dim=-1)

        if self.shuffle_patches:
            out = out[:, :, torch.randperm(out.shape[-1])]

        ood = F.fold(input=out, kernel_size=(self.kernel_h, self.kernel_w), stride=self.stride, output_size=(ood_h, ood_w))
        ones_ood = torch.ones_like(out)
        ones_ood = F.fold(input=ones_ood, kernel_size=(self.kernel_h, self.kernel_w), stride=self.stride, output_size=(ood_h, ood_w))
        final = ood / ones_ood
        final = self.blur(final)
        return self._pad(final), torch.LongTensor([ood_h, ood_w])
