import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import copy
from einops.layers.torch import Rearrange
from einops import rearrange


class _BNReluConv(nn.Sequential):
    def __init__(
        self,
        num_maps_in,
        num_maps_out,
        k=3,
        batch_norm=True,
        bias=True,
        dilation=1,
        groups=1,
    ):
        super(_BNReluConv, self).__init__()
        batchnorm_momentum = 0.01
        if batch_norm:
            self.add_module(
                "norm", nn.BatchNorm2d(num_maps_in, momentum=batchnorm_momentum)
            )
        self.add_module("relu", nn.ReLU(inplace=True))
        padding = k // 2
        self.add_module(
            "conv",
            nn.Conv2d(
                num_maps_in,
                num_maps_out,
                kernel_size=k,
                padding=padding,
                bias=bias,
                dilation=dilation,
                groups=groups,
            ),
        )


def _create_upsamping_bn_relu_conv(input_dims, upsample_dims):
    convs = []
    for in_c, out_c in zip(input_dims, upsample_dims):
        convs.append(_BNReluConv(in_c, out_c, k=1, bias=True))
    return nn.ModuleList(convs)


class SingleScale(nn.Module):
    def __init__(self, network, upsample_dims):
        super(SingleScale, self).__init__()
        self.network = network

        self.chnls = self._out_chnls()
        self.up = _create_upsamping_bn_relu_conv(self.chnls[:-1], upsample_dims)

    def _out_chnls(self):
        with torch.no_grad():
            out = self.network(torch.zeros(1, 3, 64, 64))
        return [o.shape[1] for o in out]

    def forward(self, x):
        outs_s = self.network(x)
        p1_up = [t(o) for o, t in zip(outs_s[:-1], self.up)]
        return p1_up + [outs_s[-1]]


class SingleScaleOpenClip(nn.Module):
    def __init__(self, network, upsample_dims):
        super(SingleScaleOpenClip, self).__init__()
        self.network = network

        self.chnls = self._out_chnls()
        self.up = _create_upsamping_bn_relu_conv(self.chnls[:-1], upsample_dims)

    def _out_chnls(self):
        with torch.no_grad():
            out, lf = self.network(torch.zeros(1, 3, 64, 64))
            out[-1] = lf
        return [o.shape[1] for o in out]

    def forward(self, x):
        outs_s, lf = self.network(x)
        outs_s[-1] = lf
        p1_up = [t(o) for o, t in zip(outs_s[:-1], self.up)]
        return p1_up + [outs_s[-1]]


class _Upsample(nn.Module):
    def __init__(self, num_maps_in, num_maps_out, num_classes, use_aux=False):
        super(_Upsample, self).__init__()
        # print('Upsample layer: in =', num_maps_in, ' out =', num_maps_out)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=3, groups=1)
        self.use_aux = use_aux
        if use_aux:
            self.logits_aux = _BNReluConv(num_maps_in, num_classes, k=1, bias=True)

    def forward(self, x_, skip):
        skip_size = skip.size()[2:4]

        x = F.interpolate(x_, skip_size, mode="bilinear", align_corners=False)
        x = x + skip
        x = self.blend_conv(x)
        if self.use_aux:
            aux_out = self.logits_aux(x_)
            return x, aux_out
        else:
            return x, None


class Upsample(nn.Module):
    def __init__(self, num_features, up_sizes, num_classes, use_aux):
        super(Upsample, self).__init__()
        upsample_layers = []
        for i in range(len(up_sizes)):
            upsample = _Upsample(
                num_features, up_sizes[i], num_classes, use_aux=use_aux
            )
            num_features = up_sizes[i]
            upsample_layers.append(upsample)
        self.upsample_layers = nn.ModuleList(upsample_layers)

    def forward(self, x, skip_layers):
        aux = []
        for i, skip in enumerate(reversed(skip_layers)):
            out = self.upsample_layers[i].forward(x, skip)
            x = out[0]
            aux.append(out[1])
        return x, aux


class SpatialPyramidPooling(nn.Module):
    def __init__(
        self,
        num_maps_in,
        num_levels,
        bt_size=512,
        level_size=128,
        out_size=256,
        grids=[6, 3, 2, 1],
        square_grid=False,
    ):
        super(SpatialPyramidPooling, self).__init__()
        self.grids = grids
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module("spp_bn", _BNReluConv(num_maps_in, bt_size, k=1))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module(
                "spp" + str(i), _BNReluConv(num_features, level_size, k=1)
            )
        self.spp.add_module("spp_fuse", _BNReluConv(final_size, out_size, k=1))

    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = F.upsample(level, target_size, mode="bilinear")
            levels.append(level)
        x = torch.cat(levels, 1)
        return self.spp[-1].forward(x)


class SPPWrapper(nn.Module):
    def __init__(self, num_features, spp_size=512):
        super(SPPWrapper, self).__init__()

        self.spp_size = spp_size
        spp_square_grid = False
        spp_grids = [8, 4, 2, 1]
        num_levels = 4
        level_size = self.spp_size // num_levels
        bt_size = self.spp_size
        self.spp = SpatialPyramidPooling(
            num_features,
            num_levels,
            bt_size,
            level_size,
            self.spp_size,
            spp_grids,
            spp_square_grid,
        )
        self.num_features = self.spp_size

    def forward(self, x):
        return self.spp(x)


class SingleScaleSemSeg(nn.Module):
    def __init__(self, backbone, num_classes=19, upsample_dims=128, use_aux=False):
        super(SingleScaleSemSeg, self).__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux
        up_dims = [upsample_dims] * 3
        self.backbone = SingleScale(backbone, upsample_dims=up_dims)
        self.spp = SPPWrapper(self.backbone.chnls[-1], upsample_dims)
        self.upsample = Upsample(upsample_dims, up_dims, num_classes, use_aux)
        self.logits = _BNReluConv(upsample_dims, self.num_classes, k=1, bias=True)

    def forward(self, x):
        H, W = x.shape[-2:]
        outs = self.backbone(x)
        out = self.spp(outs[-1])
        pre_logit, aux = self.upsample(out, outs[:-1])
        logit = self.logits(pre_logit)
        out = F.interpolate(logit, (H, W), mode="bilinear", align_corners=False)
        if self.use_aux:
            return out, aux
        return out

    def prepare_optim_params(self):
        return list(self.backbone.parameters()), list(
            self.upsample.parameters()
        ) + list(self.logits.parameters()) + list(self.spp.parameters())


class SingleScaleSemSegOpenClip(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes=19,
        upsample_dims=128,
        use_aux=False,
        embeddings=None,
        temperature=None,
    ):
        super(SingleScaleSemSegOpenClip, self).__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux
        up_dims = [upsample_dims] * 3
        self.backbone = SingleScaleOpenClip(backbone, upsample_dims=up_dims)
        self.spp = SPPWrapper(self.backbone.chnls[-1], upsample_dims)
        self.upsample = Upsample(upsample_dims, up_dims, num_classes, use_aux)
        self.embeddings = embeddings

        if self.embeddings is None:
            self.logits = _BNReluConv(upsample_dims, self.num_classes, k=1, bias=True)
        else:
            assert (
                embeddings.shape[0] == num_classes
            ), f"Number of text embeddings: {embeddings.shape[0]} must be equal num classes: {num_classes}"
            assert temperature, "Temperature not set"
            self.temperature = temperature

    def forward(self, x):
        H, W = x.shape[-2:]
        outs = self.backbone(x)
        out = self.spp(outs[-1])
        pre_logit, aux = self.upsample(out, outs[:-1])
        if self.embeddings is None:
            logit = self.logits(pre_logit)
        else:
            prelogit_norm = pre_logit / pre_logit.norm(dim=1, keepdim=True)
            logit = F.conv2d(
                prelogit_norm, self.embeddings[:, :, None, None].to(prelogit_norm)
            )

        out = F.interpolate(logit, (H, W), mode="bilinear", align_corners=False)
        if self.embeddings is not None:
            out = out / self.temperature

        if self.use_aux:
            return out, aux
        return out

    def prepare_optim_params(self):
        backbone_params = list(self.backbone.parameters())
        upsample_params = list(self.upsample.parameters())
        spp_params = list(self.spp.parameters())
        if self.embeddings is None:
            return backbone_params, upsample_params + spp_params + list(
                self.logits.parameters()
            )
        else:
            return backbone_params, upsample_params + spp_params


def parameter_count(module):
    trainable, non_trainable = 0, 0
    for p in module.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            non_trainable += p.numel()
    return trainable, non_trainable
