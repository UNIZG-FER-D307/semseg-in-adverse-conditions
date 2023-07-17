import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import copy
from einops.layers.torch import Rearrange
from einops import rearrange
from functools import partial

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class _ConvLNConvReLUConv(nn.Sequential):
    def __init__(self, num_maps_in, *args, **kwargs):
        super(_ConvLNConvReLUConv, self).__init__()
        dim = num_maps_in
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.add_module('dw_conv', nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True))
        self.add_module('perm_a', Permute([0, 2, 3, 1]))
        self.add_module('norm_a', norm_layer(dim))
        self.add_module('linear_a', nn.Linear(in_features=dim, out_features=4 * dim, bias=True))
        self.add_module('relu_act', nn.ReLU())
        self.add_module('linear_b', nn.Linear(in_features=4 * dim, out_features=dim, bias=True))
        self.add_module('perm_b', Permute([0, 3, 1, 2]))

class _LNReLUConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, use_ln=False, bias=True, dilatation=1):
        super(_LNReLUConv, self).__init__()
        if use_ln:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            self.add_module('perm_a', Permute([0, 2, 3, 1]))
            self.add_module('norm_a', norm_layer(num_maps_in))
            self.add_module('perm_b', Permute([0, 3, 1, 2]))

        self.add_module('relu_act', nn.ReLU())
        self.add_module('proj', nn.Conv2d(in_channels=num_maps_in, out_channels=num_maps_out, kernel_size=k, bias=bias, dilation=dilatation))

def _create_upsamping_ln_relu_conv(input_dims, upsample_dims):
    convs = []
    for in_c, out_c in zip(input_dims, upsample_dims):
        convs.append(_LNReLUConv(in_c, out_c, k=1, bias=True, use_ln=False))
    return nn.ModuleList(convs)

class Pyramid(nn.Module):
    def __init__(self, network, upsample_dims):
        super(Pyramid, self).__init__()
        self.network = network

        chnls = self._out_chnls()
        self.up1 = _create_upsamping_ln_relu_conv(chnls, upsample_dims[:4])
        self.up2 = _create_upsamping_ln_relu_conv(chnls, upsample_dims[1:5])
        self.up3 = _create_upsamping_ln_relu_conv(chnls, upsample_dims[2:])

    def _out_chnls(self):
        with torch.no_grad():
            out = self.network(torch.zeros(1, 3, 64, 64))
        return [o.shape[1] for o in out]

    def forward(self, x):
        outs_s1 = self.network(x)

        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        outs_s2 = self.network(x)

        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        outs_s3 = self.network(x)

        p1_up = [t(o) for o, t in zip(outs_s1, self.up1)]
        p2_up = [t(o) for o, t in zip(outs_s2, self.up2)]
        p3_up = [t(o) for o, t in zip(outs_s3, self.up3)]

        outputs = [
            p1_up[0],
            p1_up[1] + p2_up[0],
            p1_up[2] + p2_up[1] + p3_up[0],
            p1_up[3] + p2_up[2] + p3_up[1],
            p2_up[3] + p3_up[2],
            p3_up[3]
        ]
        return outputs



class _Upsample(nn.Module):
    def __init__(self, num_maps_in, num_maps_out, num_classes, use_aux=False):
        super(_Upsample, self).__init__()
        print('Upsample layer: in =', num_maps_in, ' out =', num_maps_out)
        assert num_maps_in == num_maps_out
        self.blend_conv = _ConvLNConvReLUConv(num_maps_in)
        self.use_aux = use_aux
        if use_aux:
            self.logits_aux = _LNReLUConv(num_maps_in, num_classes, k=1, bias=True, use_ln=False)


    def forward(self, x_, skip):
        skip_size = skip.size()[2:4]

        x = F.interpolate(x_, skip_size, mode='bilinear', align_corners=False)
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
            upsample = _Upsample(num_features, up_sizes[i], num_classes, use_aux=use_aux)
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


class DeterministicPyramidSemSeg(nn.Module):
    def __init__(self, backbone, num_classes=19, upsample_dims=128, use_aux=False):
        super(DeterministicPyramidSemSeg, self).__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux
        up_dims = [upsample_dims] * 6
        self.backbone = Pyramid(backbone, upsample_dims=up_dims)
        self.upsample = Upsample(upsample_dims, up_dims, num_classes, use_aux)
        self.logits = _LNReLUConv(upsample_dims, self.num_classes, k=1, bias=True, use_ln=False)

    def forward(self, x):
        H, W = x.shape[-2:]
        outs = self.backbone(x)
        pre_logit, aux = self.upsample(outs[-1], outs[:-1])
        logit = self.logits(pre_logit)
        out = F.interpolate(logit, (H, W), mode='bilinear', align_corners=False)
        if self.use_aux:
            return out, aux
        return out

    def prepare_optim_params(self):
        return list(self.backbone.parameters()), list(self.upsample.parameters()) + list(self.logits.parameters())

def parameter_count(module):
    trainable, non_trainable = 0, 0
    for p in module.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            non_trainable += p.numel()
    return trainable, non_trainable
