import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import copy
from einops.layers.torch import Rearrange
from einops import rearrange

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bias=True, dilation=1, groups=1):
        super(_BNReluConv, self).__init__()
        batchnorm_momentum = 0.01
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=batchnorm_momentum))
        self.add_module('relu', nn.ReLU(inplace=True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation, groups=groups))

def _create_upsamping_bn_relu_conv(input_dims, upsample_dims):
    convs = []
    for in_c, out_c in zip(input_dims, upsample_dims):
        convs.append(_BNReluConv(in_c, out_c, k=1, bias=True))
    return nn.ModuleList(convs)

class Pyramid(nn.Module):
    def __init__(self, network, upsample_dims):
        super(Pyramid, self).__init__()
        self.network1 = copy.deepcopy(network)
        self.network2 = copy.deepcopy(network)
        self.network3 = copy.deepcopy(network)

        chnls = self._out_chnls()
        self.up1 = _create_upsamping_bn_relu_conv(chnls, upsample_dims[:4])
        self.up2 = _create_upsamping_bn_relu_conv(chnls, upsample_dims[1:5])
        self.up3 = _create_upsamping_bn_relu_conv(chnls, upsample_dims[2:])


    def _extract_bn(self, network):
        bn_modules = nn.ModuleDict()
        for k, v in network.named_modules():
            if 'bn' in k:
                bn_modules[k.replace('.', '/')] = v
        return bn_modules

    def _switch_bn(self, bn_dict):
        for k, v in bn_dict.items():
            names = k.split('/')
            n = self.network
            for name in names[:-1]:
                n = n.__getattr__(name)
            n.__setattr__(names[-1], v)

    def _out_chnls(self):
        with torch.no_grad():
            out = self.network1(torch.zeros(1, 3, 64, 64))
        return [o.shape[1] for o in out]

    def forward(self, x):
        outs_s1 = self.network1(x)

        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        outs_s2 = self.network2(x)

        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        outs_s3 = self.network3(x)

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
        # print('Upsample layer: in =', num_maps_in, ' out =', num_maps_out)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=3, groups=1)
        self.use_aux = use_aux
        if use_aux:
            self.logits_aux = _BNReluConv(num_maps_in, num_classes, k=1, bias=True)


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


class PyramidSemSegNonShared(nn.Module):
    def __init__(self, backbone, num_classes=19, upsample_dims=128, use_aux=False):
        super(PyramidSemSegNonShared, self).__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux
        up_dims = [upsample_dims] * 6
        self.backbone = Pyramid(backbone, upsample_dims=up_dims)
        self.upsample = Upsample(upsample_dims, up_dims, num_classes, use_aux)
        self.logits = _BNReluConv(upsample_dims, self.num_classes, k=1, bias=True)

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

class PyramidSemSegSelfSup(nn.Module):
    def __init__(self, backbone, num_classes=19, upsample_dims=128):
        super(PyramidSemSegSelfSup, self).__init__()
        self.num_classes = num_classes
        up_dims = [upsample_dims] * 6
        self.backbone = Pyramid(backbone, upsample_dims=up_dims)
        self.upsample = Upsample(upsample_dims, up_dims)
        self.logits = _BNReluConv(upsample_dims, self.num_classes, k=1, bias=True)
        # self.selfsup_predictor = nn.Sequential(
        #     nn.Conv2d(upsample_dims, 384, 1),
        #     _BNReluConv(384, upsample_dims, k=1, bias=True))

    def forward(self, x):
        H, W = x.shape[-2:]
        outs = self.backbone(x)
        pre_logit = self.upsample(outs[-1], outs[:-1])
        logit = self.logits(pre_logit)
        out = F.interpolate(logit, (H, W), mode='bilinear', align_corners=False)
        return out

    def seflsup_forward(self, x, use_predictor=False):
        outs = self.backbone(x)
        out = self.upsample(outs[-1], outs[:-1])
        # if use_predictor:
        #     out = self.selfsup_predictor(out)
        return out

    def prepare_optim_params(self):
        return list(self.backbone.parameters()), list(self.upsample.parameters()) + list(self.logits.parameters()) # + list(self.selfsup_predictor.parameters())

def parameter_count(module):
    trainable, non_trainable = 0, 0
    for p in module.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            non_trainable += p.numel()
    return trainable, non_trainable


class NoisyPyramidSemSeg(nn.Module):
    def __init__(self, backbone, num_classes=19, upsample_dims=128, use_aux=False):
        super(NoisyPyramidSemSeg, self).__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux
        up_dims = [upsample_dims] * 6
        self.backbone = Pyramid(backbone, upsample_dims=up_dims)
        self.upsample = Upsample(upsample_dims, up_dims, num_classes, use_aux)
        self.logits = _BNReluConv(upsample_dims, self.num_classes, k=1, bias=True)
        self.t_matrix = nn.Sequential(
            _BNReluConv(upsample_dims, upsample_dims, k=3, bias=True),
            _BNReluConv(upsample_dims, self.num_classes ** 2, k=1, bias=True)
        )

    def forward(self, x, return_T=False):
        H, W = x.shape[-2:]
        outs = self.backbone(x)
        pre_logit, aux = self.upsample(outs[-1], outs[:-1])
        logit = self.logits(pre_logit)
        out = F.interpolate(logit, (H, W), mode='bilinear', align_corners=False)
        if self.use_aux:
            return out, aux
        if return_T:
            T = self.t_matrix(pre_logit)
            T = F.interpolate(T, (H, W), mode='bilinear', align_corners=False)
            T = rearrange(T, 'n (c1 c2) h w -> n c1 c2 h w', c1=self.num_classes, c2=self.num_classes)
            T = T.softmax(2)
            return out, T

        return out

    def prepare_optim_params(self):
        return list(self.backbone.parameters()), \
               list(self.upsample.parameters()) + list(self.logits.parameters()) + list(self.t_matrix.parameters())

class PyramidSemSegTH(nn.Module):
    def __init__(self, backbone, num_classes=19, upsample_dims=128, use_aux=False):
        super(PyramidSemSegTH, self).__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux
        up_dims = [upsample_dims] * 6
        self.backbone = Pyramid(backbone, upsample_dims=up_dims)
        self.upsample = Upsample(upsample_dims, up_dims, num_classes, use_aux)
        self.logits = _BNReluConv(upsample_dims, self.num_classes, k=1, bias=True)
        self.ood_logits = _BNReluConv(upsample_dims, 2, k=1, bias=True)


    def forward(self, x):
        H, W = x.shape[-2:]
        outs = self.backbone(x)
        pre_logit, aux = self.upsample(outs[-1], outs[:-1])
        logit = self.logits(pre_logit)
        out = F.interpolate(logit, (H, W), mode='bilinear', align_corners=False)

        ood_logits = self.ood_logits(pre_logit)
        ood_out = F.interpolate(ood_logits, (H, W), mode='bilinear', align_corners=False)

        if self.use_aux:
            return out, ood_out, aux
        return out, ood_out

    def prepare_optim_params(self):
        return list(self.backbone.parameters()), list(self.upsample.parameters()) + list(self.logits.parameters())