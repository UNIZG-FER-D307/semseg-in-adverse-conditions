import torch
from torch import nn as nn
from torch.autograd import Function
import torch.nn.functional as F


class BoundaryAwareFocalLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        ignore_id,
        superclass_mask=None,
        gamma=0.5,
        efficient=False,
        logits_input=True,
        check_nans=False,
    ):
        super(BoundaryAwareFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        if superclass_mask is None:
            superclass_mask = torch.zeros(
                (max(num_classes, ignore_id) + 1, num_classes), dtype=torch.float32
            )
            superclass_mask[:num_classes] = torch.eye(num_classes, dtype=torch.float32)
        self.register_buffer("superclass_mask", superclass_mask)
        self.register_buffer("gamma", torch.tensor(gamma))
        self.efficient = efficient
        self.logits_input = logits_input
        self.check_nans = check_nans

    def forward(self, input, labels, label_distance_alphas):
        with torch.no_grad():
            target = (
                self.superclass_mask[labels.view(-1)]
                .view(*labels.shape, -1)
                .permute(0, 3, 1, 2)
            )
            valid_mask = torch.logical_and(
                label_distance_alphas > 0, target.sum(1) != 0
            )
            if valid_mask.sum().le(0):
                return torch.zeros(
                    size=(0,), device=input.device, requires_grad=True
                ).sum()

        if not self.efficient:
            if self.logits_input:
                input = input.softmax(1)
            logpt = input.mul(target).sum(1)[valid_mask].log_()
            with torch.no_grad():
                w = -label_distance_alphas[valid_mask].mul_(
                    torch.exp(self.gamma * (1 - logpt.exp()))
                )  # TODO
            loss = w * logpt
            if self.check_nans:
                loss = torch.nan_to_num(loss, posinf=0, neginf=0, nan=0)
            loss = loss.mean()
        else:
            # loss = OptBoundaryAwareLoss.apply(input, target, label_distance_alphas, valid_mask, self.gamma)
            raise Exception("Unimplemented")
        return loss


class OptBoundaryAwareLoss(Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        with torch.no_grad():
            input, target, label_distance_alphas, valid_mask, gamma = args

            logpt = input.softmax(1).mul_(target).sum(1)[valid_mask].log_()

            label_distance_alphas[valid_mask] *= torch.exp(
                gamma * (1 - logpt.exp())
            ).mul_(-1)
            label_distance_alphas[valid_mask.logical_not()] = 0

            ctx.save_for_backward(input, target, label_distance_alphas, valid_mask)

            return (label_distance_alphas[valid_mask] * logpt).mean()

    @staticmethod
    def backward(ctx, *grad_outputs):
        with torch.no_grad():
            input, target, alphas, valid_mask = ctx.saved_tensors
            input = input - input.max(1, keepdims=True)[0]

            exps = input.exp()
            sumexp = exps.sum(1, keepdim=True)
            grad_input = exps.div(sumexp).mul_(-1)
            exps = exps.mul_(target)
            sumexpv = exps.sum(1, keepdim=True)
            sumexpv[sumexpv == 0] = 1e-9
            grad_pos = exps.div_(sumexpv)
            N = valid_mask.sum()

            for x in [grad_input, grad_pos]:
                if (isinf := torch.isinf(x).any()) or (isnan := torch.isnan(x).any()):
                    grad_input.fill_(0.0)
                    print("Setting gradient to zero NaN={isnan}, Inf={isinf}")
                    return grad_outputs[0] * grad_input
            return (
                grad_outputs[0]
                * grad_input.add_(grad_pos).mul_(alphas.unsqueeze(1)).div_(N),
                None,
                None,
                None,
                None,
            )


class BoundaryAwareCrossEntropy(nn.Module):
    def __init__(self, num_classes=19, ignore_id=19):
        super(BoundaryAwareCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id

    def forward(self, logits, labels, label_distance_alphas):
        loss = torch.mean(
            F.cross_entropy(
                logits, labels, ignore_index=self.ignore_id, reduction="none"
            )
            * label_distance_alphas
        )
        return loss


class BoundaryAwareFocalLossWithLabelDist(nn.Module):
    def __init__(
        self, num_classes, ignore_id, superclass_mask=None, gamma=0.5, logits_input=True
    ):
        super(BoundaryAwareFocalLossWithLabelDist, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        # Not needed since already implemented in given labels
        # if superclass_mask is None:
        #     superclass_mask = torch.zeros((max(num_classes, ignore_id) + 1, num_classes), dtype=torch.float32)
        #     superclass_mask[:num_classes] = torch.eye(num_classes, dtype=torch.float32)
        # self.register_buffer('superclass_mask', superclass_mask)
        self.register_buffer("gamma", torch.tensor(gamma))
        self.logits_input = logits_input

    def forward(self, input, labels, label_distance_alphas):
        with torch.no_grad():
            # target = self.superclass_mask[labels.view(-1)].view(*labels.shape, -1).permute(0, 3, 1, 2)
            target = labels
            valid_mask = torch.logical_and(
                label_distance_alphas > 0, target.sum(1) != 0
            )
            if valid_mask.sum().le(0):
                return torch.zeros(
                    size=(0,), device=input.device, requires_grad=True
                ).sum()

        if self.logits_input:
            input = input.softmax(1)
        logpt = input.mul(target).sum(1)[valid_mask].log_()
        with torch.no_grad():
            w = -label_distance_alphas[valid_mask].mul_(
                torch.exp(self.gamma * (1 - logpt.exp()))
            )  # TODO
        loss = (w * logpt).mean()
        return loss
