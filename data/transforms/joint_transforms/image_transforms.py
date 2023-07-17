import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional
import torchvision.transforms.functional as F
from enum import Enum
import torch
from torch import Tensor
from PIL import Image

get_image_size = F.get_image_size if hasattr(F, "get_image_size") else F._get_image_size


class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]


pil_modes_mapping = {
    InterpolationMode.NEAREST: 0,
    InterpolationMode.BILINEAR: 2,
    InterpolationMode.BICUBIC: 3,
    InterpolationMode.BOX: 4,
    InterpolationMode.HAMMING: 5,
    InterpolationMode.LANCZOS: 1,
}


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class ImageJitterRandomCrop(torch.nn.Module):
    def __init__(self, size, scale=(0.5, 1.5), ignore_id=19, input_mean=0):
        super().__init__()
        self.size = _setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        )

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if scale[0] > scale[1]:
            warnings.warn("Scale and ratio should be of kind (min, max)")
        self.scale = scale
        self.ignore_id = ignore_id
        self.input_mean = input_mean

    @staticmethod
    def get_resize_params(img: Tensor, scale: List[float]) -> int:
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = get_image_size(img)
        min_value = min(width, height)
        coef = torch.rand(1) * (scale[0] - scale[1]) + scale[1]
        val = int(coef * min_value)
        return val, coef

    @staticmethod
    def get_crop_params(
        img: Tensor, output_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format(
                    (th, tw), (h, w)
                )
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()

        return i, j, th, tw

    def forward(self, data):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        img = data

        resize_val, coef_ = self.get_resize_params(img, self.scale)

        img = F.resize(img, resize_val, pil_modes_mapping[InterpolationMode.BILINEAR])

        width, height = get_image_size(img)

        mask = torch.ones(1, height, width)

        # pad the width if needed
        if width < self.size[1]:
            padding = [self.size[1] - width, 0]
            # img = F.to_pil_image(img)
            img = F.pad(img, padding, self.input_mean, "constant")
            mask = F.pad(mask, padding, self.input_mean, "constant")

        # pad the height if needed
        if height < self.size[0]:
            padding = [0, self.size[0] - height]
            # img = F.to_pil_image(img)
            img = F.pad(img, padding, self.input_mean, "constant")
            mask = F.pad(mask, padding, self.input_mean, "constant")
            # img = F.to_tensor(img)
        i, j, h, w = self.get_crop_params(img, self.size)

        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return img, mask


class DarkenImage(torch.nn.Module):
    def __init__(self, factor_range=[0.25 / 2, 0.25]):
        super().__init__()
        assert factor_range[0] < factor_range[1]
        assert len(factor_range) == 2
        self.factor_range = factor_range

    def forward(self, data):
        img = data
        assert type(img) == torch.Tensor
        assert img.min().item() >= 0.0 and img.max().item() <= 1

        min_v, max_v = self.factor_range
        coef = (torch.rand(1).item() - min_v) / (max_v - min_v)

        return img * coef


class ColorJitter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data):
        img = data
        assert type(img) == torch.Tensor
        assert img.min().item() >= 0.0 and img.max().item() <= 1
