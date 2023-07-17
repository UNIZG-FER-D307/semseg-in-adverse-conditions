from torchvision.datasets import Cityscapes as _Cityscapes
from torch.utils.data import Dataset
import numpy as np
from .cityscapes_labels import create_id_to_train_id_mapper

cityscapes_mean = np.array([0.2869, 0.3251, 0.2839])
cityscapes_std = np.array([0.1761, 0.1810, 0.1777])

label_mapper_ = create_id_to_train_id_mapper()

class CityscapesWithOOD(Dataset):
    def __init__(self, dataroot, split, image_transform=None, target_transform=None, joint_transform=None, ood_transforms=None):
        super(CityscapesWithOOD, self).__init__()
        self._data = _Cityscapes(dataroot, split=split, mode='fine', target_type='semantic',
                       transform=image_transform, target_transform=target_transform)
        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.ood_transforms = ood_transforms
        self.tag = 'day'

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        img, lbl = self._data[index]
        ood_x, shape = self.ood_transforms((img, lbl))
        if self.joint_transform:
            img, lbl = self.joint_transform((img, lbl))
        return img, lbl, ood_x, shape

    def create_img_lbl_list(self):
        return [
            (img_path, lbl_path[0])
            for img_path, lbl_path in zip(self._data.images, self._data.targets)]

    def create_label_mapper(self):
        return label_mapper_

    def create_cache_name(self):
        return f"{type(self).__name__}_{self._data.split}_{self.tag}"
