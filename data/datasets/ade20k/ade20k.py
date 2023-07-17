from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat
import numpy as np
import os
import glob
from PIL import Image
import torch
from torchvision.transforms import ToTensor

ade20k_mean = np.array([0.4934, 0.4681, 0.4309])
ade20k_std = np.array([0.2285, 0.2294, 0.2404])

def create_ade_id_to_train_id():
    mapper = torch.arange(151) - 1
    mapper[0] = 150
    return mapper

class ADE20k(Dataset):
    def __init__(self, root, split='training', image_transform=None, target_transform=None, joint_transform=None):
        self.root = root
        self.split = split
        assert split in ['training', 'validation', 'testing']
        self.images_dir = os.path.join(root, 'images', split)
        self.labels_dir = os.path.join(root, 'annotations', split)
        self.images = list(sorted(glob.glob(self.images_dir + '/*.jpg')))
        self.labels = list(sorted(glob.glob(self.labels_dir + '/*.png')))

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index].rstrip()
        lbl_path = self.labels[index].rstrip()

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        image = Image.open(img_path).convert('RGB')

        # print(index, img_path, image.size)
        # if ToTensor()(image).shape[0] == 1:
        #     print(img_path)
        #     image = ToTensor()(image)
        #     image = image.repeat(3, 1, 1)
        #     image = Image.fromarray(image.numpy())


        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))
        label = Image.open(lbl_path)

        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        out = self.joint_transform((image, label)) if self.joint_transform else (image, label)
        return out

    def create_img_lbl_list(self):
        return [
            (img_path, lbl_path)
            for img_path, lbl_path in zip(self.images, self.labels)]

    def create_label_mapper(self):
        return None

    def create_cache_name(self):
        return f"{type(self).__name__}_{self.split}"
