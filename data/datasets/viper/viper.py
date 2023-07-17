import os
from torch.utils.data import Dataset
import numpy as np
import glob
from PIL import Image
from .labels import create_id_to_train_id_mapper

viper_mean = np.array([0.3485, 0.3422, 0.3395])
viper_std = np.array([0.2002, 0.1934, 0.1919])

mapper_viper_city = create_id_to_train_id_mapper()

class Viper(Dataset):

    def __init__(self, root, split='train', image_transform=None, target_transform=None, joint_transform=None):
        assert split in ['train', 'val']
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.root = root
        self.images_root = os.path.join(root, split, 'img')
        self.labels_root = os.path.join(root, split, 'cls')

        self.images = list(sorted(glob.glob(self.images_root + '/*/*.jpg')))
        self.labels = list(sorted(glob.glob(self.images_root + '/*/*.png')))

        print("> Found %d %s images..." % (len(self.images), split))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index].rstrip()
        lbl_path = self.labels[index].rstrip()

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        image = Image.open(img_path)

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))
        label = Image.open(lbl_path)

        image = self.image_transform(image)
        label = self.target_transform(label)

        return self.joint_transform((image, label)) if self.joint_transform else (image, label)

    def create_img_lbl_list(self):
        return [
            (img_path, lbl_path)
            for img_path, lbl_path in zip(self.images, self.labels)]

    def create_label_mapper(self):
        return mapper_viper_city

    def create_cache_name(self):
        return f"{type(self).__name__}_{self.split}"
