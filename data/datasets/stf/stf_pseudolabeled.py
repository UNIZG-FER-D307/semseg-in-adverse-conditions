import numpy as np
import os
from PIL import Image
from torch.utils import data
import glob

class STFPseudolabeled(data.Dataset):

    def __init__(self, root, image_transform=None, target_transform=None, joint_transform=None):
        self.root = root
        self.tag = 'fog'
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images_base = os.path.join(self.root)
        self.files = sorted(glob.glob(self.images_base + '/images/*.png'))
        # self.files = [self.files[i] for i in range(0, len(self.files), 4)]
        self.label_base = os.path.join(self.root, 'pseudolabelTrainIds_v2')

        if not self.files:
            raise Exception("> No files found in %s" % (self.images_base))

        print("> Found %d images..." % (len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index].rstrip()
        name = img_path.split('/')[-1]
        lbl_path = os.path.join(self.label_base, name.replace('.png', '_pseudolabel.png'))

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        image = Image.open(img_path)

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))
        label = Image.open(lbl_path)

        if self.image_transform:
            image = self.image_transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return self.joint_transform((image, label)) if self.joint_transform else (image, label)

    def create_img_lbl_list(self):
        return [
            (img_path, os.path.join(self.label_base, img_path.split('/')[-1].replace('.png', '_pseudolabel.png')))
            for img_path in self.files]

    def create_label_mapper(self):
        return None

    def create_cache_name(self):
        return f"{type(self).__name__}_v2"
