import numpy as np
import os
from PIL import Image
from torch.utils import data
import glob
bdd_mean = np.array([0.37016466, 0.41445044, 0.4244545])
bdd_std = np.array([0.25246277, 0.26930884, 0.28654876])


class BDD(data.Dataset):

    def __init__(self, root, split, image_transform=None, target_transform=None, joint_transform=None):
        self.root = root
        self.split = split
        self.tag = 'shift'
        assert split in ['train', 'val']
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images_base = os.path.join(self.root, 'images', '10k', self.split)
        self.annotations_base = os.path.join(self.root, 'labels', 'sem_seg', 'masks', self.split)

        self.files = {}
        self.files[split] = glob.glob(self.images_base + '/*.jpg')

        if not self.files[split]:
            raise Exception("> No files for split=[%s] found in %s" % (split, self.images_base))

        print("> Found %d %s images..." % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png"))

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
            (img_path, os.path.join(self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png")))
            for img_path in self.files[self.split]]

    def create_label_mapper(self):
        return None

    def create_cache_name(self):
        return f"{type(self).__name__}_{self.split}_{self.tag}"
