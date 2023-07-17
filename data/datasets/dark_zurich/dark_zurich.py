import numpy as np
import os
from PIL import Image
from torch.utils import data

dark_zurich_mean = np.array([0.2110673, 0.1518971, 0.09200527])
dark_zurich_std = np.array([0.1772894, 0.14995785, 0.1104107])

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if
            (filename.endswith(suffix) and os.path.isfile(os.path.join(looproot, filename)))]

class DarkZurich(data.Dataset):

    def __init__(self, root, split='val', image_transform=None, target_transform=None, joint_transform=None):
        self.root = root
        self.split = split
        self.tag = 'night'

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images_base = os.path.join(self.root, 'rgb_anon', self.split, self.tag)
        self.annotations_base = os.path.join(self.root, 'gt', self.split, self.tag)

        self.files = recursive_glob(rootdir=self.images_base, suffix='.png')

        if not len(self.files):
            raise Exception("> No files for split=[%s] found in %s" % (split, self.images_base))

        print("> Found %d %s images..." % (len(self.files), split))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        name = '/'.join(img_path.split('/')[-2:])
        lbl_path = os.path.join(self.annotations_base, name.replace("_rgb_anon.png", "_gt_labelTrainIds.png"))

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
            (img_path, os.path.join(self.annotations_base, '/'.join(img_path.split('/')[-2:]).replace("_rgb_anon.png", "_gt_labelTrainIds.png")))
            for img_path in self.files]

    def create_label_mapper(self):
        return None

    def create_cache_name(self):
        return f"{type(self).__name__}_{self.split}_{self.tag}"
