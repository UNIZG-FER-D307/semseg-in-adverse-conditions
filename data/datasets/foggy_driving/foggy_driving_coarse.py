import numpy as np
import os
from PIL import Image
from torch.utils import data

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if
            (filename.endswith(suffix) and os.path.isfile(os.path.join(looproot, filename)))]

class FoggyDrivingCoarse(data.Dataset):

    def __init__(self, root, image_transform=None, target_transform=None, joint_transform=None):
        self.root = root
        self.tag = 'fog'

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images_base = os.path.join(self.root, 'leftImg8bit', 'test_extra')
        self.annotations_base = os.path.join(self.root, 'gtCoarse', 'test_extra')

        self.files = recursive_glob(rootdir=self.annotations_base, suffix='_gtCoarse_labelTrainIds.png')

        if not len(self.files):
            raise Exception("> No files found in %s" % (self.root))

        print("> Found %d images..." % len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        lbl_path = self.files[index]
        name = '/'.join(lbl_path.split('/')[-2:])
        img_path = os.path.join(self.images_base, name.replace('_gtCoarse_labelTrainIds.png', '_leftImg8bit.png'))

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
            (os.path.join(self.images_base, '/'.join(lbl_path.split('/')[-2:]).replace('_gtCoarse_labelTrainIds.png', '_leftImg8bit.png')), lbl_path)
            for lbl_path in self.files]

    def create_label_mapper(self):
        return None

    def create_cache_name(self):
        return f"{type(self).__name__}_{self.tag}"
