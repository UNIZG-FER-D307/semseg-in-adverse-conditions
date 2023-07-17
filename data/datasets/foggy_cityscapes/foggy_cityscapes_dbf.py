import numpy as np
import os
from PIL import Image
from torch.utils import data

foggy_city_mean = np.array([0.43867884, 0.47308116, 0.43906222])
foggy_city_std = np.array([0.20987089, 0.20650373, 0.20920398])

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if
            (filename.endswith(suffix) and os.path.isfile(os.path.join(looproot, filename)))]

class FoggyCityscapes(data.Dataset):

    def __init__(self, root, split, image_transform=None, target_transform=None, joint_transform=None):
        self.root = root
        assert split in ['train', 'val']
        self.split = split
        self.tag = 'fog'

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images_base = os.path.join(self.root, 'leftImg8bit_foggyDBF', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        # self.files = recursive_glob(rootdir=self.images_base, suffix='0.01.png')
        self.files = recursive_glob(rootdir=self.images_base, suffix='.png')

        if not len(self.files):
            raise Exception("> No files for split=[%s] found in %s" % (split, self.images_base))

        print("> Found %d %s images..." % (len(self.files), split))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        name = img_path.split('_leftImg8bit_foggy_beta')[0] + '_gtFine_labelTrainIds.png'
        lbl_path = os.path.join(self.annotations_base, name.replace('leftImg8bit_foggyDBF', 'gtFine'))

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
            (img_path, os.path.join(self.annotations_base, (img_path.split('_leftImg8bit_foggy_beta')[0] + '_gtFine_labelTrainIds.png').replace('leftImg8bit_foggyDBF', 'gtFine')))
            for img_path in self.files]

    def create_label_mapper(self):
        return None

    def create_cache_name(self):
        return f"{type(self).__name__}_{self.split}_{self.tag}_all"
