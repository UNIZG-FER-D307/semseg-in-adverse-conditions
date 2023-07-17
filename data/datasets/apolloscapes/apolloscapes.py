import os
from PIL import Image
from torch.utils import data
import numpy as np
from .labels import create_id_to_train_id_mapper

apolloscapes_mean = np.array([0.49359848, 0.53106967, 0.54439013])
apolloscapes_std = np.array([0.33517435, 0.34729513, 0.35095801])

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if
            (filename.endswith(suffix) and os.path.isfile(os.path.join(looproot, filename)))]

mapper_as_cs = create_id_to_train_id_mapper()

class Apolloscapes(data.Dataset):

    def __init__(self, root, split='train', image_transform=None, target_transform=None, joint_transform=None):
        self.root = root
        self.tag = 'shift'
        self.split = split
        assert split in ['train', 'val']

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images, self.labels = self._load_content(split)

        print("> Found %d images..." % len(self.labels))

    def __len__(self):
        return len(self.labels)

    def _load_content(self, split):
        content = []
        for road in ['road01', 'road02', 'road03']:
            filename = os.path.join(self.root, 'split_lists', f"{road}_ins_{split}.lst")
            f = open(filename)
            lines = [l.strip() for l in f.readlines()]
            content += lines
        return [os.path.join(self.root, c.split('\t')[0]) for c in content], [os.path.join(self.root, c.split('\t')[1]) for c in content]



    def __getitem__(self, index):
        lbl_path = self.labels[index].rstrip()
        img_path = self.images[index].rstrip()

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
        return [(im, lbl) for im, lbl in zip(self.images, self.labels)]

    def create_label_mapper(self):
        return mapper_as_cs

    def create_cache_name(self):
        return f"{type(self).__name__}_{self.tag}_{self.split}"
