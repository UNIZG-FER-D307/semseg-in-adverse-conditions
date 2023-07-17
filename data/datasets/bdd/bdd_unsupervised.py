import numpy as np
import os
from PIL import Image
from torch.utils import data
import glob


class BDDUnsupervised(data.Dataset):

    def __init__(self, root, split, return_name=False, image_transform=None):
        self.root = root
        self.split = split
        self.tag = 'shift'
        assert split in ['train', 'val', 'test']
        self.image_transform = image_transform

        self.images_base = os.path.join(self.root, 'images', '100k', self.split)

        self.files = {}
        self.files[split] = glob.glob(self.images_base + '/*.jpg')
        self.return_name = return_name

        if not self.files[split]:
            raise Exception("> No files for split=[%s] found in %s" % (split, self.images_base))

        print("> Found %d %s images..." % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        image = Image.open(img_path)

        if self.image_transform:
            image = self.image_transform(image)

        if self.return_name:
            return image, img_path

        return image

    # def create_img_lbl_list(self):
    #     return [
    #         (img_path, os.path.join(self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png")))
    #         for img_path in self.files[self.split]]
    #
    # def create_label_mapper(self):
    #     return None
    #
    # def create_cache_name(self):
    #     return f"{type(self).__name__}_{self.split}_{self.tag}"
