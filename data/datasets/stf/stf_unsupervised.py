import numpy as np
import os
from PIL import Image
from torch.utils import data
import glob

stf_mean = np.array([0.37385367, 0.35929916, 0.34523512])
stf_std = np.array([0.16730218, 0.16814514, 0.1715073])

class STFUnsupervised(data.Dataset):

    def __init__(self, root, return_name=False, image_transform=None):
        self.root = root
        self.tag = 'fog'
        self.image_transform = image_transform

        self.images_base = os.path.join(self.root)

        self.files = sorted(glob.glob(self.images_base + '/images/*.png'))
        self.return_name = return_name

        if not self.files:
            raise Exception("> No files found in %s" % (self.images_base))

        print("> Found %d images..." % (len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index].rstrip()

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
