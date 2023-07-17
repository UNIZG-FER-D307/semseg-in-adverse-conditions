import numpy as np
import os
from PIL import Image
from torch.utils import data
import glob

cadcd_mean = np.array([0.4891692,  0.49386807, 0.5006766])
cadcd_std = np.array([0.2327716,  0.23851485, 0.24255964])

class CADCDUnsupervised(data.Dataset):

    def __init__(self, root, return_name=False, image_transform=None):
        self.root = root
        self.tag = 'snow'
        self.image_transform = image_transform

        self.images_base = os.path.join(self.root)

        self.files = sorted(glob.glob(self.images_base + '/images/**/*.png', recursive=True))
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
