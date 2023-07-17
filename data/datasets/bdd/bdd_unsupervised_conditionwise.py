import numpy as np
import os
from PIL import Image
from torch.utils import data
import glob

bdd_snow_mean = np.array([0.28450744, 0.29962513, 0.30584514])
bdd_snow_std = np.array([0.2401745, 0.25995231, 0.27574058])

bdd_fog_mean = np.array([0.30441993, 0.29822997, 0.28906247])
bdd_fog_std = np.array([0.25825409, 0.26739415, 0.27607027])

bdd_rain_mean = np.array([0.29844518, 0.28935396, 0.28026778])
bdd_rain_std = np.array([0.25924394, 0.26166157, 0.26514308])


class BDDConditionwiseUnsupervised(data.Dataset):
    def __init__(self, root, split, condition, return_name=False, image_transform=None):
        self.root = root
        self.split = split
        self.tag = "shift"
        assert split in ["train", "val"]
        assert condition in ["snowy", "rainy", "foggy"]
        self.image_transform = image_transform

        self.images_base = os.path.join(self.root, "images", "100k", self.split)

        self.files = {}
        images = glob.glob(self.images_base + "/*.jpg")
        valid_images = self._load_valid_images(condition, split)
        existing_images = set(images).intersection(set(valid_images))
        self.files[split] = list(existing_images)

        self.return_name = return_name

        if not self.files[split]:
            raise Exception(
                "> No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("> Found %d %s images..." % (len(self.files[split]), split))

    def _load_valid_images(self, condition, split):
        file = os.path.join(self.root, f"weather_{split}.txt")
        with open(file) as f:
            lines = f.readlines()
            lines = [line.rstrip().split(",") for line in lines]
        images = []
        for line in lines:
            img, cond = line
            if condition == cond:
                images.append(self.images_base + "/" + img)
        return sorted(list(set(images)))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception(
                "{} is not a file, can not open with imread.".format(img_path)
            )
        image = Image.open(img_path)

        if self.image_transform:
            image = self.image_transform(image)

        if self.return_name:
            return image, img_path

        return image
