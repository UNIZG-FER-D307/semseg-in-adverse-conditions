import numpy as np
import os
from PIL import Image
from torch.utils import data
import glob
from data.datasets.cityscapes.cityscapes_labels import create_id_to_train_id_mapper

night_city_mean = np.array([0.2168323, 0.19338639, 0.17289599])
night_city_std = np.array([0.19265372, 0.18376744, 0.1763268])

night_city_mapper = create_id_to_train_id_mapper()


class NightCityUnsupervised(data.Dataset):
    def __init__(self, root, split="train", return_name=False, image_transform=None):
        self.root = root
        assert split in ["train", "val"]
        self.split = split
        self.tag = "night"
        self.return_name = return_name

        self.image_transform = image_transform

        self.images_base = os.path.join(self.root, "images", self.split)

        self.files = glob.glob(self.images_base + "/*.png")

        if not len(self.files):
            raise Exception(
                "> No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("> Found %d %s images..." % (len(self.files), split))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        name = img_path.split("/")[-1]

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

    def create_img_lbl_list(self):
        return [
            (
                img_path,
                os.path.join(
                    self.annotations_base,
                    img_path.split("/")[-1].replace(".png", "_labelIds.png"),
                ),
            )
            for img_path in self.files
        ]

    def create_label_mapper(self):
        return night_city_mapper

    def create_cache_name(self):
        return f"{type(self).__name__}_{self.split}_{self.tag}"
