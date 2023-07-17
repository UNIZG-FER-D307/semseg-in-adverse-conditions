import numpy as np
import os
from PIL import Image
from torch.utils import data
import glob

vistas_snow_mean = np.array([0.43777775, 0.48304881, 0.50733597])
vistas_snow_std = np.array([0.26335171, 0.27729311, 0.29820139])
import json
from .mapping import create_vistas_to_cityscapes_mapper

import os
from config import VISTAS_ROOT

conf_file = VISTAS_ROOT
if not os.path.isdir(conf_file):
    raise Exception("No vistas config file!!")
mapper_vistas_city = create_vistas_to_cityscapes_mapper(conf_file)


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
    :param rootdir is the root directory
    :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if (
            filename.endswith(suffix)
            and os.path.isfile(os.path.join(looproot, filename))
        )
    ]


class MapillaryVistasSnow(data.Dataset):
    n_classes = 66

    def __init__(
        self,
        root,
        split,
        image_transform=None,
        target_transform=None,
        joint_transform=None,
    ):
        self.root = root
        self.split = split
        self.tag = "shift"
        assert split in ["training", "validation"]
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images_base = os.path.join(self.root, self.split, "images")
        self.annotations_base = os.path.join(self.root, self.split, "labels")

        self.files = {}
        self.files[split] = glob.glob(self.images_base + "/*.jpg")
        os.mkdir("./cache") if not os.path.isdir("./cache") else None
        json_fn = "./cache/vistas_snow_imgs" + self.split + ".json"
        if os.path.isfile(json_fn):
            with open(json_fn, "r") as json_data:
                files = json.load(json_data)
            self.files[split] = files
        else:
            self.files[split] = self._filter_snow(self.files[split])
            with open(json_fn, "w") as json_data:
                json.dump(self.files[split], json_data)

        if not self.files[split]:
            raise Exception(
                "> No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("> Found %d %s images..." % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def _filter_snow(self, images):
        snow_id = 28

        def has_snow(img_path):
            lbl_path = os.path.join(
                self.annotations_base,
                os.path.basename(img_path).replace(".jpg", ".png"),
            )
            lbl = Image.open(lbl_path)
            classes = np.unique(np.array(lbl)).tolist()
            return snow_id in classes

        return list(filter(has_snow, images))

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png")
        )

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception(
                "{} is not a file, can not open with imread.".format(img_path)
            )
        image = Image.open(img_path)

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception(
                "{} is not a file, can not open with imread.".format(lbl_path)
            )
        label = Image.open(lbl_path)
        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return (
            self.joint_transform((image, label))
            if self.joint_transform
            else (image, label)
        )

    def create_img_lbl_list(self):
        return [
            (
                img_path,
                os.path.join(
                    self.annotations_base,
                    os.path.basename(img_path).replace(".jpg", ".png"),
                ),
            )
            for img_path in self.files[self.split]
        ]

    def create_label_mapper(self):
        return mapper_vistas_city

    def create_cache_name(self):
        return f"{type(self).__name__}_{self.split}_{self.tag}"
