import os
from PIL import Image
from torch.utils import data
import numpy as np
from .label_mapping import create_wilddash_to_cityscapes_mapper

wilddash2_mean = np.array([0.4228, 0.4281, 0.4309])
wilddash2_std = np.array([0.2362, 0.2429, 0.2575])


mapper_wd_city = create_wilddash_to_cityscapes_mapper()


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


class Wilddash2(data.Dataset):
    def __init__(self, root, image_transform, target_transform, joint_transform):
        self.root = root
        self.tag = "shift"

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images_base = os.path.join(self.root, "images")
        self.annotations_base = os.path.join(self.root, "labels")

        self.images = recursive_glob(rootdir=self.images_base, suffix=".jpg")
        print("> Found %d images..." % len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index].rstrip()
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

        image = self.image_transform(image)
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
            for img_path in self.images
        ]

    def create_label_mapper(self):
        return mapper_wd_city

    def create_cache_name(self):
        return f"{type(self).__name__}_{self.tag}"
