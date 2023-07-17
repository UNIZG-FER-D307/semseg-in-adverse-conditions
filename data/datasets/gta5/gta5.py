import numpy as np
import os
from PIL import Image
from torch.utils import data

gta5_mean = np.array([0.44283997, 0.43858149, 0.42527347])
gta5_std = np.array([0.26131208, 0.2553786, 0.24981177])


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


class GTA5(data.Dataset):
    def __init__(
        self,
        root,
        image_transform=None,
        target_transform=None,
        joint_transform=None,
    ) -> None:
        super().__init__()

        self.root = root
        self.image_transform = image_transform

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images_base = os.path.join(self.root, "rgb")
        self.image_paths = recursive_glob(rootdir=self.images_base, suffix=".png")

        if not len(self.image_paths):
            raise Exception(f"> No files found in {self.root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        lbl_path = img_path.replace("rgb/", "labels/").replace(
            ".png", "_labelTrainIds.png"
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
                img_path.replace("rgb/", "labels/").replace(
                    ".png", "_labelTrainIds.png"
                ),
            )
            for img_path in self.image_paths
        ]

    def create_cache_name(self):
        return f"{type(self).__name__}_train_all"

    def create_label_mapper(self):
        return None
