import numpy as np
import os
from PIL import Image
import glob
from torch.utils import data

# TODO which mean and std to use?
# this one is cityscapes
# inference_data_mean = np.array([0.2869, 0.3251, 0.2839])
# inference_data_std = np.array([0.1761, 0.1810, 0.1777])

# this one is vistas
inference_data_mean = np.array([0.4193, 0.4585, 0.4700])
inference_data_std = np.array([0.2463, 0.2596, 0.2854])


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


class InferenceDataset(data.Dataset):
    def __init__(
        self, root, split="", ext="png", return_name=False, image_transform=None
    ):
        self.root = root
        self.split = split
        self.image_transform = image_transform
        self.return_name = return_name
        self.ext = ext

        if os.path.isdir(self.root):
            if split != "":
                self.images_base = os.path.join(self.root, self.split)
            else:
                self.images_base = self.root

            self.files = recursive_glob(self.images_base, f".{self.ext}")
            self.files = sorted(self.files)
        else:
            if root[-len(self.ext) - 1 :] != f".{self.ext}":
                raise ValueError(
                    f"Extension {self.ext} different from extension of file: {root}"
                )
            self.files = [root]

        print(self.files)

        if not len(self.files):
            raise Exception(
                "> No files found in %s with extension %s"
                % (split, self.images_base, self.ext)
            )

        print("> Found %d %s images..." % (len(self.files), split))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]

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
