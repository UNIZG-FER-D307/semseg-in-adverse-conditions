import numpy as np
import os
from PIL import Image
from torch.utils import data


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


class DarkZurichUnsupervised(data.Dataset):
    def __init__(
        self,
        root,
        split="val",
        condition="night",
        return_name=False,
        image_transform=None,
    ):
        self.root = root
        self.split = split
        self.image_transform = image_transform
        assert split in ["train", "val"]
        assert condition in ["night", "day", "twilight"]

        self.images_base = os.path.join(self.root, "rgb_anon", self.split, condition)
        self.files = recursive_glob(rootdir=self.images_base, suffix=".png")
        self.return_name = return_name

        if not len(self.files):
            raise Exception(
                "> No files for split=[%s] found in %s" % (split, self.images_base)
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

    # def create_img_lbl_list(self):
    #     return [
    #         (img_path, os.path.join(self.annotations_base, '/'.join(img_path.split('/')[-2:]).replace("_rgb_anon.png", "_gt_labelTrainIds.png")))
    #         for img_path in self.files]
    #
    # def create_label_mapper(self):
    #     return None
    #
    # def create_cache_name(self):
    #     return f"{type(self).__name__}_{self.split}_{self.tag}"
