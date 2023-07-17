import numpy as np
import os
from PIL import Image
from torch.utils import data
import glob


class BDDPseudolabeledConditionwise(data.Dataset):
    def __init__(
        self,
        root,
        split,
        condition,
        return_name=False,
        image_transform=None,
        target_transform=None,
        joint_transform=None,
    ):
        self.root = root
        self.split = split
        self.tag = condition
        assert split in ["train", "val"]
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images_base = os.path.join(self.root, "images", "100k", self.split)
        self.label_base = os.path.join(
            self.root, split, condition, "pseudolabelTrainIds_v2"
        )

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
        return images

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        name = img_path.split("/")[-1]
        lbl_path = os.path.join(
            self.label_base, name.replace(".jpg", "_pseudolabel.png")
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
                    self.label_base,
                    img_path.split("/")[-1].replace(".jpg", "_pseudolabel.png"),
                ),
            )
            for img_path in self.files[self.split]
        ]

    def create_label_mapper(self):
        return None

    def create_cache_name(self):
        return f"{type(self).__name__}_{self.split}_{self.tag}"
