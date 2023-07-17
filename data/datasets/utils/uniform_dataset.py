import json
import os
import warnings
from PIL import Image
from torch.utils import data
from .uniform_sampling import class_centroids_all as class_centroids_all_
from .uniform_sampling import build_epoch as build_epoch_


class UniformClassDataset(data.Dataset):
    def __init__(
        self, dataset, class_uniform_pct=0.5, class_uniform_tile=1024, num_classes=19
    ):
        assert class_uniform_pct > 0 and class_uniform_pct <= 1
        self.joint_transform_list = dataset.joint_transform.transforms
        self.image_transform = dataset.image_transform
        self.target_transform = dataset.target_transform
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.imgs = dataset.create_img_lbl_list()
        self.num_classes = num_classes
        assert len(self.imgs), "Found 0 images, please check the data set"

        self.centroids = self._construct_centroids(dataset, num_classes)
        self.build_epoch()
        warnings.warn("Make sure you call build epoch after every epoch!")

    def _construct_centroids(self, dataset, num_classes):
        os.mkdir("./cache") if not os.path.isdir("./cache") else None
        json_fn = "./cache/" + dataset.create_cache_name() + ".centroids"
        if os.path.isfile(json_fn):
            with open(json_fn, "r") as json_data:
                centroids = json.load(json_data)
            centroids = {int(idx): centroids[idx] for idx in centroids}
        else:
            centroids = class_centroids_all_(
                self.imgs,
                num_classes,
                id2trainid=dataset.create_label_mapper(),
                tile_size=self.class_uniform_tile,
            )
            with open(json_fn, "w") as outfile:
                json.dump(centroids, outfile, indent=4)
        return centroids

    def build_epoch(self):
        """
        Perform Uniform Sampling per epoch to create a new list for training such that it
        uniformly samples all classes
        """
        self.imgs_uniform = build_epoch_(
            self.imgs, self.centroids, self.num_classes, self.class_uniform_pct
        )

    def __getitem__(self, index):
        elem = self.imgs_uniform[index]
        if len(elem) == 4:
            img_path, lbl_path, centroid, _ = elem
        else:
            img_path, lbl_path = elem
            centroid = None
        image, label = Image.open(img_path).convert("RGB"), Image.open(lbl_path)

        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        out = (image, label)
        if self.joint_transform_list is not None:
            for idx, trans in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    out = trans((*out, centroid))
                else:
                    out = trans(out)

        return out

    def __len__(self):
        return len(self.imgs_uniform)
