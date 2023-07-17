from torchvision import transforms as tf
from torch.utils.data import DataLoader
import copy
from data.datasets import (
    Cityscapes,
    cityscapes_mean,
    cityscapes_std,
    UniformClassDataset,
    CityscapesTest,
    CityscapesWithOOD,
)
from data.datasets.cityscapes.cityscapes_labels import create_id_to_train_id_mapper
from data.transforms import JitterRandomCrop
from torch.utils.data import ConcatDataset

label_mapper = create_id_to_train_id_mapper()


def create_per_dataset_train_transform(
    transforms, mean, std, crop_size, scale, ignore_id
):
    transforms["image"] = transforms["image"] + [tf.Normalize(mean, std)]
    transforms["joint"] = [
        JitterRandomCrop(
            size=crop_size, scale=scale, ignore_id=ignore_id, input_mean=(0.0)
        )
    ] + transforms["joint"]
    return transforms


def lm(x):
    return label_mapper[(x * 255.0).long()]


def prepare_city_datasets(dataroots, train_transforms, config):
    remap_labels = tf.Lambda(lm)
    trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        cityscapes_mean,
        cityscapes_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    train_set = Cityscapes(
        dataroots["city"],
        split="train",
        image_transform=None if not trans["image"] else tf.Compose(trans["image"]),
        target_transform=None
        if not trans["target"]
        else tf.Compose(trans["target"] + [remap_labels]),
        joint_transform=None if not trans["joint"] else tf.Compose(trans["joint"]),
    )
    # val_set = Cityscapes(dataroots['city'], split='val',
    #                        image_transform=None if not trans['image'] else tf.Compose(trans['image']),
    #                        target_transform=None if not trans['target'] else tf.Compose(trans['target'] + [remap_labels]),
    #                        joint_transform=None if not trans['joint'] else tf.Compose(trans['joint']))
    # return [train_set, val_set]
    return [train_set]


def load_city_train(dataroot, bs, train_transforms, uniform=False):
    remap_labels = tf.Lambda(lm)
    train_set = Cityscapes(
        dataroot,
        split="train",
        image_transform=None
        if not train_transforms["image"]
        else tf.Compose(train_transforms["image"]),
        target_transform=None
        if not train_transforms["target"]
        else tf.Compose(train_transforms["target"] + [remap_labels]),
        joint_transform=None
        if not train_transforms["joint"]
        else tf.Compose(train_transforms["joint"]),
    )
    train_set = (
        train_set
        if not uniform
        else UniformClassDataset(
            train_set, class_uniform_pct=0.75, class_uniform_tile=512
        )
    )
    print(f"> Loaded {len(train_set)} train images.")
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=3
    )
    return train_loader


def load_city_val(dataroot, val_transforms, bs=1):
    remap_labels = tf.Lambda(lm)
    val_set = Cityscapes(
        dataroot,
        split="val",
        image_transform=None
        if not val_transforms["image"]
        else tf.Compose(val_transforms["image"]),
        target_transform=None
        if not val_transforms["target"]
        else tf.Compose(val_transforms["target"] + [remap_labels]),
        joint_transform=None
        if not val_transforms["joint"]
        else tf.Compose(val_transforms["joint"]),
    )
    print(f"> Loaded {len(val_set)} val images.")
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False, pin_memory=False, num_workers=2
    )
    return val_loader


def load_city_trainval(dataroot, bs, train_transforms, uniform=False):
    remap_labels = tf.Lambda(lm)
    train_set = Cityscapes(
        dataroot,
        split="train",
        image_transform=None
        if not train_transforms["image"]
        else tf.Compose(train_transforms["image"]),
        target_transform=None
        if not train_transforms["target"]
        else tf.Compose(train_transforms["target"] + [remap_labels]),
        joint_transform=None
        if not train_transforms["joint"]
        else tf.Compose(train_transforms["joint"]),
    )
    val_set = Cityscapes(
        dataroot,
        split="val",
        image_transform=None
        if not train_transforms["image"]
        else tf.Compose(train_transforms["image"]),
        target_transform=None
        if not train_transforms["target"]
        else tf.Compose(train_transforms["target"] + [remap_labels]),
        joint_transform=None
        if not train_transforms["joint"]
        else tf.Compose(train_transforms["joint"]),
    )
    ds = ConcatDataset(
        [
            set
            if not uniform
            else UniformClassDataset(
                set, class_uniform_pct=0.75, class_uniform_tile=512
            )
            for set in [train_set, val_set]
        ]
    )
    print(f"> Loaded {len(train_set)} train images.")
    loader = DataLoader(ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=3)
    return loader


def load_city_test(dataroot, val_transforms):
    val_set = CityscapesTest(
        dataroot,
        split="test",
        return_name=True,
        image_transform=None
        if not val_transforms["image"]
        else tf.Compose(val_transforms["image"]),
    )
    print(f"> Loaded {len(val_set)} val images.")
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=2
    )
    return val_loader


def load_city_trainval_with_ood(dataroot, bs, train_transforms, uniform=False):
    remap_labels = tf.Lambda(lm)
    train_set = CityscapesWithOOD(
        dataroot,
        split="train",
        image_transform=None
        if not train_transforms["image"]
        else tf.Compose(train_transforms["image"]),
        target_transform=None
        if not train_transforms["target"]
        else tf.Compose(train_transforms["target"] + [remap_labels]),
        joint_transform=None
        if not train_transforms["joint"]
        else tf.Compose(train_transforms["joint"]),
        ood_transforms=None
        if not train_transforms["ood"]
        else tf.Compose(train_transforms["ood"]),
    )
    val_set = CityscapesWithOOD(
        dataroot,
        split="val",
        image_transform=None
        if not train_transforms["image"]
        else tf.Compose(train_transforms["image"]),
        target_transform=None
        if not train_transforms["target"]
        else tf.Compose(train_transforms["target"] + [remap_labels]),
        joint_transform=None
        if not train_transforms["joint"]
        else tf.Compose(train_transforms["joint"]),
        ood_transforms=None
        if not train_transforms["ood"]
        else tf.Compose(train_transforms["ood"]),
    )
    ds = ConcatDataset(
        [
            set
            if not uniform
            else UniformClassDataset(
                set, class_uniform_pct=0.75, class_uniform_tile=512
            )
            for set in [train_set, val_set]
        ]
    )
    print(f"> Loaded {len(train_set)} train images.")
    loader = DataLoader(
        ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=3, drop_last=True
    )
    return loader
