from torchvision import transforms as tf
from torch.utils.data import DataLoader
from data.datasets import DarkZurich, DarkZurichFull


def lm(x):
    return (x * 255.0).long()


def load_dark_zurich_train(dataroot, bs, train_transforms):
    remap_labels = tf.Lambda(lm)
    train_set = DarkZurich(
        dataroot,
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
    print(f"> Loaded {len(train_set)} train images.")
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, pin_memory=False, num_workers=3
    )
    return train_loader


def load_dark_zurich_val(dataroot, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_set = DarkZurich(
        dataroot,
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
        val_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=2
    )
    return val_loader


def load_dark_zurich_triplets(dataroot, bs, selfsup_transforms):
    train_set = DarkZurichFull(
        dataroot,
        split="train",
        day_image_transform=tf.Compose(selfsup_transforms["day"]),
        night_image_transform=tf.Compose(selfsup_transforms["night"]),
        twilight_image_transform=tf.Compose(selfsup_transforms["twilight"]),
    )
    print(f"> Loaded {len(train_set)} train images.")
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, pin_memory=False, num_workers=4
    )
    return train_loader
