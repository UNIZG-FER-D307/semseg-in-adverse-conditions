from torchvision import transforms as tf
import copy
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from data.datasets import (
    Wilddash2,
    MapillaryVistas,
    SubsetDataset,
    BDD,
    bdd_std,
    bdd_mean,
)
from data.datasets import (
    wilddash2_mean,
    wilddash2_std,
    create_wilddash_to_cityscapes_mapper,
    vistas_mean,
    vistas_std,
    create_vistas_to_cityscapes_mapper,
)
from data.transforms import JitterRandomCrop
from data.loaders.city_loader import prepare_city_datasets
from config import VISTAS_ROOT


def create_per_dataset_train_transform(
    transforms, mean, std, crop_size=1024, scale=(0.5, 2), ignore_id=255
):
    transforms["image"] = transforms["image"] + [tf.Normalize(mean, std)]
    transforms["joint"] = [
        JitterRandomCrop(
            size=crop_size, scale=scale, ignore_id=ignore_id, input_mean=(0.0)
        )
    ] + transforms["joint"]
    return transforms


mapper_wd_city = create_wilddash_to_cityscapes_mapper()


def lm_wd(x):
    return mapper_wd_city[(x * 255.0).long()]


import os

conf_file = VISTAS_ROOT
if not os.path.isdir(conf_file):
    raise Exception("No vistas config file!!")
mapper_vistas_city = create_vistas_to_cityscapes_mapper(conf_file)


def lm_vistas(x):
    return mapper_vistas_city[(x * 255.0).long()]


def lm(x):
    return (x * 255.0).long()


def prepare_traffic_datasets(dataroots, train_transforms, config):
    [cs_train] = prepare_city_datasets(dataroots, train_transforms, config)

    remap_labels_bdd = tf.Lambda(lm)
    bdd_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        bdd_mean,
        bdd_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    bdd_train = BDD(
        dataroots["bdd"],
        split="train",
        image_transform=tf.Compose(bdd_trans["image"]),
        target_transform=tf.Compose(bdd_trans["target"] + [remap_labels_bdd]),
        joint_transform=tf.Compose(bdd_trans["joint"]),
    )
    bdd_val = BDD(
        dataroots["bdd"],
        split="val",
        image_transform=tf.Compose(bdd_trans["image"]),
        target_transform=tf.Compose(bdd_trans["target"] + [remap_labels_bdd]),
        joint_transform=tf.Compose(bdd_trans["joint"]),
    )

    remap_labels_wd = tf.Lambda(lm_wd)
    wd_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        wilddash2_mean,
        wilddash2_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    wd = Wilddash2(
        dataroots["wilddash2"],
        image_transform=tf.Compose(wd_trans["image"]),
        target_transform=tf.Compose(wd_trans["target"] + [remap_labels_wd]),
        joint_transform=tf.Compose(wd_trans["joint"]),
    )

    remap_labels_vistas = tf.Lambda(lm_vistas)
    vistas_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        vistas_mean,
        vistas_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    vistas_train = MapillaryVistas(
        dataroots["vistas"],
        split="training",
        image_transform=tf.Compose(vistas_trans["image"]),
        target_transform=tf.Compose(vistas_trans["target"] + [remap_labels_vistas]),
        joint_transform=tf.Compose(vistas_trans["joint"]),
    )
    vistas_val = MapillaryVistas(
        dataroots["vistas"],
        split="validation",
        image_transform=tf.Compose(vistas_trans["image"]),
        target_transform=tf.Compose(vistas_trans["target"] + [remap_labels_vistas]),
        joint_transform=tf.Compose(vistas_trans["joint"]),
    )
    return [wd, bdd_train, bdd_val, vistas_train, vistas_val, cs_train, cs_val]


def prepare_traffic_finetune_datasets(dataroots, train_transforms, config):
    [cs_train, cs_val] = prepare_city_datasets(dataroots, train_transforms, config)

    remap_labels_bdd = tf.Lambda(lm)
    bdd_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        bdd_mean,
        bdd_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    bdd_train = BDD(
        dataroots["bdd"],
        split="train",
        image_transform=tf.Compose(bdd_trans["image"]),
        target_transform=tf.Compose(bdd_trans["target"] + [remap_labels_bdd]),
        joint_transform=tf.Compose(bdd_trans["joint"]),
    )
    bdd_val = BDD(
        dataroots["bdd"],
        split="val",
        image_transform=tf.Compose(bdd_trans["image"]),
        target_transform=tf.Compose(bdd_trans["target"] + [remap_labels_bdd]),
        joint_transform=tf.Compose(bdd_trans["joint"]),
    )

    remap_labels_wd = tf.Lambda(lm_wd)
    wd_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        wilddash2_mean,
        wilddash2_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    wd = Wilddash2(
        dataroots["wilddash2"],
        image_transform=tf.Compose(wd_trans["image"]),
        target_transform=tf.Compose(wd_trans["target"] + [remap_labels_wd]),
        joint_transform=tf.Compose(wd_trans["joint"]),
    )

    remap_labels_vistas = tf.Lambda(lm_vistas)
    vistas_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        vistas_mean,
        vistas_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    vistas_train = MapillaryVistas(
        dataroots["vistas"],
        split="training",
        image_transform=tf.Compose(vistas_trans["image"]),
        target_transform=tf.Compose(vistas_trans["target"] + [remap_labels_vistas]),
        joint_transform=tf.Compose(vistas_trans["joint"]),
    )
    return [wd, bdd_train, bdd_val, vistas_train, vistas_val, cs_train, cs_val]


def load_traffic_train(dataroots, bs, train_transforms, config):
    datasets = prepare_traffic_datasets(dataroots, train_transforms, config)
    ds = ConcatDataset(datasets)
    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(
        ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=5
    )
    return train_loader


def prepare_city_vistas_datasets(dataroots, train_transforms, config):
    [cs_train] = prepare_city_datasets(dataroots, train_transforms, config)

    remap_labels_vistas = tf.Lambda(lm_vistas)
    vistas_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        vistas_mean,
        vistas_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    vistas_train = MapillaryVistas(
        dataroots["vistas"],
        split="training",
        image_transform=tf.Compose(vistas_trans["image"]),
        target_transform=tf.Compose(vistas_trans["target"] + [remap_labels_vistas]),
        joint_transform=tf.Compose(vistas_trans["joint"]),
    )
    return [vistas_train, cs_train]


def load_city_vistas_train(dataroots, bs, train_transforms, config):
    datasets = prepare_city_vistas_datasets(dataroots, train_transforms, config)
    ds = ConcatDataset(datasets)
    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(
        ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=3
    )
    return train_loader
