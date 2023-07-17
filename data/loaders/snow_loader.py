from torchvision import transforms as tf
from torch.utils.data import DataLoader, ConcatDataset
import copy
from data.datasets import ACDC, MapillaryVistasSnow, create_vistas_to_cityscapes_mapper
from data.datasets import (
    acdc_snow_mean,
    acdc_snow_std,
    vistas_snow_std,
    vistas_snow_mean,
)
from data.transforms import JitterRandomCrop
from data.datasets import (
    bdd_snow_std,
    bdd_snow_mean,
    BDDPseudolabeledConditionwise,
    cadcd_std,
    cadcd_mean,
    CADCDPseudolabeled,
    UniformClassDataset,
)


def create_per_dataset_train_transform(
    transforms, mean, std, crop_size, scale, ignore_id
):
    input_mean = tuple([int(a * 255) for a in mean.tolist()])
    transforms["image"] = transforms["image"] + [tf.Normalize(mean, std)]
    transforms["joint"] = [
        JitterRandomCrop(
            size=crop_size, scale=scale, ignore_id=ignore_id, input_mean=(0.0)
        )
    ] + transforms["joint"]
    return transforms


def lm(x):
    return (x * 255.0).long()


import os
from config import VISTAS_ROOT

conf_file = VISTAS_ROOT
if not os.path.isdir(conf_file):
    raise Exception("No vistas config file!!")
mapper_vistas_city = create_vistas_to_cityscapes_mapper(conf_file)


def lm_vistas(x):
    return mapper_vistas_city[(x * 255.0).long()]


def prepare_snow_datasets(dataroots, train_transforms, config):
    remap_labels = tf.Lambda(lm)
    acdc_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_snow_mean,
        acdc_snow_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    acdc_snow = ACDC(
        dataroots["acdc"],
        split="train",
        tag="snow",
        image_transform=tf.Compose(acdc_trans["image"]),
        target_transform=tf.Compose(acdc_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(acdc_trans["joint"]),
    )

    remap_vistas_labels = tf.Lambda(lm_vistas)
    vistassnow_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        vistas_snow_mean,
        vistas_snow_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    vistas_snow_train = MapillaryVistasSnow(
        dataroots["vistas"],
        split="training",
        image_transform=tf.Compose(vistassnow_trans["image"]),
        target_transform=tf.Compose(vistassnow_trans["target"] + [remap_vistas_labels]),
        joint_transform=tf.Compose(vistassnow_trans["joint"]),
    )
    vistas_snow_val = MapillaryVistasSnow(
        dataroots["vistas"],
        split="validation",
        image_transform=tf.Compose(vistassnow_trans["image"]),
        target_transform=tf.Compose(vistassnow_trans["target"] + [remap_vistas_labels]),
        joint_transform=tf.Compose(vistassnow_trans["joint"]),
    )

    bdd_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        bdd_snow_mean,
        bdd_snow_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    bdd_snow = BDDPseudolabeledConditionwise(
        dataroots["bdd100k"],
        split="train",
        condition="snowy",
        image_transform=tf.Compose(bdd_trans["image"]),
        target_transform=tf.Compose(bdd_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(bdd_trans["joint"]),
    )

    cadcd_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        cadcd_mean,
        cadcd_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    cadcd_snow = CADCDPseudolabeled(
        dataroots["cadcd"],
        image_transform=tf.Compose(cadcd_trans["image"]),
        target_transform=tf.Compose(cadcd_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(cadcd_trans["joint"]),
    )

    return [acdc_snow, vistas_snow_train, vistas_snow_val, bdd_snow, cadcd_snow]


def load_snow_finetune(dataroots, bs, train_transforms, config):
    ds_rain = prepare_snow_datasets(dataroots, train_transforms, config)

    if config["uniform"]:
        ds_rain = [
            UniformClassDataset(ds, class_uniform_pct=0.75, class_uniform_tile=512)
            for ds in ds_rain
        ]
        print("> Created uniform dataset")
    ds = ConcatDataset(ds_rain)

    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(
        ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=5, drop_last=True
    )
    return train_loader


def load_snow_train(dataroots, bs, train_transforms, config):
    datasets = prepare_snow_datasets(dataroots, train_transforms, config)
    ds = datasets[0]
    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(
        ds, batch_size=bs, shuffle=True, pin_memory=False, num_workers=3
    )
    return train_loader


def load_snow_acdc_val(dataroot, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_set = ACDC(
        dataroot,
        split="val",
        tag="snow",
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


def load_snow_eval(dataroots, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_transforms["image"] = val_transforms["image"] + [
        tf.Normalize(acdc_snow_mean, acdc_snow_std)
    ]
    acdc_trans = val_transforms
    dataset = ACDC(
        dataroots["acdc"],
        split="val",
        tag="snow",
        image_transform=tf.Compose(acdc_trans["image"]),
        target_transform=tf.Compose(acdc_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(acdc_trans["joint"]),
    )
    val_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=2
    )
    return val_loader
