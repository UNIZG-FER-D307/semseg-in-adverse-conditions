from torchvision import transforms as tf
from torch.utils.data import DataLoader
import copy
import torch
from torch.utils.data import ConcatDataset
from data.datasets import (
    DarkZurich,
    ACDC,
    NightCity,
    UniformClassDataset,
    DarkZurichPseudolabeled,
    dark_zurich_night_mean,
    dark_zurich_twilight_mean,
    dark_zurich_night_std,
    dark_zurich_twilight_std,
)
from data.datasets import (
    dark_zurich_std,
    dark_zurich_mean,
    acdc_night_mean,
    acdc_night_std,
    night_city_std,
    night_city_mean,
)
from data.transforms import JitterRandomCrop
from data.transforms.joint_transforms.transforms import AppendTransitionMatrix
from data.datasets.cityscapes.cityscapes_labels import create_id_to_train_id_mapper
from data.datasets import (
    NightOwlsPseudolabeled,
    nightowls_mean,
    nightowls_std,
    NightCityPseudolabeled,
)


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


def create_per_dataset_train_transform_with_noisy_lbl(
    transforms, mean, std, crop_size, scale, ignore_id, T
):
    transforms["image"] = transforms["image"] + [tf.Normalize(mean, std)]
    transforms["joint"] = (
        [
            JitterRandomCrop(
                size=crop_size, scale=scale, ignore_id=ignore_id, input_mean=(0.0)
            )
        ]
        + transforms["joint"]
        + [AppendTransitionMatrix(T)]
    )
    return transforms


def lm(x):
    return (x * 255.0).long()


night_city_mapper = create_id_to_train_id_mapper()


def lm_night_city(x):
    return night_city_mapper[(x * 255.0).long()]


def prepare_dark_datasets(dataroots, train_transforms, config):
    remap_labels = tf.Lambda(lm)
    dz_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        dark_zurich_mean,
        dark_zurich_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dz = DarkZurich(
        dataroots["dark_zurich"],
        image_transform=tf.Compose(dz_trans["image"]),
        target_transform=tf.Compose(dz_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(dz_trans["joint"]),
    )

    acdc_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_night_mean,
        acdc_night_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    acdc_dark = ACDC(
        dataroots["acdc"],
        split="train",
        tag="night",
        image_transform=tf.Compose(acdc_trans["image"]),
        target_transform=tf.Compose(acdc_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(acdc_trans["joint"]),
    )

    # nc_trans = create_per_dataset_train_transform(copy.deepcopy(train_transforms), night_city_mean, night_city_std,
    #                                                 crop_size=config['crop_size'], scale=config['jitter_range'],
    #                                                 ignore_id=config['ignore_id'])
    # remap_nc_labels = tf.Lambda(lm_night_city)
    # nc_train = NightCity(dataroots['night_city'], split='train',
    #                  image_transform=tf.Compose(nc_trans['image']),
    #                  target_transform=tf.Compose(nc_trans['target'] + [remap_nc_labels]),
    #                  joint_transform=tf.Compose(nc_trans['joint']))
    # nc_val = NightCity(dataroots['night_city'], split='val',
    #                  image_transform=tf.Compose(nc_trans['image']),
    #                  target_transform=tf.Compose(nc_trans['target'] + [remap_nc_labels]),
    #                  joint_transform=tf.Compose(nc_trans['joint']))

    remap_labels = tf.Lambda(lm)
    dz_night_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        dark_zurich_night_mean,
        dark_zurich_night_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dz_twilight_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        dark_zurich_twilight_mean,
        dark_zurich_twilight_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dz_pseudo_night = DarkZurichPseudolabeled(
        dataroots["dark_zurich"],
        split="train",
        condition="night",
        image_transform=tf.Compose(dz_night_trans["image"]),
        target_transform=tf.Compose(dz_night_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(dz_night_trans["joint"]),
    )
    dz_pseudo_twilight = DarkZurichPseudolabeled(
        dataroots["dark_zurich"],
        split="train",
        condition="twilight",
        image_transform=tf.Compose(dz_twilight_trans["image"]),
        target_transform=tf.Compose(dz_twilight_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(dz_twilight_trans["joint"]),
    )

    nowls_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        nightowls_mean,
        nightowls_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    nowls_train = NightOwlsPseudolabeled(
        dataroots["night_owls"],
        image_transform=tf.Compose(nowls_trans["image"]),
        target_transform=tf.Compose(nowls_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(nowls_trans["joint"]),
    )

    nc_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        night_city_mean,
        night_city_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    nc_train = NightCityPseudolabeled(
        dataroots["night_city"],
        image_transform=tf.Compose(nc_trans["image"]),
        target_transform=tf.Compose(nc_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(nc_trans["joint"]),
    )

    # datasets = [dz, acdc_dark, nc_train]
    # datasets = [nc_train]
    datasets = [
        dz,
        acdc_dark,
        dz_pseudo_night,
        dz_pseudo_twilight,
        nowls_train,
        nc_train,
    ]
    return datasets


def load_night_finetune(dataroots, bs, train_transforms, config):
    ds_rain = prepare_dark_datasets(dataroots, train_transforms, config)
    if config["uniform_class"]:
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


def load_night_city_for_confmat(dataroot, val_transforms):
    remap_labels = tf.Lambda(lm_night_city)
    val_set = NightCity(
        dataroot,
        split="train",
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


def load_dark_train(dataroots, bs, train_transforms, config):
    datasets = prepare_dark_datasets(dataroots, train_transforms, config)
    if config["uniform"]:
        datasets = [
            UniformClassDataset(ds, class_uniform_pct=0.75, class_uniform_tile=512)
            for ds in datasets
        ]
        print("Uniform datasets created")
    ds = ConcatDataset(datasets)
    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(
        ds, batch_size=bs, shuffle=True, pin_memory=False, num_workers=3
    )
    return train_loader


def prepare_dark_datasets_with_noisy_labels(dataroots, train_transforms, config):
    remap_labels = tf.Lambda(lm)
    dz_trans = create_per_dataset_train_transform_with_noisy_lbl(
        copy.deepcopy(train_transforms),
        dark_zurich_mean,
        dark_zurich_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
        T=torch.eye(config["num_classes"]),
    )
    dz = DarkZurich(
        dataroots["dark_zurich"],
        image_transform=tf.Compose(dz_trans["image"]),
        target_transform=tf.Compose(dz_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(dz_trans["joint"]),
    )

    acdc_trans = create_per_dataset_train_transform_with_noisy_lbl(
        copy.deepcopy(train_transforms),
        acdc_night_mean,
        acdc_night_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
        T=torch.eye(config["num_classes"]),
    )
    acdc_dark = ACDC(
        dataroots["acdc"],
        split="train",
        tag="night",
        image_transform=tf.Compose(acdc_trans["image"]),
        target_transform=tf.Compose(acdc_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(acdc_trans["joint"]),
    )

    cm = torch.load("./cache/night_city_confmat_v2.pt")
    T = cm
    nc_trans = create_per_dataset_train_transform_with_noisy_lbl(
        copy.deepcopy(train_transforms),
        night_city_mean,
        night_city_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
        T=T,
    )
    remap_nc_labels = tf.Lambda(lm_night_city)
    nc_train = NightCity(
        dataroots["night_city"],
        split="train",
        image_transform=tf.Compose(nc_trans["image"]),
        target_transform=tf.Compose(nc_trans["target"] + [remap_nc_labels]),
        joint_transform=tf.Compose(nc_trans["joint"]),
    )
    # nc_val = NightCity(dataroots['night_city'], split='val',
    #                  image_transform=tf.Compose(nc_trans['image']),
    #                  target_transform=tf.Compose(nc_trans['target'] + [remap_nc_labels]),
    #                  joint_transform=tf.Compose(nc_trans['joint']))

    datasets = [dz, acdc_dark, nc_train]
    # datasets = [nc_train]
    # datasets = [dz, acdc_dark]
    return datasets


def load_dark_train_noisy(dataroots, bs, train_transforms, config):
    datasets = prepare_dark_datasets_with_noisy_labels(
        dataroots, train_transforms, config
    )
    if config["uniform"]:
        datasets = [
            UniformClassDataset(ds, class_uniform_pct=0.75, class_uniform_tile=512)
            for ds in datasets
        ]
        print("Uniform datasets created")
    ds = ConcatDataset(datasets)
    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(
        ds, batch_size=bs, shuffle=True, pin_memory=False, num_workers=0
    )
    return train_loader


def load_dark_acdc_val(dataroot, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_set = ACDC(
        dataroot,
        split="val",
        tag="night",
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


def load_dark_eval(dataroots, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_transforms["image"] = val_transforms["image"] + [
        tf.Normalize(acdc_night_mean, acdc_night_std)
    ]
    acdc_trans = val_transforms
    dataset = ACDC(
        dataroots["acdc"],
        split="val",
        tag="night",
        image_transform=tf.Compose(acdc_trans["image"]),
        target_transform=tf.Compose(acdc_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(acdc_trans["joint"]),
    )
    val_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=2
    )
    return val_loader
