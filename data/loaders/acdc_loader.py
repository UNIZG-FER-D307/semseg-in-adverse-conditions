from torchvision import transforms as tf
import copy
from torch.utils.data import DataLoader, ConcatDataset, Subset
from data.datasets import (
    ACDCFull,
    ACDC,
    ACDCFullTest,
    ACDCWithInvalid,
    UniformClassDataset,
    acdc_fog_std,
    acdc_fog_mean,
    acdc_night_std,
    acdc_night_mean,
    acdc_snow_std,
    acdc_snow_mean,
    acdc_rain_mean,
    acdc_rain_std,
    ACDCTest,
)
from data.transforms import JitterRandomCrop
import torch


def create_per_split_val_transform(transforms, mean, std):
    transforms["image"] = transforms["image"] + [tf.Normalize(mean, std)]
    return transforms


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


def lm(x):
    return (x * 255.0).long()


def load_acdc_full_train(dataroot, bs, train_transforms, uniform_class=False):
    remap_labels = tf.Lambda(lm)
    train_set = ACDCFull(
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
    print(f"> Loaded {len(train_set)} train images.")
    train_set = (
        UniformClassDataset(
            train_set,
            class_uniform_pct=0.5,
            class_uniform_tile=1024,
            id_to_trainid=None,
        )
        if uniform_class
        else train_set
    )
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, pin_memory=False, num_workers=3
    )
    return train_loader


def load_acdc_full_train_val(dataroot, bs, train_transforms, uniform_class=False):
    remap_labels = tf.Lambda(lm)
    train_set = ACDCFull(
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
    val_set = ACDCFull(
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
    train_set = (
        UniformClassDataset(
            train_set,
            class_uniform_pct=0.5,
            class_uniform_tile=1024,
            id_to_trainid=None,
        )
        if uniform_class
        else train_set
    )
    val_set = (
        UniformClassDataset(
            val_set, class_uniform_pct=0.5, class_uniform_tile=1024, id_to_trainid=None
        )
        if uniform_class
        else val_set
    )

    data = ConcatDataset([train_set, val_set])
    print(f"> Loaded {len(data)} train images.")
    train_loader = DataLoader(
        data, batch_size=bs, shuffle=True, pin_memory=True, num_workers=5
    )
    return train_loader


def load_acdc_specific_train(dataroot, tag, bs, train_transforms):
    remap_labels = tf.Lambda(lm)
    train_set = ACDC(
        dataroot,
        tag=tag,
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
    print(f"> Loaded {len(train_set)} train images.")
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, pin_memory=False, num_workers=3
    )
    return train_loader


def load_acdc_full_val(dataroot, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_set = ACDCFull(
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
        val_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=2
    )
    return val_loader


def load_acdc_specific_val(dataroot, tag, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_set = ACDC(
        dataroot,
        tag=tag,
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
        val_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=2
    )
    return val_loader


def load_acdc_full_val_with_per_split_means(dataroot, val_transforms):
    remap_labels = tf.Lambda(lm)
    trans_snow = create_per_split_val_transform(
        copy.deepcopy(val_transforms), acdc_snow_mean, acdc_snow_std
    )
    snow_set = ACDC(
        dataroot,
        tag="snow",
        split="val",
        image_transform=None
        if not trans_snow["image"]
        else tf.Compose(trans_snow["image"]),
        target_transform=None
        if not trans_snow["target"]
        else tf.Compose(trans_snow["target"] + [remap_labels]),
        joint_transform=None
        if not trans_snow["joint"]
        else tf.Compose(trans_snow["joint"]),
    )
    trans_rain = create_per_split_val_transform(
        copy.deepcopy(val_transforms), acdc_rain_mean, acdc_rain_std
    )
    rain_set = ACDC(
        dataroot,
        tag="rain",
        split="val",
        image_transform=None
        if not trans_rain["image"]
        else tf.Compose(trans_rain["image"]),
        target_transform=None
        if not trans_rain["target"]
        else tf.Compose(trans_rain["target"] + [remap_labels]),
        joint_transform=None
        if not trans_rain["joint"]
        else tf.Compose(trans_rain["joint"]),
    )
    trans_fog = create_per_split_val_transform(
        copy.deepcopy(val_transforms), acdc_fog_mean, acdc_fog_std
    )
    fog_set = ACDC(
        dataroot,
        tag="fog",
        split="val",
        image_transform=None
        if not trans_fog["image"]
        else tf.Compose(trans_fog["image"]),
        target_transform=None
        if not trans_fog["target"]
        else tf.Compose(trans_fog["target"] + [remap_labels]),
        joint_transform=None
        if not trans_fog["joint"]
        else tf.Compose(trans_fog["joint"]),
    )
    trans_dark = create_per_split_val_transform(
        copy.deepcopy(val_transforms), acdc_night_mean, acdc_night_std
    )
    dark_set = ACDC(
        dataroot,
        tag="night",
        split="val",
        image_transform=None
        if not trans_dark["image"]
        else tf.Compose(trans_dark["image"]),
        target_transform=None
        if not trans_dark["target"]
        else tf.Compose(trans_dark["target"] + [remap_labels]),
        joint_transform=None
        if not trans_dark["joint"]
        else tf.Compose(trans_dark["joint"]),
    )
    val_set = ConcatDataset([rain_set, fog_set, dark_set, snow_set])
    print(f"> Loaded {len(val_set)} val images.")
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=3
    )
    return val_loader


def load_acdc_calibration_val_with_per_split_means(
    dataroot, val_transforms, bs=1, split="val", calib_percent=0.2
):
    remap_labels = tf.Lambda(lm)
    trans_snow = create_per_split_val_transform(
        copy.deepcopy(val_transforms), acdc_snow_mean, acdc_snow_std
    )
    snow_set = ACDC(
        dataroot,
        tag="snow",
        split=split,
        image_transform=None
        if not trans_snow["image"]
        else tf.Compose(trans_snow["image"]),
        target_transform=None
        if not trans_snow["target"]
        else tf.Compose(trans_snow["target"] + [remap_labels]),
        joint_transform=None
        if not trans_snow["joint"]
        else tf.Compose(trans_snow["joint"]),
    )
    trans_rain = create_per_split_val_transform(
        copy.deepcopy(val_transforms), acdc_rain_mean, acdc_rain_std
    )
    rain_set = ACDC(
        dataroot,
        tag="rain",
        split=split,
        image_transform=None
        if not trans_rain["image"]
        else tf.Compose(trans_rain["image"]),
        target_transform=None
        if not trans_rain["target"]
        else tf.Compose(trans_rain["target"] + [remap_labels]),
        joint_transform=None
        if not trans_rain["joint"]
        else tf.Compose(trans_rain["joint"]),
    )
    trans_fog = create_per_split_val_transform(
        copy.deepcopy(val_transforms), acdc_fog_mean, acdc_fog_std
    )
    fog_set = ACDC(
        dataroot,
        tag="fog",
        split=split,
        image_transform=None
        if not trans_fog["image"]
        else tf.Compose(trans_fog["image"]),
        target_transform=None
        if not trans_fog["target"]
        else tf.Compose(trans_fog["target"] + [remap_labels]),
        joint_transform=None
        if not trans_fog["joint"]
        else tf.Compose(trans_fog["joint"]),
    )
    trans_dark = create_per_split_val_transform(
        copy.deepcopy(val_transforms), acdc_night_mean, acdc_night_std
    )
    dark_set = ACDC(
        dataroot,
        tag="night",
        split=split,
        image_transform=None
        if not trans_dark["image"]
        else tf.Compose(trans_dark["image"]),
        target_transform=None
        if not trans_dark["target"]
        else tf.Compose(trans_dark["target"] + [remap_labels]),
        joint_transform=None
        if not trans_dark["joint"]
        else tf.Compose(trans_dark["joint"]),
    )

    calib_ds, rest_ds = [], []
    for set in [rain_set, fog_set, dark_set, snow_set]:
        d = len(set)
        indices = torch.randperm(d)
        calib, rest = (
            indices[: int(d * calib_percent)],
            indices[int(d * calib_percent) :],
        )
        calib_ds.append(Subset(set, calib))
        rest_ds.append(Subset(set, rest))
    calib_val_set = ConcatDataset(calib_ds)
    rest_val_set = ConcatDataset(rest_ds)
    print(f"> Loaded {len(rest_val_set)} val images.")
    print(f"> Loaded {len(calib_val_set)} calibration images.")
    calib_val_loader = DataLoader(
        calib_val_set, batch_size=bs, shuffle=True, pin_memory=False, num_workers=3
    )
    rest_val_loader = DataLoader(
        rest_val_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=3
    )
    return calib_val_loader, rest_val_loader


def load_acdc_full_train_with_per_split_means(
    dataroot, bs, train_transforms, config, uniform_class=False, dark_oversample_coef=0
):
    remap_labels = tf.Lambda(lm)
    trans_snow = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_snow_mean,
        acdc_snow_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    snow_set = ACDC(
        dataroot,
        tag="snow",
        split="train",
        image_transform=None
        if not trans_snow["image"]
        else tf.Compose(trans_snow["image"]),
        target_transform=None
        if not trans_snow["target"]
        else tf.Compose(trans_snow["target"] + [remap_labels]),
        joint_transform=None
        if not trans_snow["joint"]
        else tf.Compose(trans_snow["joint"]),
    )

    trans_rain = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_rain_mean,
        acdc_rain_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    rain_set = ACDC(
        dataroot,
        tag="rain",
        split="train",
        image_transform=None
        if not trans_rain["image"]
        else tf.Compose(trans_rain["image"]),
        target_transform=None
        if not trans_rain["target"]
        else tf.Compose(trans_rain["target"] + [remap_labels]),
        joint_transform=None
        if not trans_rain["joint"]
        else tf.Compose(trans_rain["joint"]),
    )

    trans_fog = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_fog_mean,
        acdc_fog_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    fog_set = ACDC(
        dataroot,
        tag="fog",
        split="train",
        image_transform=None
        if not trans_fog["image"]
        else tf.Compose(trans_fog["image"]),
        target_transform=None
        if not trans_fog["target"]
        else tf.Compose(trans_fog["target"] + [remap_labels]),
        joint_transform=None
        if not trans_fog["joint"]
        else tf.Compose(trans_fog["joint"]),
    )

    trans_dark = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_night_mean,
        acdc_night_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dark_set = ACDC(
        dataroot,
        tag="night",
        split="train",
        image_transform=None
        if not trans_dark["image"]
        else tf.Compose(trans_dark["image"]),
        target_transform=None
        if not trans_dark["target"]
        else tf.Compose(trans_dark["target"] + [remap_labels]),
        joint_transform=None
        if not trans_dark["joint"]
        else tf.Compose(trans_dark["joint"]),
    )

    snow_set = (
        UniformClassDataset(snow_set, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else snow_set
    )
    rain_set = (
        UniformClassDataset(rain_set, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else rain_set
    )
    fog_set = (
        UniformClassDataset(fog_set, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else fog_set
    )
    dark_set = (
        UniformClassDataset(dark_set, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else dark_set
    )
    train_set = ConcatDataset(
        [rain_set, fog_set, dark_set, snow_set] + [dark_set] * dark_oversample_coef
    )
    print(f"> Loaded {len(train_set)} train images.")
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=3
    )
    return train_loader


def load_acdc_full_test(dataroot, val_transforms, split="test"):
    val_set = ACDCFullTest(
        dataroot,
        split=split,
        image_transform=None
        if not val_transforms["image"]
        else tf.Compose(val_transforms["image"]),
    )
    print(f"> Loaded {len(val_set)} test images.")
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=3
    )
    return val_loader


def load_acdc_full_test_with_per_split_means(dataroot, val_transforms, split="test"):
    trans_snow = create_per_split_val_transform(
        copy.deepcopy(val_transforms), acdc_snow_mean, acdc_snow_std
    )
    snow_set = ACDCTest(
        dataroot,
        tag="snow",
        split=split,
        image_transform=None
        if not trans_snow["image"]
        else tf.Compose(trans_snow["image"]),
    )
    trans_rain = create_per_split_val_transform(
        copy.deepcopy(val_transforms), acdc_rain_mean, acdc_rain_std
    )
    rain_set = ACDCTest(
        dataroot,
        tag="rain",
        split=split,
        image_transform=None
        if not trans_rain["image"]
        else tf.Compose(trans_rain["image"]),
    )
    trans_fog = create_per_split_val_transform(
        copy.deepcopy(val_transforms), acdc_fog_mean, acdc_fog_std
    )
    fog_set = ACDCTest(
        dataroot,
        tag="fog",
        split=split,
        image_transform=None
        if not trans_fog["image"]
        else tf.Compose(trans_fog["image"]),
    )
    trans_dark = create_per_split_val_transform(
        copy.deepcopy(val_transforms), acdc_night_mean, acdc_night_std
    )
    dark_set = ACDCTest(
        dataroot,
        tag="night",
        split=split,
        image_transform=None
        if not trans_dark["image"]
        else tf.Compose(trans_dark["image"]),
    )
    val_set = ConcatDataset([rain_set, fog_set, dark_set, snow_set])
    print(f"> Loaded {len(val_set)} val images.")
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=3
    )
    return val_loader


def load_acdc_full_trainval_with_per_split_means(
    dataroot, bs, train_transforms, config, uniform_class=False
):
    remap_labels = tf.Lambda(lm)
    trans_snow = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_snow_mean,
        acdc_snow_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    snow_set = ACDC(
        dataroot,
        tag="snow",
        split="train",
        image_transform=None
        if not trans_snow["image"]
        else tf.Compose(trans_snow["image"]),
        target_transform=None
        if not trans_snow["target"]
        else tf.Compose(trans_snow["target"] + [remap_labels]),
        joint_transform=None
        if not trans_snow["joint"]
        else tf.Compose(trans_snow["joint"]),
    )
    snow_set_val = ACDC(
        dataroot,
        tag="snow",
        split="val",
        image_transform=None
        if not trans_snow["image"]
        else tf.Compose(trans_snow["image"]),
        target_transform=None
        if not trans_snow["target"]
        else tf.Compose(trans_snow["target"] + [remap_labels]),
        joint_transform=None
        if not trans_snow["joint"]
        else tf.Compose(trans_snow["joint"]),
    )

    trans_rain = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_rain_mean,
        acdc_rain_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    rain_set = ACDC(
        dataroot,
        tag="rain",
        split="train",
        image_transform=None
        if not trans_rain["image"]
        else tf.Compose(trans_rain["image"]),
        target_transform=None
        if not trans_rain["target"]
        else tf.Compose(trans_rain["target"] + [remap_labels]),
        joint_transform=None
        if not trans_rain["joint"]
        else tf.Compose(trans_rain["joint"]),
    )
    rain_set_val = ACDC(
        dataroot,
        tag="rain",
        split="val",
        image_transform=None
        if not trans_rain["image"]
        else tf.Compose(trans_rain["image"]),
        target_transform=None
        if not trans_rain["target"]
        else tf.Compose(trans_rain["target"] + [remap_labels]),
        joint_transform=None
        if not trans_rain["joint"]
        else tf.Compose(trans_rain["joint"]),
    )

    trans_fog = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_fog_mean,
        acdc_fog_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    fog_set = ACDC(
        dataroot,
        tag="fog",
        split="train",
        image_transform=None
        if not trans_fog["image"]
        else tf.Compose(trans_fog["image"]),
        target_transform=None
        if not trans_fog["target"]
        else tf.Compose(trans_fog["target"] + [remap_labels]),
        joint_transform=None
        if not trans_fog["joint"]
        else tf.Compose(trans_fog["joint"]),
    )
    fog_set_val = ACDC(
        dataroot,
        tag="fog",
        split="val",
        image_transform=None
        if not trans_fog["image"]
        else tf.Compose(trans_fog["image"]),
        target_transform=None
        if not trans_fog["target"]
        else tf.Compose(trans_fog["target"] + [remap_labels]),
        joint_transform=None
        if not trans_fog["joint"]
        else tf.Compose(trans_fog["joint"]),
    )

    trans_dark = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_night_mean,
        acdc_night_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dark_set = ACDC(
        dataroot,
        tag="night",
        split="train",
        image_transform=None
        if not trans_dark["image"]
        else tf.Compose(trans_dark["image"]),
        target_transform=None
        if not trans_dark["target"]
        else tf.Compose(trans_dark["target"] + [remap_labels]),
        joint_transform=None
        if not trans_dark["joint"]
        else tf.Compose(trans_dark["joint"]),
    )
    dark_set_val = ACDC(
        dataroot,
        tag="night",
        split="val",
        image_transform=None
        if not trans_dark["image"]
        else tf.Compose(trans_dark["image"]),
        target_transform=None
        if not trans_dark["target"]
        else tf.Compose(trans_dark["target"] + [remap_labels]),
        joint_transform=None
        if not trans_dark["joint"]
        else tf.Compose(trans_dark["joint"]),
    )

    snow_set = (
        UniformClassDataset(snow_set, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else snow_set
    )
    snow_set_val = (
        UniformClassDataset(
            snow_set_val, class_uniform_pct=0.75, class_uniform_tile=512
        )
        if uniform_class
        else snow_set_val
    )
    rain_set = (
        UniformClassDataset(rain_set, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else rain_set
    )
    rain_set_val = (
        UniformClassDataset(
            rain_set_val, class_uniform_pct=0.75, class_uniform_tile=512
        )
        if uniform_class
        else rain_set_val
    )
    fog_set = (
        UniformClassDataset(fog_set, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else fog_set
    )
    fog_set_val = (
        UniformClassDataset(fog_set_val, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else fog_set_val
    )
    dark_set = (
        UniformClassDataset(dark_set, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else dark_set
    )
    dark_set_val = (
        UniformClassDataset(
            dark_set_val, class_uniform_pct=0.75, class_uniform_tile=512
        )
        if uniform_class
        else dark_set_val
    )

    train_set = ConcatDataset(
        [
            rain_set,
            rain_set_val,
            fog_set,
            fog_set_val,
            dark_set,
            dark_set_val,
            snow_set,
            snow_set_val,
        ]
    )
    print(f"> Loaded {len(train_set)} train images.")
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=3
    )
    return train_loader


def load_acdc_full_train_with_per_split_means_and_invalid_regions(
    dataroot, bs, train_transforms, config, uniform_class=False, dark_oversample_coef=0
):
    remap_labels = tf.Lambda(lm)
    trans_snow = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_snow_mean,
        acdc_snow_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    snow_set = ACDCWithInvalid(
        dataroot,
        tag="snow",
        split="train",
        image_transform=None
        if not trans_snow["image"]
        else tf.Compose(trans_snow["image"]),
        target_transform=None
        if not trans_snow["target"]
        else tf.Compose(trans_snow["target"] + [remap_labels]),
        joint_transform=None
        if not trans_snow["joint"]
        else tf.Compose(trans_snow["joint"]),
    )

    trans_rain = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_rain_mean,
        acdc_rain_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    rain_set = ACDCWithInvalid(
        dataroot,
        tag="rain",
        split="train",
        image_transform=None
        if not trans_rain["image"]
        else tf.Compose(trans_rain["image"]),
        target_transform=None
        if not trans_rain["target"]
        else tf.Compose(trans_rain["target"] + [remap_labels]),
        joint_transform=None
        if not trans_rain["joint"]
        else tf.Compose(trans_rain["joint"]),
    )

    trans_fog = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_fog_mean,
        acdc_fog_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    fog_set = ACDCWithInvalid(
        dataroot,
        tag="fog",
        split="train",
        image_transform=None
        if not trans_fog["image"]
        else tf.Compose(trans_fog["image"]),
        target_transform=None
        if not trans_fog["target"]
        else tf.Compose(trans_fog["target"] + [remap_labels]),
        joint_transform=None
        if not trans_fog["joint"]
        else tf.Compose(trans_fog["joint"]),
    )

    trans_dark = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_night_mean,
        acdc_night_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dark_set = ACDCWithInvalid(
        dataroot,
        tag="night",
        split="train",
        image_transform=None
        if not trans_dark["image"]
        else tf.Compose(trans_dark["image"]),
        target_transform=None
        if not trans_dark["target"]
        else tf.Compose(trans_dark["target"] + [remap_labels]),
        joint_transform=None
        if not trans_dark["joint"]
        else tf.Compose(trans_dark["joint"]),
    )

    snow_set = (
        UniformClassDataset(snow_set, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else snow_set
    )
    rain_set = (
        UniformClassDataset(rain_set, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else rain_set
    )
    fog_set = (
        UniformClassDataset(fog_set, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else fog_set
    )
    dark_set = (
        UniformClassDataset(dark_set, class_uniform_pct=0.75, class_uniform_tile=512)
        if uniform_class
        else dark_set
    )
    train_set = ConcatDataset(
        [rain_set, fog_set, dark_set, snow_set] + [dark_set] * dark_oversample_coef
    )
    print(f"> Loaded {len(train_set)} train images.")
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=3
    )

    return train_loader
