from torchvision import transforms as tf
import copy
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from data.datasets import RainyCityscapes, ACDC
from data.datasets import rainy_city_mean, rainy_city_std, acdc_rain_mean, acdc_rain_std
from data.transforms import JitterRandomCrop
from data.datasets import (
    BDDPseudolabeledConditionwise,
    bdd_rain_std,
    bdd_rain_mean,
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


def prepare_rainy_datasets(dataroots, train_transforms, config):
    remap_labels = tf.Lambda(lm)
    cr_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        rainy_city_mean,
        rainy_city_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    cr = RainyCityscapes(
        dataroots["rainy_city"],
        split="train",
        image_transform=tf.Compose(cr_trans["image"]),
        target_transform=tf.Compose(cr_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(cr_trans["joint"]),
    )

    acdc_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_rain_mean,
        acdc_rain_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    acdc_rain = ACDC(
        dataroots["acdc"],
        split="train",
        tag="rain",
        image_transform=tf.Compose(acdc_trans["image"]),
        target_transform=tf.Compose(acdc_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(acdc_trans["joint"]),
    )

    bdd_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        bdd_rain_mean,
        bdd_rain_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    bdd_rain = BDDPseudolabeledConditionwise(
        dataroots["bdd100k"],
        split="train",
        condition="rainy",
        image_transform=tf.Compose(bdd_trans["image"]),
        target_transform=tf.Compose(bdd_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(bdd_trans["joint"]),
    )
    datasets = [cr, acdc_rain, bdd_rain]
    # datasets = [acdc_rain]
    return datasets


def load_rain_finetune(dataroots, bs, train_transforms, config):
    ds_rain = prepare_rainy_datasets(dataroots, train_transforms, config)

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


def load_rainy_train(dataroots, bs, train_transforms, config):
    datasets = prepare_rainy_datasets(dataroots, train_transforms, config)
    ds = ConcatDataset(datasets)
    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(
        ds, batch_size=bs, shuffle=True, pin_memory=False, num_workers=3
    )
    return train_loader


def load_rainy_acdc_val(dataroot, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_set = ACDC(
        dataroot,
        split="val",
        tag="rain",
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


def load_rain_eval(dataroots, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_transforms["image"] = val_transforms["image"] + [
        tf.Normalize(acdc_rain_mean, acdc_rain_std)
    ]
    acdc_trans = val_transforms
    dataset = ACDC(
        dataroots["acdc"],
        split="val",
        tag="rain",
        image_transform=tf.Compose(acdc_trans["image"]),
        target_transform=tf.Compose(acdc_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(acdc_trans["joint"]),
    )
    val_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=2
    )
    return val_loader
