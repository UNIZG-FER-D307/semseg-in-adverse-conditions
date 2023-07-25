from torchvision import transforms as tf
import copy
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from data.datasets import (
    DarkZurichUnsupervised,
    BDDUnsupervised,
    bdd_std,
    bdd_mean,
    dark_zurich_night_std,
    dark_zurich_night_mean,
    dark_zurich_day_mean,
    dark_zurich_twilight_mean,
    dark_zurich_twilight_std,
    dark_zurich_day_std,
    NightOwlsUnsupervised,
    nightowls_std,
    nightowls_mean,
    NightOwlsPseudolabeled,
    night_city_mean,
    night_city_std,
    ACDCUnsupervised,
    acdc_night_mean,
    acdc_night_std,
    acdc_fog_mean,
    acdc_fog_std,
    acdc_rain_mean,
    acdc_rain_std,
    acdc_snow_mean,
    acdc_snow_std,
)
from data.transforms.joint_transforms.image_transforms import ImageJitterRandomCrop
from data.transforms import JitterRandomCrop
from data.datasets import (
    BDDPseudolabeled,
    DarkZurichPseudolabeled,
    UniformClassDataset,
    NightCityUnsupervised,
    BDDConditionwiseUnsupervised,
    CADCDUnsupervised,
    CityscapesUnsupervised,
)
from data.datasets import (
    bdd_rain_mean,
    bdd_fog_mean,
    bdd_snow_mean,
    bdd_rain_std,
    bdd_snow_std,
    bdd_fog_std,
    cadcd_mean,
    cadcd_std,
    cityscapes_mean,
    cityscapes_std,
)
from data.datasets import STFUnsupervised, stf_mean, stf_std


def create_per_dataset_transform(
    transforms, mean, std, scale, ignore_id, crop_size=1024
):
    transforms["image"] = transforms["image"] + [
        tf.Normalize(mean, std),
        # tf.Lambda(rescale_if_needed),
        # tf.RandomCrop(crop_size)
        ImageJitterRandomCrop(crop_size, scale, ignore_id=ignore_id, input_mean=(0.0)),
    ]
    return transforms


def prepare_unsupervised_datasets(dataroots, train_transforms, config):
    bdd_trans = create_per_dataset_transform(
        copy.deepcopy(train_transforms),
        bdd_mean,
        bdd_std,
        crop_size=config["unsupervised_crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    bdd_train = BDDUnsupervised(
        dataroots["bdd100k"],
        split="train",
        image_transform=tf.Compose(bdd_trans["image"]),
    )
    bdd_val = BDDUnsupervised(
        dataroots["bdd100k"],
        split="val",
        image_transform=tf.Compose(bdd_trans["image"]),
    )
    bdd_test = BDDUnsupervised(
        dataroots["bdd100k"],
        split="test",
        image_transform=tf.Compose(bdd_trans["image"]),
    )

    dz_day_trans = create_per_dataset_transform(
        copy.deepcopy(train_transforms),
        dark_zurich_day_mean,
        dark_zurich_day_std,
        crop_size=config["unsupervised_crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dz_day_train = DarkZurichUnsupervised(
        dataroots["dark_zurich"],
        split="train",
        condition="day",
        image_transform=tf.Compose(dz_day_trans["image"]),
    )

    dz_twilight_trans = create_per_dataset_transform(
        copy.deepcopy(train_transforms),
        dark_zurich_twilight_mean,
        dark_zurich_twilight_std,
        crop_size=config["unsupervised_crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dz_twilight_train = DarkZurichUnsupervised(
        dataroots["dark_zurich"],
        split="train",
        condition="twilight",
        image_transform=tf.Compose(dz_twilight_trans["image"]),
    )

    dz_night_trans = create_per_dataset_transform(
        copy.deepcopy(train_transforms),
        dark_zurich_night_mean,
        dark_zurich_night_std,
        crop_size=config["unsupervised_crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dz_night_train = DarkZurichUnsupervised(
        dataroots["dark_zurich"],
        split="train",
        condition="night",
        image_transform=tf.Compose(dz_night_trans["image"]),
    )
    dz_night_val = DarkZurichUnsupervised(
        dataroots["dark_zurich"],
        split="val",
        condition="night",
        image_transform=tf.Compose(dz_night_trans["image"]),
    )

    return [
        bdd_val,
        bdd_train,
        bdd_test,
        dz_day_train,
        dz_twilight_train,
        dz_night_train,
        dz_night_val,
    ]


def load_unsupervised_train(dataroots, bs, train_transforms, config):
    datasets = prepare_unsupervised_datasets(dataroots, train_transforms, config)
    ds = ConcatDataset(datasets)
    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(
        ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=3
    )
    return train_loader


def prepare_datasets_for_pseudolabeling(dataroots, train_transforms):
    dz_day_train = DarkZurichUnsupervised(
        dataroots["dark_zurich"],
        split="train",
        condition="day",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"]
            + [tf.Normalize(dark_zurich_day_mean, dark_zurich_day_std)]
        ),
    )

    """
    dz_twilight_train = DarkZurichUnsupervised(
        dataroots["dark_zurich"],
        split="train",
        condition="twilight",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"]
            + [tf.Normalize(dark_zurich_twilight_mean, dark_zurich_twilight_std)]
        ),
    )

    dz_night_train = DarkZurichUnsupervised(
        dataroots["dark_zurich"],
        split="train",
        condition="night",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"]
            + [tf.Normalize(dark_zurich_night_mean, dark_zurich_night_std)]
        ),
    )

    dz_night_val = DarkZurichUnsupervised(
        dataroots["dark_zurich"],
        split="val",
        condition="night",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"]
            + [tf.Normalize(dark_zurich_night_mean, dark_zurich_night_std)]
        ),
    )

    bdd_train = BDDUnsupervised(
        dataroots["bdd100k"],
        split="train",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"] + [tf.Normalize(bdd_mean, bdd_std)]
        ),
    )
    bdd_val = BDDUnsupervised(
        dataroots["bdd100k"],
        split="val",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"] + [tf.Normalize(bdd_mean, bdd_std)]
        ),
    )
    bdd_test = BDDUnsupervised(
        dataroots["bdd100k"],
        split="test",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"] + [tf.Normalize(bdd_mean, bdd_std)]
        ),
    )
    dz_night_val = DarkZurichUnsupervised(
        dataroots["dark_zurich"],
        split="val",
        condition="night",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"]
            + [tf.Normalize(dark_zurich_night_mean, dark_zurich_night_std)]
        ),
    )

    nightowls_train = NightOwlsUnsupervised(
        dataroots["night_owls"],
        split="validation",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"] + [tf.Normalize(nightowls_mean, nightowls_std)]
        ),
    )
    nightcity_train = NightCityUnsupervised(
        dataroots["night_city"],
        split="val",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"] + [tf.Normalize(night_city_mean, night_city_std)]
        ),
    )

    bdd100k_rainy_train = BDDConditionwiseUnsupervised(
        dataroots["bdd100k"],
        split="val",
        condition="rainy",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"] + [tf.Normalize(bdd_rain_mean, bdd_rain_std)]
        ),
    )
    bdd100k_foggy_train = BDDConditionwiseUnsupervised(
        dataroots["bdd100k"],
        split="train",
        condition="foggy",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"] + [tf.Normalize(bdd_fog_mean, bdd_fog_std)]
        ),
    )
    bdd100k_snowy_train = BDDConditionwiseUnsupervised(
        dataroots["bdd100k"],
        split="val",
        condition="snowy",
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"] + [tf.Normalize(bdd_snow_mean, bdd_snow_std)]
        ),
    )
    cadcd_all = CADCDUnsupervised(
        dataroots["cadcd"],
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"] + [tf.Normalize(cadcd_mean, cadcd_std)]
        ),
    )
    stf_all = STFUnsupervised(
        dataroots["stf"],
        return_name=True,
        image_transform=tf.Compose(
            train_transforms["image"] + [tf.Normalize(stf_mean, stf_std)]
        ),
    )
    """

    return [dz_day_train]


def load_datasets_for_pseudolabeling(dataroots, train_transforms):
    datasets = prepare_datasets_for_pseudolabeling(dataroots, train_transforms)
    ds = ConcatDataset(datasets)
    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(
        ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=3
    )
    return train_loader


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


def prepare_pseudolabeled_datasets(dataroots, train_transforms, config):
    remap_labels_bdd = tf.Lambda(lm)
    bdd_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        bdd_mean,
        bdd_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    bdd_train = BDDPseudolabeled(
        dataroots["bdd100k"],
        split="train",
        image_transform=tf.Compose(bdd_trans["image"]),
        target_transform=tf.Compose(bdd_trans["target"] + [remap_labels_bdd]),
        joint_transform=tf.Compose(bdd_trans["joint"]),
    )
    bdd_val = BDDPseudolabeled(
        dataroots["bdd100k"],
        split="val",
        image_transform=tf.Compose(bdd_trans["image"]),
        target_transform=tf.Compose(bdd_trans["target"] + [remap_labels_bdd]),
        joint_transform=tf.Compose(bdd_trans["joint"]),
    )
    bdd_test = BDDPseudolabeled(
        dataroots["bdd100k"],
        split="test",
        image_transform=tf.Compose(bdd_trans["image"]),
        target_transform=tf.Compose(bdd_trans["target"] + [remap_labels_bdd]),
        joint_transform=tf.Compose(bdd_trans["joint"]),
    )

    remap_labels_dz = tf.Lambda(lm)
    dz_day_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        dark_zurich_day_mean,
        dark_zurich_day_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dz_day_train = DarkZurichPseudolabeled(
        dataroots["dark_zurich"],
        split="train",
        condition="day",
        image_transform=tf.Compose(dz_day_trans["image"]),
        target_transform=tf.Compose(dz_day_trans["target"] + [remap_labels_dz]),
        joint_transform=tf.Compose(dz_day_trans["joint"]),
    )

    dz_twilight_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        dark_zurich_twilight_mean,
        dark_zurich_twilight_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dz_twilight_train = DarkZurichPseudolabeled(
        dataroots["dark_zurich"],
        split="train",
        condition="twilight",
        image_transform=tf.Compose(dz_twilight_trans["image"]),
        target_transform=tf.Compose(dz_twilight_trans["target"] + [remap_labels_dz]),
        joint_transform=tf.Compose(dz_twilight_trans["joint"]),
    )

    dz_night_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        dark_zurich_night_mean,
        dark_zurich_night_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dz_night_train = DarkZurichPseudolabeled(
        dataroots["dark_zurich"],
        split="train",
        condition="night",
        image_transform=tf.Compose(dz_night_trans["image"]),
        target_transform=tf.Compose(dz_night_trans["target"] + [remap_labels_dz]),
        joint_transform=tf.Compose(dz_night_trans["joint"]),
    )
    dz_night_val = DarkZurichPseudolabeled(
        dataroots["dark_zurich"],
        split="val",
        condition="night",
        image_transform=tf.Compose(dz_night_trans["image"]),
        target_transform=tf.Compose(dz_night_trans["target"] + [remap_labels_dz]),
        joint_transform=tf.Compose(dz_night_trans["joint"]),
    )

    remap_labels_no = tf.Lambda(lm)
    nightowls_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        nightowls_mean,
        nightowls_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    nightowls_train = NightOwlsPseudolabeled(
        dataroots["night_owls"],
        split="training",
        image_transform=tf.Compose(nightowls_trans["image"]),
        target_transform=tf.Compose(nightowls_trans["target"] + [remap_labels_no]),
        joint_transform=tf.Compose(nightowls_trans["joint"]),
    )

    return [
        bdd_val,
        bdd_train,
        bdd_test,
        dz_day_train,
        dz_twilight_train,
        dz_night_train,
        dz_night_val,
        nightowls_train,
    ]


def load_pseudolabeled_datasets(dataroots, bs, train_transforms, config):
    datasets = prepare_pseudolabeled_datasets(dataroots, train_transforms, config)
    if config["pseudo_uniform"]:
        datasets = [
            UniformClassDataset(ds, class_uniform_pct=0.75, class_uniform_tile=512)
            for ds in datasets
        ]
        print("> Created uniform dataset")
    ds = ConcatDataset(datasets)
    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(
        ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=3
    )
    return train_loader
