import copy

from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms as tf

from data.datasets import (
    ACDC,
    Cityscapes,
    DarkZurich,
    UniformClassDataset,
    DarkZurichPseudolabeled,
    ACDCPseudolabeled,
    Wilddash2,
    acdc_fog_mean,
    acdc_fog_std,
    acdc_night_mean,
    acdc_night_std,
    acdc_rain_mean,
    acdc_rain_std,
    acdc_snow_mean,
    acdc_snow_std,
    cityscapes_mean,
    cityscapes_std,
    create_wilddash_to_cityscapes_mapper,
    dark_zurich_mean,
    dark_zurich_std,
    wilddash2_mean,
    wilddash2_std,
    dark_zurich_night_mean,
    dark_zurich_night_std,
)

from data.datasets.cityscapes.cityscapes_labels import create_id_to_train_id_mapper
from data.transforms import JitterRandomCrop
from torch.utils.data import ConcatDataset

mapper_wd_city = create_wilddash_to_cityscapes_mapper()
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


def lm_city(x):
    return label_mapper[(x * 255.0).long()]


def prepare_city_datasets(dataroots, train_transforms, config):
    remap_labels = tf.Lambda(lm_city)
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
    return [train_set]


def lm_wd(x):
    return mapper_wd_city[(x * 255.0).long()]


def lm(x):
    return (x * 255.0).long()


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


def prepare_architecture_sweep_datasets(dataroots, train_transforms, config):
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

    remap_labels = tf.Lambda(lm)

    print(f"> Loaded {len(dz)} dark zurich images.")
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
        dataroots["acdc"],
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

    print(f"> Loaded {len(snow_set)} snow acdc images.")
    remap_labels = tf.Lambda(lm)
    trans_rain = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_rain_mean,
        acdc_rain_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    rain_set = ACDC(
        dataroots["acdc"],
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

    print(f"> Loaded {len(rain_set)} rainy acdc images.")
    remap_labels = tf.Lambda(lm)
    trans_fog = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_fog_mean,
        acdc_fog_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    fog_set = ACDC(
        dataroots["acdc"],
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

    print(f"> Loaded {len(fog_set)} foggy acdc images.")
    remap_labels = tf.Lambda(lm)
    trans_dark = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_night_mean,
        acdc_night_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dark_set = ACDC(
        dataroots["acdc"],
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

    print(f"> Loaded {len(dark_set)} dark acdc images.")

    train_city = prepare_city_datasets(dataroots, train_transforms, config)[0]

    print(f"> Loaded {len(train_city)} city train images.")

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

    print(f"> Loaded {len(wd)} wilddash images.")

    mix_datasets = [
        train_city,
        dz,
        rain_set,
        fog_set,
        dark_set,
        snow_set,
        wd,
    ]
    if config["uniform_class"]:
        mix_datasets = [
            UniformClassDataset(ds, class_uniform_pct=0.75, class_uniform_tile=512)
            for ds in mix_datasets
        ]
        print("> Created uniform dataset")

    return mix_datasets


def prepare_architecure_sweep_city(dataroots, train_transforms, config):
    train_city = prepare_city_datasets(dataroots, train_transforms, config)[0]

    print(f"> Loaded {len(train_city)} city train images.")

    if config["uniform_class"]:
        train_city = UniformClassDataset(
            train_city, class_uniform_pct=0.75, class_uniform_tile=512
        )

    return train_city


def load_architecture_sweep_city(dataroots, bs, train_transforms, config):
    dataset = prepare_architecure_sweep_city(dataroots, train_transforms, config)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )

    return train_loader


def prepare_architecture_sweep_datasets_wo_acdc(dataroots, train_transforms, config):
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

    print(f"> Loaded {len(dz)} dark zurich images.")

    train_city = prepare_city_datasets(dataroots, train_transforms, config)[0]

    print(f"> Loaded {len(train_city)} city train images.")

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

    print(f"> Loaded {len(wd)} wilddash images.")

    mix_datasets = [
        train_city,
        dz,
        wd,
    ]
    if config["uniform_class"]:
        mix_datasets = [
            UniformClassDataset(ds, class_uniform_pct=0.75, class_uniform_tile=512)
            for ds in mix_datasets
        ]
        print("> Created uniform dataset")

    return mix_datasets


def prepare_architecture_sweep_ps_night_datasets(dataroots, train_transforms, config):
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

    remap_labels = tf.Lambda(lm)
    dz_night_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        dark_zurich_night_mean,
        dark_zurich_night_std,
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

    print(f"> Loaded {len(dz)} dark zurich images.")
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
        dataroots["acdc"],
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

    print(f"> Loaded {len(snow_set)} snow acdc images.")
    remap_labels = tf.Lambda(lm)
    trans_rain = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_rain_mean,
        acdc_rain_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    rain_set = ACDC(
        dataroots["acdc"],
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

    print(f"> Loaded {len(rain_set)} rainy acdc images.")
    remap_labels = tf.Lambda(lm)
    trans_fog = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_fog_mean,
        acdc_fog_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    fog_set = ACDC(
        dataroots["acdc"],
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

    print(f"> Loaded {len(fog_set)} foggy acdc images.")
    remap_labels = tf.Lambda(lm)
    trans_dark = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_night_mean,
        acdc_night_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    dark_set = ACDC(
        dataroots["acdc"],
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

    print(f"> Loaded {len(dark_set)} dark acdc images.")

    train_city = prepare_city_datasets(dataroots, train_transforms, config)[0]

    print(f"> Loaded {len(train_city)} city train images.")

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

    print(f"> Loaded {len(wd)} wilddash images.")

    mix_datasets = [
        train_city,
        dz,
        rain_set,
        fog_set,
        snow_set,
        dark_set,
        wd,
        dz_pseudo_night,
    ]
    if config["uniform_class"]:
        mix_datasets = [
            UniformClassDataset(ds, class_uniform_pct=0.75, class_uniform_tile=512)
            for ds in mix_datasets
        ]
        print("> Created uniform dataset")

    return mix_datasets


def prepare_architecture_sweep_ps_night_datasets_4city(
    dataroots, train_transforms, config
):
    remap_labels = tf.Lambda(lm)
    dz_night_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        dark_zurich_night_mean,
        dark_zurich_night_std,
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

    print(f"> Loaded {len(dz_pseudo_night)} dark zurich images.")

    train_city = prepare_city_datasets(dataroots, train_transforms, config)[0]

    print(f"> Loaded {len(train_city)} city train images.")

    mix_datasets = [
        train_city,
        dz_pseudo_night,
    ]

    if config["uniform_class"]:
        mix_datasets = [
            UniformClassDataset(ds, class_uniform_pct=0.75, class_uniform_tile=512)
            for ds in mix_datasets
        ]
        print("> Created uniform dataset")

    return mix_datasets


def load_architecture_sweep_train_datasets(dataroots, bs, train_transforms, config):
    print(dataroots)
    datasets = prepare_architecture_sweep_datasets(dataroots, train_transforms, config)
    train_set = ConcatDataset(datasets=datasets)

    print(f"> Loaded {len(train_set)} train images (for architecture sweep).")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )

    return train_loader


def load_architecture_sweep_train_datasets_wo_acdc(
    dataroots, bs, train_transforms, config
):
    datasets = prepare_architecture_sweep_datasets_wo_acdc(
        dataroots, train_transforms, config
    )
    train_set = ConcatDataset(datasets=datasets)

    print(f"> Loaded {len(train_set)} train images (for architecture sweep).")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )

    return train_loader


def load_architecture_sweep_train_ps_night_datasets(
    dataroots, bs, train_transforms, config
):
    datasets = prepare_architecture_sweep_ps_night_datasets(
        dataroots, train_transforms, config
    )
    train_set = ConcatDataset(datasets=datasets)

    print(f"> Loaded {len(train_set)} train images (for architecture sweep).")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )

    return train_loader


def load_architecture_sweep_train_ps_night_datasets_4city(
    dataroots, bs, train_transforms, config
):
    datasets = prepare_architecture_sweep_ps_night_datasets_4city(
        dataroots, train_transforms, config
    )
    train_set = ConcatDataset(datasets=datasets)

    print(f"> Loaded {len(train_set)} train images (for architecture sweep).")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )

    return train_loader


def prepare_architecture_sweep_ps_night_datasets_4city_onacdc_night(
    dataroots, train_transforms, config
):
    remap_labels = tf.Lambda(lm)
    acdc_night_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_night_mean,
        acdc_night_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    acdc_pseudo_night = ACDCPseudolabeled(
        dataroots["acdc"],
        split="train",
        tag="night",
        image_transform=tf.Compose(acdc_night_trans["image"]),
        target_transform=tf.Compose(acdc_night_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(acdc_night_trans["joint"]),
    )

    print(f"> Loaded {len(acdc_pseudo_night)} dark zurich images.")

    train_city = prepare_city_datasets(dataroots, train_transforms, config)[0]

    print(f"> Loaded {len(train_city)} city train images.")

    mix_datasets = [
        train_city,
        acdc_pseudo_night,
    ]

    if config["uniform_class"]:
        mix_datasets = [
            UniformClassDataset(ds, class_uniform_pct=0.75, class_uniform_tile=512)
            for ds in mix_datasets
        ]
        print("> Created uniform dataset")

    return mix_datasets


def load_architecture_sweep_train_ps_night_datasets_4city_onacdc_night(
    dataroots, bs, train_transforms, config
):
    datasets = prepare_architecture_sweep_ps_night_datasets_4city_onacdc_night(
        dataroots, train_transforms, config
    )
    train_set = ConcatDataset(datasets=datasets)

    print(f"> Loaded {len(train_set)} train images (for architecture sweep).")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )

    return train_loader


def prepare_architecture_sweep_ps_datasets_4city_onacdc(
    dataroots, train_transforms, config
):
    remap_labels = tf.Lambda(lm)
    acdc_night_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_night_mean,
        acdc_night_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    acdc_pseudo_night = ACDCPseudolabeled(
        dataroots["acdc"],
        split="train",
        tag="night",
        image_transform=tf.Compose(acdc_night_trans["image"]),
        target_transform=tf.Compose(acdc_night_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(acdc_night_trans["joint"]),
    )
    acdc_snow_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_snow_mean,
        acdc_snow_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    acdc_pseudo_snow = ACDCPseudolabeled(
        dataroots["acdc"],
        split="train",
        tag="snow",
        image_transform=tf.Compose(acdc_snow_trans["image"]),
        target_transform=tf.Compose(acdc_snow_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(acdc_snow_trans["joint"]),
    )
    acdc_rain_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_rain_mean,
        acdc_rain_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    acdc_pseudo_rain = ACDCPseudolabeled(
        dataroots["acdc"],
        split="train",
        tag="rain",
        image_transform=tf.Compose(acdc_rain_trans["image"]),
        target_transform=tf.Compose(acdc_rain_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(acdc_rain_trans["joint"]),
    )
    acdc_fog_trans = create_per_dataset_train_transform(
        copy.deepcopy(train_transforms),
        acdc_fog_mean,
        acdc_fog_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    acdc_pseudo_fog = ACDCPseudolabeled(
        dataroots["acdc"],
        split="train",
        tag="fog",
        image_transform=tf.Compose(acdc_fog_trans["image"]),
        target_transform=tf.Compose(acdc_fog_trans["target"] + [remap_labels]),
        joint_transform=tf.Compose(acdc_fog_trans["joint"]),
    )

    train_city = prepare_city_datasets(dataroots, train_transforms, config)[0]

    print(f"> Loaded {len(train_city)} city train images.")

    mix_datasets = [
        train_city,
        acdc_pseudo_night,
        acdc_pseudo_fog,
        acdc_pseudo_rain,
        acdc_pseudo_snow,
    ]

    if config["uniform_class"]:
        mix_datasets = [
            UniformClassDataset(ds, class_uniform_pct=0.75, class_uniform_tile=512)
            for ds in mix_datasets
        ]
        print("> Created uniform dataset")

    return mix_datasets


def load_architecture_sweep_train_ps_datasets_4city_onacdc(
    dataroots, bs, train_transforms, config
):
    datasets = prepare_architecture_sweep_ps_datasets_4city_onacdc(
        dataroots, train_transforms, config
    )
    train_set = ConcatDataset(datasets=datasets)

    print(f"> Loaded {len(train_set)} train images (for architecture sweep).")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )

    return train_loader
