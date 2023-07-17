from torchvision import transforms as tf
from data.transforms import JitterRandomCrop
from data.datasets import GTA5, UniformClassDataset
from torch.utils.data import DataLoader
from data import gta5_mean, gta5_std


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


def load_gta5_train_dataset(dataroot, bs, train_transforms, config):
    remap_labels = tf.Lambda(lm)
    gta5_trans = create_per_dataset_train_transform(
        train_transforms,
        mean=gta5_mean,
        std=gta5_std,
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        ignore_id=config["ignore_id"],
    )
    train_set = GTA5(
        dataroot,
        image_transform=None
        if not gta5_trans["image"]
        else tf.Compose(gta5_trans["image"]),
        target_transform=None
        if not gta5_trans["target"]
        else tf.Compose(gta5_trans["target"] + [remap_labels]),
        joint_transform=None
        if not gta5_trans["joint"]
        else tf.Compose(gta5_trans["joint"]),
    )

    print(f"> Loaded {len(train_set)} train images.")
    train_set = (
        UniformClassDataset(
            train_set,
            class_uniform_pct=0.75,
            class_uniform_tile=1024,
        )
        if config["uniform_class"]
        else train_set
    )

    return DataLoader(
        train_set,
        batch_size=bs,
        shuffle=True,
        pin_memory=False,
        num_workers=3,
        drop_last=True,
    )
