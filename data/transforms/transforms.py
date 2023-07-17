import torchvision.transforms as tf
from .joint_transforms import (
    JitterRandomCrop,
    RandomHorizontalFlip,
    LabelToBoundaryClone,
    ImageJitterRandomCrop,
)
from .joint_transforms.image_transforms import DarkenImage
from .ood_transforms import CreateRandomPatch
from torchvision.transforms import ColorJitter


def create_basic_transform(mean, std):
    transforms = {
        "image": [tf.ToTensor(), tf.Normalize(mean, std)],
        "target": [
            tf.ToTensor(),
        ],
        "joint": None,
    }
    return transforms


def create_basic_BA_transform(mean, std, num_classes, ignore_id):
    transforms = {
        "image": [tf.ToTensor(), tf.Normalize(mean, std)],
        "target": [
            tf.ToTensor(),
        ],
        "joint": [LabelToBoundaryClone(num_classes=num_classes, ignore_id=ignore_id)],
    }
    return transforms


def create_jitter_random_crop_transform(
    crop_size, scale, input_mean, ignore_id, mean, std
):
    transforms = {
        "image": [tf.ToTensor(), tf.Normalize(mean, std)],
        "target": [
            tf.ToTensor(),
        ],
        "joint": [
            JitterRandomCrop(
                size=crop_size,
                scale=scale,
                ignore_id=ignore_id,
                input_mean=(input_mean),
            ),
            RandomHorizontalFlip(),
        ],
        "ood": [CreateRandomPatch()],
    }
    return transforms


def create_jitter_random_crop_BA_transform(
    crop_size, scale, input_mean, num_classes, ignore_id, mean, std
):
    transforms = {
        "image": [tf.ToTensor(), tf.Normalize(mean, std)],
        "target": [
            tf.ToTensor(),
        ],
        "joint": [
            JitterRandomCrop(
                size=crop_size,
                scale=scale,
                ignore_id=ignore_id,
                input_mean=(input_mean),
            ),
            RandomHorizontalFlip(),
            LabelToBoundaryClone(num_classes=num_classes, ignore_id=ignore_id),
        ],
        "ood": [CreateRandomPatch()],
    }
    return transforms


import numpy as np


def create_dark_jitter_random_crop_BA_transform(
    crop_size, scale, input_mean, num_classes, ignore_id
):
    # cityscapes for darken image transform
    mean = np.array([0.04771631, 0.05432942, 0.0471795])
    std = np.array([0.03458254, 0.03553367, 0.03456279])
    transforms = {
        "image": [tf.ToTensor(), DarkenImage(), tf.Normalize(mean, std)],
        "target": [
            tf.ToTensor(),
        ],
        "joint": [
            JitterRandomCrop(
                size=crop_size,
                scale=scale,
                ignore_id=ignore_id,
                input_mean=(input_mean),
            ),
            RandomHorizontalFlip(),
            LabelToBoundaryClone(num_classes=num_classes, ignore_id=ignore_id),
        ],
    }
    return transforms


def create_raw_train_BA_colorjitter_transform(
    num_classes, ignore_id, brightness=0, contrast=0, saturation=0, hue=0
):
    transforms = {
        "image": [
            tf.ToTensor(),
            ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
            ),
        ],
        "target": [
            tf.ToTensor(),
        ],
        "joint": [
            RandomHorizontalFlip(),
            LabelToBoundaryClone(num_classes=num_classes, ignore_id=ignore_id),
        ],
    }
    return transforms


def create_raw_train_BA_transform(num_classes, ignore_id):
    transforms = {
        "image": [
            tf.ToTensor(),
        ],
        "target": [
            tf.ToTensor(),
        ],
        "joint": [
            RandomHorizontalFlip(),
            LabelToBoundaryClone(num_classes=num_classes, ignore_id=ignore_id),
        ],
    }
    return transforms


def create_raw_train_transform():
    transforms = {
        "image": [
            tf.ToTensor(),
        ],
        "target": [
            tf.ToTensor(),
        ],
        "joint": [
            RandomHorizontalFlip(),
        ],
    }
    return transforms


def create_raw_val_BA_transform(num_classes, ignore_id):
    transforms = {
        "image": [
            tf.ToTensor(),
        ],
        "target": [
            tf.ToTensor(),
        ],
        "joint": [LabelToBoundaryClone(num_classes=num_classes, ignore_id=ignore_id)],
    }
    return transforms


def create_raw_val_transform():
    transforms = {
        "image": [
            tf.ToTensor(),
        ],
        "target": [
            tf.ToTensor(),
        ],
        "joint": None,
    }
    return transforms


def create_selfsup_basic_transform(
    crop_size, day_mean, day_std, twilight_mean, twilight_std, night_mean, night_std
):
    transforms = {
        "day": [
            tf.ToTensor(),
            tf.Normalize(day_mean, day_std),
            tf.RandomCrop(crop_size)
            # ImageJitterRandomCrop(size=crop_size, scale=(0.5, 2), ignore_id=255, input_mean=(0.))
        ],
        "twilight": [
            tf.ToTensor(),
            tf.Normalize(twilight_mean, twilight_std),
            tf.RandomCrop(crop_size)
            # ImageJitterRandomCrop(size=crop_size, scale=(0.5, 2), ignore_id=255, input_mean=(0.))
        ],
        "night": [
            tf.ToTensor(),
            tf.Normalize(night_mean, night_std),
            tf.RandomCrop(crop_size)
            # ImageJitterRandomCrop(size=crop_size, scale=(0.7, 1.4), ignore_id=255, input_mean=(0.))
        ],
    }
    return transforms


def create_unsup_transform():
    transforms = {
        "image": [
            tf.ToTensor(),
        ]
    }
    return transforms
