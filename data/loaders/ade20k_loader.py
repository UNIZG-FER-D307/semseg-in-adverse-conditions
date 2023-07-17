from torchvision import transforms as tf
from torch.utils.data import DataLoader
from data.datasets import ADE20k, UniformClassDataset
from data.datasets.ade20k.ade20k import create_ade_id_to_train_id
from torch.utils.data import ConcatDataset
import torch

mapper = create_ade_id_to_train_id()
def lm(x):
    return mapper[(x * 255.).long()]

def load_ade20k_train(dataroot, bs, train_transforms, uniform=False):
    remap_labels = tf.Lambda(lm)
    train_set = ADE20k(dataroot, split='training',
                                  image_transform=None if not train_transforms['image'] else tf.Compose(train_transforms['image']),
                                  target_transform=None if not train_transforms['target'] else tf.Compose(train_transforms['target'] + [remap_labels]),
                                  joint_transform=None if not train_transforms['joint'] else tf.Compose(train_transforms['joint']))
    train_set = train_set if not uniform else UniformClassDataset(train_set, class_uniform_pct=0.75, class_uniform_tile=512)
    print(f"> Loaded {len(train_set)} train images.")
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=False, pin_memory=True, num_workers=3)
    return train_loader

def load_ade20k_val(dataroot, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_set = ADE20k(dataroot, split='validation',
                                  image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                                  target_transform=None if not val_transforms['target'] else tf.Compose(val_transforms['target']+ [remap_labels]),
                                  joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(val_set)} val images.")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=1)
    return val_loader

def load_ade20k_trainval(dataroot, bs, train_transforms, uniform=False):
    remap_labels = tf.Lambda(lm)
    train_set = ADE20k(dataroot, split='training',
                                  image_transform=None if not train_transforms['image'] else tf.Compose(train_transforms['image']),
                                  target_transform=None if not train_transforms['target'] else tf.Compose(train_transforms['target'] + [remap_labels]),
                                  joint_transform=None if not train_transforms['joint'] else tf.Compose(train_transforms['joint']))
    val_set = ADE20k(dataroot, split='validation',
                                  image_transform=None if not train_transforms['image'] else tf.Compose(train_transforms['image']),
                                  target_transform=None if not train_transforms['target'] else tf.Compose(train_transforms['target'] + [remap_labels]),
                                  joint_transform=None if not train_transforms['joint'] else tf.Compose(train_transforms['joint']))
    ds = ConcatDataset(
        [set if not uniform else UniformClassDataset(set, class_uniform_pct=0.75, class_uniform_tile=512) for set in [train_set, val_set]]
    )
    print(f"> Loaded {len(ds)} train images.")
    loader = DataLoader(ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=5)
    return loader

# def load_city_test(dataroot, val_transforms):
#     val_set = CityscapesTest(dataroot, split='test', return_name=True,
#                                   image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']))
#     print(f"> Loaded {len(val_set)} val images.")
#     val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=2)
#     return val_loader
