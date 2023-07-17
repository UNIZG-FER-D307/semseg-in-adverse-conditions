from torchvision import transforms as tf
from torch.utils.data import DataLoader
from data.datasets import FishyscapesLostAndFound

def lm(x):
    return (x * 255.).long()

def load_fishyscapes_lost_and_found_val(dataroot, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_set = FishyscapesLostAndFound(dataroot,
                                  image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                                  target_transform=None if not val_transforms['target'] else tf.Compose(val_transforms['target']+ [remap_labels]),
                                  joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(val_set)} val images.")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=2)
    return val_loader
