from torchvision import transforms as tf
from torch.utils.data import DataLoader
import copy
from torch.utils.data import ConcatDataset
from data.datasets import FoggyZurich, FoggyDriving, FoggyCityscapes, ACDC, FoggyDrivingCoarse
from data.datasets import foggy_city_std, foggy_city_mean, foggy_driving_mean, foggy_driving_std, foggy_zurich_mean, \
    foggy_zurich_std, acdc_fog_mean, acdc_fog_std
from data.transforms import JitterRandomCrop
from data.datasets import BDDPseudolabeledConditionwise, bdd_fog_std, bdd_fog_mean, STFPseudolabeled, stf_std, stf_mean, UniformClassDataset

def create_per_dataset_train_transform(transforms, mean, std, crop_size, scale, ignore_id):
    transforms['image'] = transforms['image'] + [tf.Normalize(mean, std)]
    transforms['joint'] = [
                              JitterRandomCrop(size=crop_size, scale=scale, ignore_id=ignore_id, input_mean=(0.))
                          ] + transforms['joint']
    return transforms

def lm(x):
    return (x * 255.).long()

def prepare_foggy_datasets(dataroots, train_transforms, config):
    remap_labels = tf.Lambda(lm)
    fz_trans = create_per_dataset_train_transform(copy.deepcopy(train_transforms), foggy_zurich_mean, foggy_zurich_std,
                                                  crop_size=config['crop_size'], scale=config['jitter_range'], ignore_id=config['ignore_id'])
    fz = FoggyZurich(dataroots['foggy_zurich'],
                     image_transform=tf.Compose(fz_trans['image']),
                     target_transform=tf.Compose(fz_trans['target'] + [remap_labels]),
                     joint_transform=tf.Compose(fz_trans['joint']))

    fd_trans = create_per_dataset_train_transform(copy.deepcopy(train_transforms), foggy_driving_mean, foggy_driving_std,
                                                  crop_size=config['crop_size'], scale=config['jitter_range'], ignore_id=config['ignore_id'])
    fd = FoggyDriving(dataroots['foggy_driving'],
                      image_transform=tf.Compose(fd_trans['image']),
                      target_transform=tf.Compose(fd_trans['target'] + [remap_labels]),
                      joint_transform=tf.Compose(fd_trans['joint']))
    fdc = FoggyDrivingCoarse(dataroots['foggy_driving'],
                      image_transform=tf.Compose(fd_trans['image']),
                      target_transform=tf.Compose(fd_trans['target'] + [remap_labels]),
                      joint_transform=tf.Compose(fd_trans['joint']))

    fc_trans = create_per_dataset_train_transform(copy.deepcopy(train_transforms), foggy_city_mean, foggy_city_std,
                                                  crop_size=config['crop_size'], scale=config['jitter_range'], ignore_id=config['ignore_id'])
    fc_train = FoggyCityscapes(dataroots['foggy_city'], split='train',
                               image_transform=tf.Compose(fc_trans['image']),
                               target_transform=tf.Compose(fc_trans['target'] + [remap_labels]),
                               joint_transform=tf.Compose(fc_trans['joint']))
    fc_val = FoggyCityscapes(dataroots['foggy_city'], split='val',
                               image_transform=tf.Compose(fc_trans['image']),
                               target_transform=tf.Compose(fc_trans['target'] + [remap_labels]),
                               joint_transform=tf.Compose(fc_trans['joint']))

    acdc_trans = create_per_dataset_train_transform(copy.deepcopy(train_transforms), acdc_fog_mean, acdc_fog_std,
                                                    crop_size=config['crop_size'], scale=config['jitter_range'], ignore_id=config['ignore_id'])
    acdc_fog = ACDC(dataroots['acdc'], split='train', tag='fog',
                    image_transform=tf.Compose(acdc_trans['image']),
                    target_transform=tf.Compose(acdc_trans['target'] + [remap_labels]),
                    joint_transform=tf.Compose(acdc_trans['joint']))

    bdd_trans = create_per_dataset_train_transform(copy.deepcopy(train_transforms), bdd_fog_mean, bdd_fog_std,
                                                    crop_size=config['crop_size'], scale=config['jitter_range'],
                                                    ignore_id=config['ignore_id'])
    bdd_fog = BDDPseudolabeledConditionwise(dataroots['bdd100k'], split='train', condition='foggy',
                    image_transform=tf.Compose(bdd_trans['image']),
                    target_transform=tf.Compose(bdd_trans['target'] + [remap_labels]),
                    joint_transform=tf.Compose(bdd_trans['joint']))

    stf_trans = create_per_dataset_train_transform(copy.deepcopy(train_transforms), stf_mean, stf_std,
                                                   crop_size=config['crop_size'], scale=config['jitter_range'],
                                                   ignore_id=config['ignore_id'])
    stf_fog = STFPseudolabeled(dataroots['stf'],
                                            image_transform=tf.Compose(stf_trans['image']),
                                            target_transform=tf.Compose(stf_trans['target'] + [remap_labels]),
                                            joint_transform=tf.Compose(stf_trans['joint']))

    datasets = [fz, fd, fdc, fc_train, fc_val, acdc_fog, bdd_fog, stf_fog]
    # datasets = [acdc_fog]
    return datasets

def load_fog_finetune(dataroots, bs, train_transforms, config):
    ds_rain = prepare_foggy_datasets(dataroots, train_transforms, config)
    ds_rain = ds_rain[:-1] # remove stf, its more snowy
    if config['uniform_class']:
        ds_rain = [
            UniformClassDataset(ds, class_uniform_pct=0.75, class_uniform_tile=512)
            for ds in ds_rain
        ]
        print('> Created uniform dataset')
    ds = ConcatDataset(ds_rain)

    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=5, drop_last=True)
    return train_loader

def load_foggy_train(dataroots, bs, train_transforms, config):
    datasets = prepare_foggy_datasets(dataroots, train_transforms, config)
    ds = ConcatDataset(datasets)
    print(f"> Loaded {len(ds)} train images.")
    train_loader = DataLoader(ds, batch_size=bs, shuffle=True, pin_memory=False, num_workers=3)
    return train_loader

def load_foggy_acdc_val(dataroot, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_set = ACDC(dataroot, split='val', tag='fog',
                                  image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                                  target_transform=None if not val_transforms['target'] else tf.Compose(val_transforms['target']+ [remap_labels]),
                                  joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(val_set)} val images.")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=2)
    return val_loader

def load_fog_eval(dataroots, val_transforms):
    remap_labels = tf.Lambda(lm)
    val_transforms['image'] = val_transforms['image'] + [tf.Normalize(acdc_fog_mean, acdc_fog_std)]
    acdc_trans = val_transforms
    dataset = ACDC(dataroots['acdc'], split='val', tag='fog',
                    image_transform=tf.Compose(acdc_trans['image']),
                    target_transform=tf.Compose(acdc_trans['target'] + [remap_labels]),
                    joint_transform=tf.Compose(acdc_trans['joint']))
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=2)
    return val_loader