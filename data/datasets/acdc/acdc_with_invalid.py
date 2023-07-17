import numpy as np
import os
from PIL import Image
from torch.utils import data

acdc_mean = np.array([0.40875698, 0.38671499, 0.37331395]) # mean of whole non clear
acdc_std = np.array([0.26754202, 0.27372122, 0.29311962]) # std of whole non clear

acdc_night_mean = np.array([0.22906968, 0.17921552, 0.13496838])
acdc_night_std = np.array([0.17685317, 0.159368, 0.14453114])

acdc_fog_mean = np.array([0.50454965, 0.49473348, 0.50199624])
acdc_fog_std = np.array([0.24546748, 0.24972116, 0.26429628])

acdc_snow_mean = np.array([0.50527985, 0.49587269, 0.50571361])
acdc_snow_std = np.array([0.23504374, 0.23778903, 0.24963569])

acdc_rain_mean = np.array([0.42422908, 0.41312432, 0.40999521])
acdc_rain_std = np.array([0.30029695, 0.30299347, 0.31839669])


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if
            (filename.endswith(suffix) and os.path.isfile(os.path.join(looproot, filename)))]

class ACDCWithInvalid(data.Dataset):

    def __init__(self, root, split, tag, image_transform=None, target_transform=None, joint_transform=None):
        self.root = root
        assert split in ['train', 'val']
        self.split = split
        assert tag in ['snow', 'rain', 'fog', 'night']
        self.tag = tag

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images_base = os.path.join(self.root, 'rgb_anon', tag, self.split)
        self.image_paths = recursive_glob(rootdir=self.images_base, suffix='.png')

        if not len(self.image_paths):
            raise Exception(f"> No files for split={split} found in {self.root}")

        # print(f"> ACDC: found {len(self.image_paths)} images for tag {tag}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        lbl_path = img_path.replace('rgb_anon/', 'gt/').replace('_rgb_anon.png', '_gt_labelTrainIds.png')

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        image = Image.open(img_path)

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))
        label = Image.open(lbl_path)

        inv_lbl_path = lbl_path.replace('_gt_labelTrainIds.png', '_gt_invIds.png')
        if not os.path.isfile(inv_lbl_path) or not os.path.exists(inv_lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(inv_lbl_path))
        inv_label = Image.open(inv_lbl_path)
        inv_label = np.array(inv_label)
        label = np.array(label)
        label[inv_label == 1] = 254
        label = Image.fromarray(label)

        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return self.joint_transform((image, label)) if self.joint_transform else (image, label)

    def create_img_lbl_list(self):
        return [
            (img_path, img_path.replace('rgb_anon/', 'gt/').replace('_rgb_anon.png', '_gt_labelTrainIds.png'))
            for img_path in self.image_paths]

    def create_label_mapper(self):
        return None

    def create_cache_name(self):
        return f"{type(self).__name__}_{self.split}_{self.tag}"


class ACDCFullWithInvalid(data.Dataset):

    def __init__(self, root, split, image_transform=None, target_transform=None, joint_transform=None):
        self.night = ACDCWithInvalid(root, split, 'night', image_transform, target_transform, joint_transform)
        self.snow = ACDCWithInvalid(root, split, 'snow', image_transform, target_transform, joint_transform)
        self.rain = ACDCWithInvalid(root, split, 'rain', image_transform, target_transform, joint_transform)
        self.fog = ACDCWithInvalid(root, split, 'fog', image_transform, target_transform, joint_transform)
        self.data = data.ConcatDataset([self.night, self.snow, self.rain, self.fog])
        # print(f"> Loaded {len(self)} images!")
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.split = split


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def create_img_lbl_list(self):
        return self.night.create_img_lbl_list() + self.snow.create_img_lbl_list() \
               + self.rain.create_img_lbl_list() + self.fog.create_img_lbl_list()

