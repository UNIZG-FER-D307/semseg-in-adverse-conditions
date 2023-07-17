import numpy as np
import os
from PIL import Image
from torch.utils import data
from collections import defaultdict

dark_zurich_day_mean = np.array([0.36500044, 0.34681207, 0.35096232])
dark_zurich_day_std = np.array([0.25851004, 0.26869385, 0.28189976])

dark_zurich_night_mean = np.array([0.22538802, 0.16680287, 0.116497])
dark_zurich_night_std = np.array([0.17072221, 0.14552379, 0.12518748])

dark_zurich_twilight_mean = np.array([0.23792198, 0.19852797, 0.18778289])
dark_zurich_twilight_std = np.array([0.17287927, 0.162831, 0.19137911])


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if
            (filename.endswith(suffix) and os.path.isfile(os.path.join(looproot, filename)))]

class DarkZurichFull(data.Dataset):

    def __init__(self, root, split='train', day_image_transform=None, twilight_image_transform=None, night_image_transform=None):
        self.root = root
        self.split = split

        self.day_image_transform = day_image_transform
        self.twilight_image_transform = twilight_image_transform
        self.night_image_transform = night_image_transform
        self.images_base = os.path.join(self.root, 'rgb_anon', self.split)
        all_images = recursive_glob(rootdir=self.images_base, suffix='.png')
        self.day_images = self._extract_day_images(all_images)
        self.name2gps, rest = self._load_gps_coordinates()
        self.gps2night = rest[1][1]
        self.gps2twilight = rest[2][1]

        self.night_coord = self._extract_coordinates(self.gps2night)
        self.twilight_coord = self._extract_coordinates(self.gps2twilight)

        if not len(all_images):
            raise Exception("> No files for split=[%s] found in %s" % (split, self.images_base))

    def _extract_day_images(self, all_images):
        images = [im.replace(self.images_base, '') for im in all_images]
        return [self.split + im for im in list(filter(lambda x: 'day' in x, images))]

    def _load_gps_coordinates(self):
        csv_bases = os.path.join(self.root, 'gps', self.split)
        csv_files = recursive_glob(rootdir=csv_bases, suffix='.csv')
        name_to_gps = dict()
        g2n = []
        for condition in ['day', 'night', 'twilight']:
            gps_to_name = defaultdict(list)
            for file in filter(lambda x: condition in x, csv_files):
                with open(file) as f:
                    content = f.readlines()
                    content = [c.strip() for c in content]
                    for c in content:
                        name, a, b = c.split(',')
                        gps = str(round(float(a), 5)) + ',' + str(round(float(b), 5))
                        name = name + '_rgb_anon.png'
                        name_to_gps[name] = gps
                        gps_to_name[gps].append(name)
            g2n.append((condition, gps_to_name))
        return name_to_gps, g2n

    def _extract_coordinates(self, gps2im):
        return [(float(k.split(',')[0]), float(k.split(',')[1])) for k in gps2im.keys()]

    def _query_nearest(self, gps, coord):
        a, b = float(gps.split(',')[0]), float(gps.split(',')[1])
        id = np.argmin([np.sqrt((a - a_) ** 2 + (b - b_) ** 2) for a_, b_ in coord])
        nearest_a, nearest_b = coord[id]
        k = str(nearest_a) + ',' + str(nearest_b)
        return k

    def __len__(self):
        return len(self.day_images)

    def __getitem__(self, index):
        day_img_name = self.day_images[index]
        day_img_path = os.path.join(self.root, 'rgb_anon', day_img_name)

        if not os.path.isfile(day_img_path) or not os.path.exists(day_img_path):
            raise Exception("{} is not a file, can not open with imread.".format(day_img_path))
        day_img = Image.open(day_img_path)

        gps = self.name2gps[day_img_name]
        twilight_img_name = np.random.choice(self.gps2twilight[self._query_nearest(gps, self.twilight_coord)])
        night_img_name = np.random.choice(self.gps2night[self._query_nearest(gps, self.night_coord)])

        twilight_img_path = os.path.join(self.root, 'rgb_anon', twilight_img_name)
        if not os.path.isfile(twilight_img_path) or not os.path.exists(twilight_img_path):
            raise Exception("{} is not a file, can not open with imread.".format(twilight_img_path))
        assert 'twilight' in twilight_img_path
        twilight_img = Image.open(twilight_img_path)

        night_img_path = os.path.join(self.root, 'rgb_anon', night_img_name)
        if not os.path.isfile(night_img_path) or not os.path.exists(night_img_path):
            raise Exception("{} is not a file, can not open with imread.".format(night_img_path))
        assert 'night' in night_img_path
        night_img = Image.open(night_img_path)

        if self.day_image_transform:
            day_img = self.day_image_transform(day_img)
        if self.twilight_image_transform:
            twilight_img = self.twilight_image_transform(twilight_img)
        if self.night_image_transform:
            night_img = self.night_image_transform(night_img)

        return day_img, twilight_img, night_img
