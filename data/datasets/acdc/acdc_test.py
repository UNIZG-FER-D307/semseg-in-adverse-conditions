import os
from PIL import Image
from torch.utils import data

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if
            (filename.endswith(suffix) and os.path.isfile(os.path.join(looproot, filename)))]

class ACDCTest(data.Dataset):

    def __init__(self, root, tag, split='test', image_transform=None):
        self.root = root
        assert tag in ['snow', 'rain', 'fog', 'night']
        self.tag = tag

        self.image_transform = image_transform

        self.images_base = os.path.join(self.root, 'rgb_anon', tag, split)
        self.image_paths = recursive_glob(rootdir=self.images_base, suffix='.png')

        if not len(self.image_paths):
            raise Exception(f"> No files for split=test found in {self.root}")

        # print(f"> ACDC: found {len(self.image_paths)} images for tag {tag}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        image = Image.open(img_path)

        if self.image_transform:
            image = self.image_transform(image)

        return image, img_path


class ACDCFullTest(data.Dataset):

    def __init__(self, root, split='test', image_transform=None):
        night = ACDCTest(root, 'night', split, image_transform)
        snow = ACDCTest(root, 'snow', split, image_transform)
        rain = ACDCTest(root, 'rain', split, image_transform)
        fog = ACDCTest(root, 'fog', split, image_transform)
        self.data = data.ConcatDataset([night, snow, rain, fog])
        # print(f"> Loaded {len(self)} images!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

