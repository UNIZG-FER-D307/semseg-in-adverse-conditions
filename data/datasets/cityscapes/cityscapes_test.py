import numpy as np
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

class CityscapesTest(data.Dataset):

    def __init__(self, root, split='test', return_name=False, image_transform=None):
        self.root = root
        self.split = split
        self.image_transform = image_transform
        assert split in ['test']

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.files = recursive_glob(rootdir=self.images_base, suffix='.png')
        self.return_name = return_name

        if not len(self.files):
            raise Exception("> No files for split=[%s] found in %s" % (split, self.images_base))

        print("> Found %d %s images..." % (len(self.files), split))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        image = Image.open(img_path)

        if self.image_transform:
            image = self.image_transform(image)

        if self.return_name:
            return image, img_path
        return image