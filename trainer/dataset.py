import os

import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class DirDataset(Dataset):
    IMAGE_PATH = 'images_no_dirt'
    MASK_PATH = 'masks_no_dirt'
    DEFAULT_CLASSES = (191, 205, 236, 255)

    def __init__(
        self,
        data_dir: str,
        img_size: tuple = (1920, 1080),
        classes: list = DEFAULT_CLASSES,
        ignore_index: int = 255,
        transform=None
    ):
        """
        Args:
            data_dir (str): where to load the data from path, i.e. '/path/to/folder/with/data_semantics/'
            img_size: image dimensions (width, height)
        """
        self.img_size = img_size
        self.class_map = dict(zip(range(len(classes)), classes))
        self.n_classes = len(classes)
        self.ignore_index = ignore_index
        self.transform = transform
        self.data_dir = data_dir
        self.img_path = os.path.join(self.data_dir, self.IMAGE_PATH)
        self.mask_path = os.path.join(self.data_dir, self.MASK_PATH)
        self.img_list = sorted(self.get_filenames(self.img_path))
        self.mask_list = sorted(self.get_filenames(self.mask_path))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.resize(self.img_size)
        img = np.array(img)

        mask = Image.open(self.mask_list[idx])
        mask = mask.resize(self.img_size)
        mask = np.array(mask)
        mask = self.encode_segmap(mask)

        if self.transform:
            img = self.transform(img)

        return img, mask

    def encode_segmap(self, mask):
        for class_n in self.class_map.keys():
            mask[mask == self.class_map[class_n]] = class_n
        mask[mask >= self.n_classes] = self.ignore_index
        return mask

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list
