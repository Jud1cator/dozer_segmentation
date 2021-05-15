import os
from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms as transforms

from dataset import DirDataset


class CustomDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 8,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: where to load the data from path, i.e. '/path/to/folder/with/data_semantics/'
            val_split: size of validation test (default 0.2)
            test_split: size of test set (default 0.1)
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # split into train, val, test
        dataset = DirDataset(
            self.data_dir,
            transform=self._default_transforms()
        )

        val_len = round(val_split * len(dataset))
        test_len = round(test_split * len(dataset))
        train_len = len(dataset) - val_len - test_len

        self.trainset, self.valset, self.testset = random_split(
            dataset,
            lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def _default_transforms(self) -> Callable:
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        return trans
