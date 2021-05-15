from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from unet import UNet


class SemSegment(pl.LightningModule):

    def __init__(
        self,
        num_classes,
        lr: float = 0.01,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
        ignore_index: int = 255,
        class_weights: list = None
    ):
        """
        Basic model for semantic segmentation. Uses UNet architecture by default.
        The default parameters in this model are for the KITTI dataset. Note, if you'd like to use this model as is,
        you will first need to download the KITTI dataset yourself. You can download the dataset `here.
        <http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015>`_
        Implemented by:
            - `Annika Brundyn <https://github.com/annikabrundyn>`_
        Args:
            num_layers: number of layers in each side of U-net (default 5)
            features_start: number of features in first layer (default 64)
            bilinear: whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
            lr: learning (default 0.01)
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        # self.class_weights = torch.Tensor(np.array([
        #     0.1, 45.46760687, 35.33661025, 15.24232113, 11.88717358
        # ])).cuda()
        self.loss = torch.nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=self.ignore_index
        )
        self.net = UNet(
            num_classes=num_classes,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        # loss_val = F.cross_entropy(out, mask, weight=self.class_weights)
        loss_val = self.loss(out, mask)
        self.log('train_loss', loss_val)
        return {'loss': loss_val}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        # loss_val = F.cross_entropy(out, mask, weight=self.class_weights)
        loss_val = self.loss(out, mask)
        self.log('val_loss', loss_val)
        return {'val_loss': loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', loss_val)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument(
            "--bilinear",
            action='store_true',
            default=False,
            help="whether to use bilinear interpolation or transposed"
        )

        return parser
