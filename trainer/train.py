import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datamodule import CustomDataModule
from model import SemSegment


def main():
    class_weights = np.array(
        [0.13130568, 1., 0.75924934, 0.7089291]
    )
    class_weights = torch.Tensor(class_weights).cuda()
    model = SemSegment(
        num_classes=4,
        num_layers=5,
        features_start=16,
        class_weights=class_weights
    )

    dm = CustomDataModule(
        '/home/judicator/dozer/datasets/set_2',
        batch_size=1,
        shuffle=True,
        pin_memory=True
    )

    checkpoint_callback = ModelCheckpoint()
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=3,
        verbose=True,
    )
    trainer = Trainer(
        max_epochs=100,
        gpus=1,
        callbacks=[checkpoint_callback, stop_callback]
    )

    trainer.fit(model, datamodule=dm)
    torch.save(model.state_dict(), "./trainer/models/model_set2.json")


if __name__ == '__main__':
    main()
