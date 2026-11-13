import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.data.dataset import DenoisingDataset
from src.models.train_unet import UNetModel


SIGMA = 0.03
BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 0.0001
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


def train():
    L.seed_everything(42, verbose=False)

    train_data = DenoisingDataset("data/processed", "train", SIGMA)
    val_data = DenoisingDataset("data/processed", "val", SIGMA)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = UNetModel(learning_rate=LEARNING_RATE)

    checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/denoising_{SIGMA}_{DEVICE}",
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator=DEVICE,
        devices=1,
        callbacks=[checkpoint],
        logger=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_loader, val_loader)

    print(f"Best model saved to: {checkpoint.best_model_path}")
