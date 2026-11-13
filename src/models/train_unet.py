import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from src.models.unet import UNet
from src.data.dataset import SuperResolutionDataset


LR_SIZE = 32
BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 0.0001
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"



class UNetModel(L.LightningModule):
    def __init__(self, learning_rate=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.unet = UNet()
        self.loss_function = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, images):
        return self.unet(images)

    def training_step(self, batch, batch_idx):
        prediction = self(batch["input"])
        loss = self.loss_function(prediction, batch["target"])
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=prediction.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        prediction = self(batch["input"]).clamp(0, 1)
        target = batch["target"]
        loss = self.loss_function(prediction, target)

        mse = ((prediction - target) ** 2).mean(dim=(1, 2, 3))
        psnr = (-10 * torch.log10(mse)).mean()

        self.log("val_loss", loss, prog_bar=True, batch_size=prediction.shape[0])
        self.log("val_psnr", psnr, prog_bar=True, batch_size=prediction.shape[0])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train():
    L.seed_everything(42, verbose=False)

    train_data = SuperResolutionDataset("data/processed", "train", LR_SIZE)
    val_data = SuperResolutionDataset("data/processed", "val", LR_SIZE)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = UNetModel(learning_rate=LEARNING_RATE)

    checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/sr_{LR_SIZE}_{DEVICE}",
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
