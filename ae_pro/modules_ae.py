import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


class AE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=4, num_embeddings=512):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU()
        )

        # Pre-quantization convolution
        self.pre_quant_conv = nn.Conv2d(256, latent_channels, 1)

        # Decoder
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def preprocess(self, x):
        return 2 * x - 1  # [0, 1] -> [-1, 1]

    def encode(self, x, return_multiple=False):
        x = self.preprocess(x)
        h = self.encoder(x)
        z = self.pre_quant_conv(h)
        return z

    def decode(self, z_q):
        h = self.post_quant_conv(z_q)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, None, None

    def get_latent_vector(self, x):
        assert not self.training
        z_q = self.encode(x)
        return z_q.reshape(z_q.size(0), -1)

class AELightning(pl.LightningModule):
    def __init__(self, input_channels=3, latent_channels=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = AE(input_channels, latent_channels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, z, _, _ = self.model(x)
        recon_loss = F.mse_loss(x, x_recon)

        self.log_dict({
            "train_recon": recon_loss,
        }, prog_bar=True)
        return recon_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, z, _, _ = self.model(x)
        recon_loss = F.mse_loss(x, x_recon)

        self.log_dict({
            "val_recon": recon_loss,
        })

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=20,
            eta_min=1e-4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def configure_callbacks(self):
        return [
            ModelCheckpoint(
                monitor="val_recon",
                filename="best",
                save_top_k=1,
                mode="min",
                save_last=True
            ),
            LearningRateMonitor(logging_interval='step'),
            # EarlyStopping(monitor="val_recon", mode="min", patience=10)
        ]
