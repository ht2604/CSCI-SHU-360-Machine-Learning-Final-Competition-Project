import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=4):
        super().__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # 64 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # 128 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),            # 256 x 32 x 32
            nn.ReLU()
        )

        # --- Quantization conv ---
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, kernel_size=1)    # 4+4 channels

        # --- Decoder ---
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),            # 128 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 64 x 64 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),  # 3 x 128 x 128
            nn.Sigmoid()  # Predict within value range [0, 1]
        )

    def preprocess(self, x):
        return 2 * x - 1  # Value range: [0, 1] => [-1, 1]

    def encode(self, x, return_multiple=False):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)  # Split along channel (4+4)
        if self.training:
            logvar = logvar.clamp(-30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
        else:
            z = mean
        if return_multiple:
            return z, mean, logvar
        return z

    def decode(self, z):
        h = self.post_quant_conv(z)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        z, mean, logvar = self.encode(x, True)
        x_recon = self.decode(z)
        return x_recon, z, mean, logvar

    def get_latent_vector(self, x):
        # For linear probing, we need a flattened version of z
        assert not self.training
        z, mean, logvar = self.encode(x, True)
        return z.reshape(z.size(0), -1)  # Flatten to [batch_size, latent_dim]


class VAELightning(pl.LightningModule):
    def __init__(self, input_channels=3, latent_channels=8, lr=1e-3, kl_weight=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConvVAE(input_channels, latent_channels)

    def forward(self, x):
        return self.model(x)

    def vae_loss(self, x, x_recon, mean, logvar):
        recon_loss = F.mse_loss(x, x_recon, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + self.hparams.kl_weight * kl_loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, _, mean, logvar = self.model(x)
        loss, recon_loss, kl_loss = self.vae_loss(x, x_recon, mean, logvar)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_recon', recon_loss, prog_bar=True)
        self.log('train_kl', kl_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, _, mean, logvar = self.model(x)
        loss, recon_loss, kl_loss = self.vae_loss(x, x_recon, mean, logvar)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_recon', recon_loss)
        self.log('val_kl', kl_loss)

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
                monitor="val_loss",
                filename="best",
                save_top_k=1,
                mode="min",
                save_last=True
            ),
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(monitor="val_loss", mode="min", patience=10)
        ]
