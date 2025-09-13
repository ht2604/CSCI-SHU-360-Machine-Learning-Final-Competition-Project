import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torchvision import models


class Codebook(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        B, D, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous()  # [B, H, W, D]
        z_e_flat = z_e_flat.view(-1, D)  # [B*H*W, D]

        distances = (torch.sum(z_e_flat**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(z_e_flat, self.embedding.weight.t()))

        indices = torch.argmin(distances, dim=1)
        z_q_flat = self.embedding(indices)  # [B*H*W, D]

        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        commitment_loss = F.mse_loss(z_e.detach(), z_q)
        codebook_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        z_q = z_e + (z_q - z_e).detach()

        return z_q, vq_loss, indices


class VQVAE(nn.Module):
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

        # Codebook
        self.codebook = Codebook(num_embeddings, latent_channels)

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

    def encode(self, x, return_multiple=False):
        h = self.encoder(x)
        h = self.pre_quant_conv(h)
        z_q, vq_loss, indices = self.codebook(h)
        if return_multiple:
            return z_q, vq_loss, indices
        return z_q

    def decode(self, z_q):
        h = self.post_quant_conv(z_q)
        return self.decoder(h)

    def forward(self, x):
        z_q, vq_loss, indices = self.encode(x, True)
        x_recon = self.decode(z_q)
        return x_recon, z_q, vq_loss, indices

    def get_latent_vector(self, x):
        assert not self.training
        z_q, _, _, _ = self.encode(x, True)
        return z_q.reshape(z_q.size(0), -1)


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 1, 1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


class VQGANLightning(pl.LightningModule):
    def __init__(self, input_channels=3, latent_channels=4, num_embeddings=512,
                 lr=1e-4, adv_weight=1.0, use_perceptual=False):
        super().__init__()
        self.save_hyperparameters()

        self.model = VQVAE(input_channels, latent_channels, num_embeddings)
        self.model.load_state_dict(torch.load("./submit/version_24_best.pt"))
        self.discriminator = Discriminator(input_channels)

        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, x, x_recon, vq_loss):
        # Reconstruction loss
        recon_loss = F.mse_loss(x, x_recon)


        # Adversarial loss
        fake_logits = self.discriminator(x_recon)
        gen_adv_loss = -torch.mean(fake_logits)

        total_loss = recon_loss + vq_loss + self.hparams.adv_weight * gen_adv_loss
        return total_loss, recon_loss, vq_loss, gen_adv_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        gen_opt, disc_opt = self.optimizers()

        # Generator update
        self.toggle_optimizer(gen_opt)
        x_recon, _, vq_loss, _ = self.model(x)
        total_loss, recon_loss, vq_loss, adv_loss = self._calculate_loss(x, x_recon, vq_loss)

        gen_opt.zero_grad()
        self.manual_backward(total_loss)
        gen_opt.step()
        self.untoggle_optimizer(gen_opt)

        log_dict = {
            "train_loss": total_loss,
            "train_recon": recon_loss,
            "train_vq": vq_loss,
            "train_adv": adv_loss
        }
        if self.perceptual_loss:
            log_dict["train_perceptual"] = recon_loss - F.mse_loss(x, x_recon)
        self.log_dict(log_dict, prog_bar=True)

        # Discriminator update
        self.toggle_optimizer(disc_opt)
        with torch.no_grad():
            x_recon, *_ = self.model(x)

        real_logits = self.discriminator(x)
        fake_logits = self.discriminator(x_recon.detach())

        # Hinge loss
        d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
        d_loss = d_loss / 2

        disc_opt.zero_grad()
        self.manual_backward(d_loss)
        disc_opt.step()
        self.untoggle_optimizer(disc_opt)

        self.log_dict({
            "train_d_loss": d_loss,
            "train_d_real": real_logits.mean(),
            "train_d_fake": fake_logits.mean()
        })

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, _, vq_loss, _ = self.model(x)
        total_loss, recon_loss, vq_loss, adv_loss = self._calculate_loss(x, x_recon, vq_loss)

        log_dict = {
            "val_loss": total_loss,
            "val_recon": recon_loss,
            "val_vq": vq_loss,
            "val_adv": adv_loss
        }
        if self.perceptual_loss:
            log_dict["val_perceptual"] = recon_loss - F.mse_loss(x, x_recon)
        self.log_dict(log_dict)

    def configure_optimizers(self):
        gen_opt = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        disc_opt = optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)

        gen_sch = optim.lr_scheduler.CosineAnnealingLR(gen_opt, T_max=20)
        disc_sch = optim.lr_scheduler.CosineAnnealingLR(disc_opt, T_max=20)

        return [gen_opt, disc_opt], [gen_sch, disc_sch]

    def configure_callbacks(self):
        return [
            ModelCheckpoint(monitor="val_loss", mode="min"),
            LearningRateMonitor(),
            EarlyStopping(monitor="val_loss", patience=10, mode="min")
        ]
