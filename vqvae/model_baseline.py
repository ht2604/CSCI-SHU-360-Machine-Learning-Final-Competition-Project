import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8):
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
