import torch
import torch.nn as nn
import torch.nn.functional as F


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

        avg_probs = torch.histc(indices.float(), bins=self.num_embeddings, min=0, max=self.num_embeddings-1)
        perplexity = torch.exp(-torch.sum((avg_probs / (B*H*W)) * torch.log(avg_probs / (B*H*W) + 1e-10)))
        return z_q, vq_loss, indices, perplexity


class Model(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_embeddings=1024):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=2, padding=1),  # 64 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # 256 x 32 x 32
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

    def preprocess(self, x):
        return 2 * x - 1  # [0, 1] -> [-1, 1]

    def encode(self, x, return_multiple=False):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.pre_quant_conv(h)
        z_q, vq_loss, indices, perplexity = self.codebook(h)
        if return_multiple:
            return z_q, vq_loss, indices, perplexity
        return z_q

    def decode(self, z_q):
        h = self.post_quant_conv(z_q)
        return self.decoder(h)

    def forward(self, x):
        z_q, vq_loss, _, perplexity = self.encode(x, True)
        x_recon = self.decode(z_q)
        return x_recon, z_q, vq_loss, perplexity

    def get_latent_vector(self, x):
        assert not self.training
        z_q, _, _, _ = self.encode(x, True)
        return z_q.reshape(z_q.size(0), -1)