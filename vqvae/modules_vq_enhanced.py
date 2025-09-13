import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

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


class VQVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=4, num_embeddings=512):
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
        #更深的编码器结构
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),  # 64x64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128x32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 256x16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),  # 512x16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )



        # Pre-quantization convolution
        self.pre_quant_conv = nn.Conv2d(256, latent_channels, 1)

        # Codebook
        self.codebook = Codebook(num_embeddings, latent_channels)

        # # Decoder
        # self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(256, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, input_channels, 4, stride=2, padding=1),
        #     nn.Sigmoid()
        # )

        # 更深的解码器结构
        # 对应的解码器
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(latent_channels, 512, 3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 256x32x32
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 128x64x64
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x128x128
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(64, input_channels, 3, padding=1),
        #     nn.Sigmoid()   )

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


class VQVAEEnhancedLightning(pl.LightningModule):
    def __init__(self, input_channels=3, latent_channels=8, num_embeddings=1024, lr=1e-3, num_classes=170, class_loss_weight=0.1, class_weight=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = VQVAE(input_channels, latent_channels, num_embeddings)

        dummy = torch.zeros(1, input_channels, 128, 128)
        z_q, _, _, _ = self.model.encode(dummy, return_multiple=True)
        self.latent_dim = z_q.reshape(z_q.size(0), -1).size(1)
        # self.classifier = nn.Linear(self.latent_dim, num_classes)
        self.classifier = nn.Sequential(
                nn.Linear(self.latent_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, num_classes)
            )
        self.register_buffer("ce_weights", torch.from_numpy(class_weight).float())

        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, x, x_recon, vq_loss):
        recon_loss = F.mse_loss(x, x_recon)
        total_loss = recon_loss + vq_loss
        return total_loss, recon_loss, vq_loss

    def _classification_loss(self, z_q, label):
        z_q_flat = z_q.reshape(z_q.size(0), -1)
        logits = self.classifier(z_q_flat)
        # 标签平滑
        cls_loss = F.cross_entropy(logits, label, weight=self.ce_weights, label_smoothing=0.1)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == label).float().mean()
        return cls_loss, acc
    
    #loss function 优化
    def _calculate_loss(self, x, x_recon, vq_loss):
    # 使用L1损失和SSIM结合
        l1_loss = F.l1_loss(x, x_recon)
        device = x.device  # 获取输入张量所在的设备
        ssim_metric = SSIM().to(device)  # 将 SSIM 移动到与输入相同的设备
    
        # 使用L1损失和SSIM结合
        l1_loss = F.l1_loss(x, x_recon)
    
        # 计算 SSIM 损失
        ssim_loss = 1 - ssim_metric(x, x_recon)  # 使用SSIM计算损失
        
        # 加权组合
        recon_loss = 0.85 * l1_loss + 0.15 * ssim_loss
        total_loss = recon_loss + vq_loss
        return total_loss, recon_loss, vq_loss

    def training_step(self, batch, batch_idx):
        x, label = batch
        x_recon, z_q, vq_loss, perplexity = self.model(x)
        loss, recon_loss, vq_loss = self._calculate_loss(x, x_recon, vq_loss)
        cls_loss, acc = self._classification_loss(z_q, label)

        total_loss = loss + cls_loss * self.hparams.class_loss_weight
        score = recon_loss / (acc + 1e-8)

         # 渐进式权重调整
        # current_epoch = self.current_epoch
        # cls_weight = min(self.hparams.class_loss_weight * (current_epoch / 10), self.hparams.class_loss_weight)
        
        # loss, recon_loss, vq_loss = self._calculate_loss(x, x_recon, vq_loss)
        # cls_loss, acc = self._classification_loss(z_q, label)
        
        # total_loss = loss + cls_weight * cls_loss
        # score = recon_loss / (acc + 1e-8)
        
        # # 添加梯度裁剪
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        self.log_dict({
            "train_loss": total_loss,
            "train_recon": recon_loss,
            "train_vq_loss": vq_loss,
            "train_perplexity": perplexity,
            "train_cls_loss": cls_loss,
            "train_acc": acc,
            "train_score": score,
        }, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        batch_size = x.size(0)

        x_recon, z_q, vq_loss, perplexity = self.model(x)

        rec_sum = F.mse_loss(x, x_recon, reduction="sum")
        cls_loss, acc = self._classification_loss(z_q, label)

        self.validation_step_outputs.append({
            "recon_loss_sum": rec_sum,
            "acc_sum": acc * batch_size,
            "batch_size": batch_size
        })

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        total_samples = sum(out["batch_size"] for out in outputs)
        acc = sum(out["acc_sum"] for out in outputs) / total_samples
        recon_loss = sum(out["recon_loss_sum"] for out in outputs) / total_samples
        score = recon_loss / (acc + 1e-8)
        self.log_dict({
            "val_recon_sum": recon_loss,
            "val_acc": acc,
            "val_score": score,
        }, prog_bar=True)
        self.validation_step_outputs.clear()

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
                monitor="val_recon_sum",
                filename="best",
                save_top_k=1,
                mode="min",
                save_last=True
            ),
            LearningRateMonitor(logging_interval='step'),
            # EarlyStopping(monitor="val_recon_sum", mode="min", patience=20)
        ]
