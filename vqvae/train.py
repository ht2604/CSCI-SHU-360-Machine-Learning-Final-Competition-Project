import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl

from modules_vq import VQVAELightning
from modules_baseline import VAELightning
from modules_vq_enhanced import VQVAEEnhancedLightning
from modules_vq_gan import VQGANLightning
from dataset import *
from utility import *

train_dataset = CustomDataset("train.npz")
# train_dataset = ArgumentatedDataset("train.npz")

with open('label2type.txt', 'r') as f:
    label2type = eval(f.read())

weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_dataset.labels), y=train_dataset.labels)

# Create dataloaders
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_size = 3300
train_size = len(train_dataset) - val_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=0)

# sample data batch
images, labels = next(iter(train_loader))
print(f"images shape: {images.shape}")
print(f"labels shape: {labels.shape}")
print(f"image range:", images.min(), images.max())

device = 'cuda'

# pl.seed_everything(1234)

# model = VAELightning(
#     input_channels=3,
#     latent_channels=8,
#     lr=1e-3,
#     kl_weight=0.1
# )

# model = VQVAELightning(
#     input_channels=3,
#     latent_channels=8,
#     num_embeddings=1024,
#     lr=1e-3,
# )

# model = VQVAEEnhancedLightning(
#     input_channels=3,
#     latent_channels=8,
#     num_embeddings=1024,
#     lr=1e-3,
#     class_loss_weight=0.005,
#     class_weight=weights
# )
checkpoint = "./lightning_logs/version_62/checkpoints/last.ckpt"
model = VQVAEEnhancedLightning.load_from_checkpoint(checkpoint)

# model = VQGANLightning(
#     input_channels=3,
#     latent_channels=4,
#     num_embeddings=512,
# )

trainer = pl.Trainer(
    max_epochs=1000,
    accelerator='auto',
    enable_progress_bar=True,
)

trainer.fit(model, train_loader, val_loader)

log_dir = trainer.log_dir
model.to(device)

plot_reconstructions(model, train_loader, device, num_images=8)
plt.savefig(f"{log_dir}/recon.png", dpi=300)
plt.close()

train_mse = compute_reconstruction_mse(model, train_loader, device)
val_mse = compute_reconstruction_mse(model, val_loader, device)
print(f"train MSE: {train_mse:.3e}")
print(f"val MSE: {val_mse:.3e}")
# train MSE: 1.592e+02
# val MSE: 1.584e+02

with open(f"{log_dir}/summary.txt", 'w') as f:
    print(f"train MSE: {train_mse:.3e}", file=f)
    print(f"val MSE: {val_mse:.3e}", file=f)
