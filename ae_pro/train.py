# import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
# from tqdm import tqdm
import pytorch_lightning as pl

# from modules_vq import VQVAELightning
# from modules_baseline import VAELightning
# from modules_vq_gan import VQGANLightning
# from modules_vq_enhanced import VQVAEEnhancedLightning
# from modules_vq_pro import VQVAEProLightning
from modules_ae import AELightning
# from modules_ae_pro import EnhancedAELightning
from dataset import *
from utility import *

print("start")
train_dataset = CustomDataset("train.npz")
# train_dataset = ArgumentatedDataset("train.npz")

with open('label2type.txt', 'r') as f:
    label2type = eval(f.read())

class_counts = len(np.unique(train_dataset.labels))

# weights = compute_class_weight(class_weight="balanced", classes=class_counts, y=train_dataset.labels)
weights = np.ones(class_counts)


# Create dataloaders
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_size = 3300
# val_size = 1
train_size = len(train_dataset) - val_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=32)

# train_labels = train_dataset.labels[train_subset.indices]
# class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_dataset.labels), y=train_labels)
# sample_weights = class_weights[train_labels]
# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
# train_loader = DataLoader(train_subset, batch_size=128, num_workers=32, sampler=sampler)

val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=32)

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

# model = VQGANLightning(
#     input_channels=3,
#     latent_channels=8,
#     num_embeddings=1024,
#     adv_weight=1e-2,
#     generator_epoch=1,
#     discriminator_epoch=2,
#     lr=1e-4,
# )

# model = VQVAEEnhancedLightning(
#     input_channels=3,
#     latent_channels=8,
#     num_embeddings=1024,
#     lr=1e-3,
#     class_loss_weight=0.005,
#     class_weight=weights
# )
# checkpoint = "./lightning_logs/version_62_d/checkpoints/last.ckpt"
# model = VQVAEEnhancedLightning.load_from_checkpoint(checkpoint)
# model.hparams.lr = 1e-4
# model.save_hyperparameters()

# model = VQVAEProLightning(
#     recon_loss_type="mse",
#     # recon_loss_type="mixed",
#     class_loss_weight=0.005
# )

model = AELightning(
    latent_channels=8,
    lr=1e-3
)

# model = EnhancedAELightning(
#     input_channels=3, 
#     latent_channels=128, 
#     grad_weight=0.1, 
#     laplacian_weight=0.0, 
#     perceptual_weight=0.0
# )


trainer = pl.Trainer(
    max_epochs=500,
    accelerator='auto',
    enable_progress_bar=True,
    log_every_n_steps=41,
    # devices=1
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
