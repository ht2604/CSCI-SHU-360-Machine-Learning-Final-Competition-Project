import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight


def get_latent_vector(model, x):
    # For linear probing, we need a flattened version of z
    z = model.encode(x)
    return z.reshape(z.size(0), -1)  # Flatten to [batch_size, latent_dim]


def plot_reconstructions(model, dataloader, device, num_images=8):
    model.eval()
    with torch.no_grad():
        x = next(iter(dataloader))[0].to(device)
        x_recon, z, _, _ = model(x)
        x = x.cpu().numpy()
        x_recon = x_recon.cpu().numpy()
        print(f"Latent bottleneck dimension: {z.flatten(start_dim=1).shape[1]}")

        plt.figure(figsize=(16, 4))
        for i in range(num_images):
            # Original
            plt.subplot(2, num_images, i+1)
            plt.imshow(x[i].transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)
            plt.axis('off')

            # Reconstruction
            plt.subplot(2, num_images, i+1+num_images)
            plt.imshow(x_recon[i].transpose(1, 2, 0))
            plt.axis('off')


def compute_reconstruction_mse(model, data_loader, device):
    model.eval()
    total_sse = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            x_recon, _, _, _ = model(images)
            mse = F.mse_loss(images, x_recon, reduction='sum')
            total_sse += mse.item()
    return total_sse / len(data_loader.dataset)


def extract_latent_features(model, data_loader, device):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            # Get the flattened latent representation
            features = model.get_latent_vector(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.vstack(all_features), np.concatenate(all_labels)


def linear_probing(train_features, train_labels, test_features, test_labels):
    print("Training linear classifier...")
    # classes for Pokemon types
    classifier = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        # n_jobs=-1,
        verbose=1
    )
    classifier.fit(train_features, train_labels)

    predictions = classifier.predict(train_features)
    accuracy_train = accuracy_score(train_labels, predictions)
    predictions = classifier.predict(test_features)
    accuracy_test = accuracy_score(test_labels, predictions)

    print(f"train accuracy: {accuracy_train:.4f}")
    print(f"test  accuracy: {accuracy_test:.4f}")
    return accuracy_test


def plot_imgs(imgs):
    num_images = len(imgs)
    plt.figure(figsize=(16, 4))
    for i in range(num_images):
        # Original
        plt.subplot(1, num_images, i+1)
        plt.imshow(imgs[i].transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)
        plt.axis('off')
