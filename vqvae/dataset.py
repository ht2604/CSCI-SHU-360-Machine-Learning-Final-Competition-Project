import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class CustomDataset(Dataset):  # load from npz data file
    def __init__(self, npz_path):
        npz_data = np.load(npz_path)
        self.images = npz_data["images"]  # (N, 3, 128, 128) in np.uint8
        self.labels = npz_data["labels"]  # (N,) in np.int64
        assert self.images.shape[0] == self.labels.shape[0]
        print(f"{npz_path}: images shape {self.images.shape}, "
              f"labels shape {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]) / 255  # convert to [0, 1] range
        label = torch.tensor(self.labels[idx])
        return image, label


class ArgumentatedDataset(Dataset):  # load from npz data file
    def __init__(self, npz_path):
        npz_data = np.load(npz_path)
        self.images = npz_data["images"]  # (N, 3, 128, 128) in np.uint8
        self.labels = npz_data["labels"]  # (N,) in np.int64
        assert self.images.shape[0] == self.labels.shape[0]
        print(f"{npz_path}: images shape {self.images.shape}, "
              f"labels shape {self.labels.shape}")

        # augmentation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].transpose((1, 2, 0)) # (128, 128, 3)
        image = self.transform(image)  # Converts to tensor [0, 1]
        label = torch.tensor(self.labels[idx])
        return image, label
