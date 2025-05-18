import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np


class ImageDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.labels[idx]

        image = torch.tensor(image, dtype=torch.float32) / 255.0
        label = torch.tensor(label, dtype=torch.long)

        return image, label


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # Flatten 28x28 to 784
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
        )

    def forward(self, x):
        return self.model(x)
