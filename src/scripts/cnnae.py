import sys

sys.path.append("../classes")


import os
import torch
import matplotlib.pyplot as plt
from CNNAE import ConvolutionalAE
from DataLoader import DataLoaderFactory


data_dir = "../../data/quickdraw_subset_np"

input_file_name = "train_images.npy"
label_file_name = "train_labels.npy"

inputs_path = os.path.join(data_dir, input_file_name)
labels_path = os.path.join(data_dir, label_file_name)

train_loader, val_loader = DataLoaderFactory.build(
    inputs_path, labels_path, val_ratio=0.1
)

ae = ConvolutionalAE()

optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)

ae.train_model(train_loader=train_loader, steps=1000, optimizer=optimizer)

ae.plot_history()

un_trainedae = ConvolutionalAE()

images, _ = train_loader.get_batch(3)
x = torch.from_numpy(images).unsqueeze(1)
x = x / 255.0

n = images.shape[0]


for i in range(1, n + 1):

    plt.subplot(n, 3, 1 + (i - 1) * 3)
    plt.imshow(images[i - 1], cmap="gray")
    plt.axis("off")
    plt.title("Original")

    x_hat = un_trainedae(x)[i - 1].squeeze(0).detach().numpy()
    plt.subplot(n, 3, 2 + (i - 1) * 3)
    plt.imshow(x_hat, cmap="gray")
    plt.axis("off")
    plt.title("Untrained")

    x_hat = ae(x)[i - 1].squeeze(0).detach().numpy()
    plt.subplot(n, 3, 3 + (i - 1) * 3)
    plt.imshow(x_hat, cmap="gray")
    plt.axis("off")
    plt.title("Trained")

plt.tight_layout()
plt.show()
