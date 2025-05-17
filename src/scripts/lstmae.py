import sys

sys.path.append("../classes")

import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
from LSTMAE import LSTMAutoencoder
from DataLoader import DataLoaderFactory

data_dir = "../../data/quickdraw_subset_np"

input_file_name = "train_images.npy"
label_file_name = "train_labels.npy"

inputs_path = os.path.join(data_dir, input_file_name)
labels_path = os.path.join(data_dir, label_file_name)

train_loader, val_loader = DataLoaderFactory.build(
    inputs_path, labels_path, val_ratio=0.1
)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

device = torch.device(device)
print(f"Using device: {device}")

model = LSTMAutoencoder(input_dim=28, hidden_dim=128, latent_dim=128, num_layers=2).to(
    device
)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train_model(
    train_loader=train_loader,
    steps=10000,
    optimizer=optimizer,
    device=device,
    log_freq=10,
)

model.plot_history()

model.eval()

un_trainedae = LSTMAutoencoder(
    input_dim=28, hidden_dim=128, latent_dim=128, num_layers=2
).to(device)

x, _ = val_loader.get_batch(3)
x = torch.from_numpy(x).to(device)
images = x / 255.0

n = images.shape[0]

for i in range(1, n + 1):

    plt.subplot(n, 3, 1 + (i - 1) * 3)
    plt.imshow(images[i - 1].to("cpu"), cmap="gray")
    plt.axis("off")
    plt.title("Original")

    with torch.no_grad():
        reconstructed, _ = un_trainedae(images)
    plt.subplot(n, 3, 2 + (i - 1) * 3)
    plt.imshow(reconstructed[i - 1].to("cpu").numpy(), cmap="gray")
    plt.axis("off")
    plt.title("Untrained")

    with torch.no_grad():
        reconstructed, _ = model(images)
    plt.subplot(n, 3, 3 + (i - 1) * 3)
    plt.imshow(reconstructed[i - 1].to("cpu").numpy(), cmap="gray")
    plt.axis("off")
    plt.title("Trained")

plt.tight_layout()
plt.show()
