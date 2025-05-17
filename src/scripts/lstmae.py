import sys

sys.path.append("../classes")

import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
from LSTMAE import LSTMAutoencoder
from DataLoader import DataLoaderFactory
import numpy as np
from sklearn.manifold import TSNE

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

print("Plotting t-SNE of Encoded Representations...")

model.eval()

embeddings = []
labels = []

n = train_loader.num_samples
batch_size = 256

with torch.no_grad():
    for i in range(n // batch_size):
        x, y = train_loader.get_batch(batch_size)
        x = torch.from_numpy(x).to(device)
        images = x / 255.0

        _, latent = model(images)
        embeddings.append(latent.cpu().numpy())
        labels.append(y)


embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)

tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab10", s=5)
plt.colorbar()
plt.title("2D t-SNE of Encoded Representations")
plt.show()
