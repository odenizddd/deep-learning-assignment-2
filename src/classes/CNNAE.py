import torch
from DataLoader import DataLoader


class CNNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 14x14
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 7x7
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(32 * 7 * 7, 64)  # latent space

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x.view(x.size(0), -1))


class CNNDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 32 * 7 * 7)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                32, 16, 3, stride=2, output_padding=1, padding=1
            ),  # 14x14
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                16, 1, 3, stride=2, output_padding=1, padding=1
            ),  # 28x28
            torch.nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 32, 7, 7)
        return self.decoder(x)


class ConvolutionalAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = CNNDecoder()

        self.history = {}

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def train_model(
        self,
        train_loader: DataLoader,
        steps: int,
        optimizer: torch.optim.Optimizer,
        log_freq: int = 5,
    ):

        self.train()
        for step in range(steps):
            x, _ = train_loader.get_batch(256)
            x = torch.from_numpy(x).unsqueeze(1)
            x = x / 255.0

            x_hat = self.forward(x)
            loss = torch.nn.functional.mse_loss(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.history[step] = {"loss": loss.item()}
            if step % log_freq == 0:
                print(f"Step {step}")
                print(f"Loss: {loss.item()}")

    def plot_history(self):
        import matplotlib.pyplot as plt

        steps = list(self.history.keys())
        losses = [self.history[step]["loss"] for step in steps]

        plt.plot(steps, losses)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()
