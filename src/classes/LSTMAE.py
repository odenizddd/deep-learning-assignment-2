import torch
import torch.nn as nn
from DataLoader import DataLoader


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.history = {}

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        _, (h_n, _) = self.encoder(x)
        latent = self.latent(h_n[-1])  # take last layer's hidden state

        # expand for each time step
        hidden = self.decoder_input(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        output, _ = self.decoder(hidden)
        return output, latent

    def parameters(self):
        return (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.latent.parameters())
            + list(self.decoder_input.parameters())
        )

    def train_model(
        self,
        train_loader: DataLoader,
        steps: int,
        optimizer: torch.optim.Optimizer,
        device,
        log_freq: int = 5,
    ):
        self.train()

        for step in range(steps):
            x, _ = train_loader.get_batch(256)
            x = torch.from_numpy(x).to(device)
            x = x / 255.0

            output, _ = self.forward(x)
            loss = torch.nn.functional.mse_loss(output, x)

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
