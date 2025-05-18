import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
os.makedirs('vae_results', exist_ok=True)

# Data loading
def load_data():
    train_images = np.load('quickdraw_subset_np/train_images.npy')
    train_labels = np.load('quickdraw_subset_np/train_labels.npy')
    test_images = np.load('quickdraw_subset_np/test_images.npy')
    test_labels = np.load('quickdraw_subset_np/test_labels.npy')
    
    # Normalize images to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    # Convert to PyTorch tensors
    train_images_tensor = torch.FloatTensor(train_images).unsqueeze(1)  # Add channel dimension
    train_labels_tensor = torch.LongTensor(train_labels)
    test_images_tensor = torch.FloatTensor(test_images).unsqueeze(1)
    test_labels_tensor = torch.LongTensor(test_labels)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader, train_images.shape[1:]

# VAE Models

# 1. RNN Encoder + CNN Decoder
class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # GRU encoder
        self.gru = nn.GRU(input_dim[0], hidden_dim, batch_first=True, bidirectional=True)
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(hidden_dim * 2 * input_dim[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2 * input_dim[1], latent_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for RNN: (batch, channels, height, width) -> (batch, width, height)
        x = x.view(batch_size, self.input_dim[1], self.input_dim[0])
        
        # GRU forward pass
        output, _ = self.gru(x)
        
        # Flatten output
        output = output.contiguous().view(batch_size, -1)
        
        # Get mean and log variance
        mu = self.fc_mu(output)
        logvar = self.fc_logvar(output)
        
        return mu, logvar

class CNNDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(CNNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Initial feature map size after deconvolution
        initial_size = output_dim[0] // 4
        
        # Fully connected layer to hidden representation
        self.fc = nn.Linear(latent_dim, 128 * initial_size * initial_size)
        
        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)
        
        self.initial_size = initial_size
        
    def forward(self, z):
        # Fully connected layer
        h = F.relu(self.fc(z))
        
        # Reshape for deconvolution
        h = h.view(-1, 128, self.initial_size, self.initial_size)
        
        # Deconvolution layers
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        
        # Output layer (sigmoid for pixel values between 0 and 1)
        reconstructed = torch.sigmoid(self.deconv3(h))
        
        return reconstructed

# 2. CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size after convolutions
        conv_output_size = (input_dim[0] // 4) * (input_dim[1] // 4) * 128
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_size, latent_dim)
        
    def forward(self, x):
        # Convolutional layers with ReLU activation
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        
        # Flatten
        h = h.view(h.size(0), -1)
        
        # Get mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

# Complete VAE
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

# Add specific classes for RNNVAE and CNNVAE for compatibility with evaluate_vae_metrics.py
class RNNVAE(nn.Module):
    def __init__(self, latent_dim=32, input_dim=(28, 28), hidden_dim=256):
        super(RNNVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # Define encoder and decoder
        self.encoder = RNNEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = CNNDecoder(latent_dim, input_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

class CNNVAE(nn.Module):
    def __init__(self, latent_dim=32, input_dim=(28, 28)):
        super(CNNVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # Define encoder and decoder
        self.encoder = CNNEncoder(input_dim, latent_dim)
        self.decoder = CNNDecoder(latent_dim, input_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

# Conditional VAE
class ConditionalVAE(nn.Module):
    def __init__(self, encoder=None, decoder=None, num_classes=5, latent_dim=32, input_dim=(28, 28)):
        super(ConditionalVAE, self).__init__()
        
        # Create encoder and decoder if not provided
        if encoder is None:
            self.encoder = CNNEncoder(input_dim, latent_dim)
        else:
            self.encoder = encoder
            
        if decoder is None:
            self.decoder = CNNDecoder(latent_dim, input_dim)
        else:
            self.decoder = decoder
            
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embeddings for class labels
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x, labels):
        # Get class embeddings
        class_embed = self.class_embedding(labels)
        
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Add class embedding to latent vector
        z_conditioned = z + class_embed
        
        # Decode
        reconstructed = self.decoder(z_conditioned)
        
        return reconstructed, mu, logvar
    
    def generate(self, num_samples, label, device):
        """Generate samples for a specific class"""
        # Sample from prior
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # Get class embedding
        label_tensor = torch.full((num_samples,), label, dtype=torch.long).to(device)
        class_embed = self.class_embedding(label_tensor)
        
        # Add class embedding
        z_conditioned = z + class_embed
        
        # Decode
        samples = self.decoder(z_conditioned)
        
        return samples
        
    def decode(self, z, class_one_hot=None):
        """Decode latent vector with class conditioning"""
        # If class one-hot vector is provided, use it
        if class_one_hot is not None:
            # Convert one-hot to class index (assume batch dimension)
            if class_one_hot.shape[1] == self.num_classes:
                class_indices = torch.argmax(class_one_hot, dim=1)
                class_embed = self.class_embedding(class_indices)
                z_conditioned = z + class_embed
            else:
                z_conditioned = z
        else:
            # No conditioning, just use z as is
            z_conditioned = z
            
        # Decode
        return self.decoder(z_conditioned)

# VAE Loss Function
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (binary cross entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE, KLD, BCE + beta * KLD

# Training function
def train_vae(model, train_loader, optimizer, device, beta=1.0):
    model.train()
    train_loss = 0
    bce_loss_total = 0
    kld_loss_total = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        if isinstance(model, ConditionalVAE):
            reconstructed, mu, logvar = model(data, _)
        else:
            reconstructed, mu, logvar = model(data)
            
        bce, kld, loss = vae_loss(reconstructed, data, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        bce_loss_total += bce.item()
        kld_loss_total += kld.item()
        
    return train_loss / len(train_loader.dataset), bce_loss_total / len(train_loader.dataset), kld_loss_total / len(train_loader.dataset)

# Evaluation function
def evaluate_vae(model, test_loader, device, beta=1.0):
    model.eval()
    test_loss = 0
    bce_loss_total = 0
    kld_loss_total = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            if isinstance(model, ConditionalVAE):
                reconstructed, mu, logvar = model(data, _)
            else:
                reconstructed, mu, logvar = model(data)
                
            bce, kld, loss = vae_loss(reconstructed, data, mu, logvar, beta)
            
            test_loss += loss.item()
            bce_loss_total += bce.item()
            kld_loss_total += kld.item()
            
    return test_loss / len(test_loader.dataset), bce_loss_total / len(test_loader.dataset), kld_loss_total / len(test_loader.dataset)

# Function to generate and save samples
def generate_samples(model, num_samples, device, filename):
    model.eval()
    with torch.no_grad():
        # Sample latent vectors from normal distribution
        z = torch.randn(num_samples, model.decoder.latent_dim).to(device)
        
        # Generate samples
        samples = model.decoder(z)
        
        # Convert to numpy array
        samples = samples.cpu().numpy()
        
        # Plot samples
        fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
        for i, ax in enumerate(axes):
            ax.imshow(samples[i, 0], cmap='gray')
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(f'vae_results/{filename}')
        plt.close()
        
        return samples

# Function to generate samples from conditional VAE
def generate_conditional_samples(model, num_samples_per_class, device, filename):
    model.eval()
    num_classes = 5  # Based on the dataset (0: rabbit, 1: yoga, 2: hand, 3: snowman, 4: motorbike)
    
    with torch.no_grad():
        # Generate samples for each class
        all_samples = []
        all_labels = []
        
        for label in range(num_classes):
            samples = model.generate(num_samples_per_class, label, device)
            
            # Convert to numpy array
            samples_np = samples.cpu().numpy()
            all_samples.append(samples_np)
            all_labels.extend([label] * num_samples_per_class)
            
        # Plot samples
        fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=(2*num_samples_per_class, 2*num_classes))
        
        for i in range(num_classes):
            for j in range(num_samples_per_class):
                axes[i, j].imshow(all_samples[i][j, 0], cmap='gray')
                axes[i, j].axis('off')
                if j == 0:
                    class_names = ['rabbit', 'yoga', 'hand', 'snowman', 'motorbike']
                    axes[i, j].set_title(class_names[i], fontsize=10)
                
        plt.tight_layout()
        plt.savefig(f'vae_results/{filename}')
        plt.close()
        
        return np.concatenate(all_samples), np.array(all_labels)

# Load the pretrained classifier
def load_classifier():
    try:
        with open('classifier_model.pkl', 'rb') as f:
            classifier = pickle.load(f)
        return classifier
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return None

# Main training and evaluation loop
def main():
    # Load data
    train_loader, test_loader, input_dim = load_data()
    print(f"Input dimensions: {input_dim}")
    
    # Hyperparameters
    latent_dim = 32
    hidden_dim = 256
    num_epochs = 50
    beta = 1.0  # Weight for KL divergence term
    
    # 1. RNN Encoder + CNN Decoder VAE
    print("Training RNN Encoder + CNN Decoder VAE...")
    rnn_encoder = RNNEncoder(input_dim, hidden_dim, latent_dim).to(device)
    cnn_decoder = CNNDecoder(latent_dim, input_dim).to(device)
    rnn_cnn_vae = VAE(rnn_encoder, cnn_decoder).to(device)
    
    optimizer = optim.Adam(rnn_cnn_vae.parameters(), lr=1e-2)
    
    # For tracking losses
    train_losses_rnn = []
    bce_losses_rnn = []
    kld_losses_rnn = []
    
    for epoch in range(num_epochs):
        train_loss, bce_loss, kld_loss = train_vae(rnn_cnn_vae, train_loader, optimizer, device, beta)
        train_losses_rnn.append(train_loss)
        bce_losses_rnn.append(bce_loss)
        kld_losses_rnn.append(kld_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, BCE: {bce_loss:.4f}, KLD: {kld_loss:.4f}")
    
    # 2. CNN Encoder + CNN Decoder VAE
    print("\nTraining CNN Encoder + CNN Decoder VAE...")
    cnn_encoder = CNNEncoder(input_dim, latent_dim).to(device)
    cnn_cnn_vae = VAE(cnn_encoder, cnn_decoder).to(device)
    
    optimizer = optim.Adam(cnn_cnn_vae.parameters(), lr=1e-3)
    
    # For tracking losses
    train_losses_cnn = []
    bce_losses_cnn = []
    kld_losses_cnn = []
    
    for epoch in range(num_epochs):
        train_loss, bce_loss, kld_loss = train_vae(cnn_cnn_vae, train_loader, optimizer, device, beta)
        train_losses_cnn.append(train_loss)
        bce_losses_cnn.append(bce_loss)
        kld_losses_cnn.append(kld_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, BCE: {bce_loss:.4f}, KLD: {kld_loss:.4f}")
    
    # 3. Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses_rnn, label='RNN-CNN VAE')
    plt.plot(range(1, num_epochs+1), train_losses_cnn, label='CNN-CNN VAE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('vae_results/loss_comparison.png')
    plt.close()
    
    # Plot BCE and KLD components
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), bce_losses_rnn, label='RNN-CNN BCE')
    plt.plot(range(1, num_epochs+1), kld_losses_rnn, label='RNN-CNN KLD')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Component')
    plt.title('RNN-CNN VAE Loss Components')
    plt.legend()
    plt.grid(True)
    plt.savefig('vae_results/rnn_cnn_loss_components.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), bce_losses_cnn, label='CNN-CNN BCE')
    plt.plot(range(1, num_epochs+1), kld_losses_cnn, label='CNN-CNN KLD')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Component')
    plt.title('CNN-CNN VAE Loss Components')
    plt.legend()
    plt.grid(True)
    plt.savefig('vae_results/cnn_cnn_loss_components.png')
    plt.close()
    
    # 4. Generate samples from both models
    print("\nGenerating samples...")
    rnn_samples = generate_samples(rnn_cnn_vae, 10, device, 'rnn_cnn_samples.png')
    cnn_samples = generate_samples(cnn_cnn_vae, 10, device, 'cnn_cnn_samples.png')
    
    # 5. Determine best model based on final loss
    best_model = rnn_cnn_vae if train_losses_rnn[-1] < train_losses_cnn[-1] else cnn_cnn_vae
    best_encoder = rnn_encoder if train_losses_rnn[-1] < train_losses_cnn[-1] else cnn_encoder
    best_model_name = "RNN-CNN" if train_losses_rnn[-1] < train_losses_cnn[-1] else "CNN-CNN"
    
    print(f"\nBest performing model: {best_model_name} VAE")
    
    # 6. Create and train Conditional VAE with best model architecture
    print("\nTraining Conditional VAE...")
    num_classes = 5  # rabbit, yoga, hand, snowman, motorbike
    
    conditional_vae = ConditionalVAE(
        best_encoder, 
        cnn_decoder, 
        num_classes, 
        latent_dim
    ).to(device)
    
    optimizer = optim.Adam(conditional_vae.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        model = conditional_vae
        model.train()
        train_loss = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(data, labels)
            
            bce, kld, loss = vae_loss(reconstructed, data, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader.dataset)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")
    
    # 7. Generate conditional samples (5 samples for each of rabbit, yoga, and snowman)
    print("\nGenerating conditional samples...")
    samples, labels = generate_conditional_samples(
        conditional_vae, 5, device, 'conditional_samples.png'
    )
    
    # 8. Evaluate with classifier from Assignment 1
    print("\nEvaluating samples with pretrained classifier...")
    try:
        classifier = load_classifier()
        
        if classifier is None:
            print("Could not load classifier model. Skipping evaluation.")
        else:
            # Evaluate specifically rabbit, yoga, and snowman samples
            target_classes = [0, 1, 3]  # rabbit, yoga, snowman
            for class_idx in target_classes:
                class_name = ['rabbit', 'yoga', 'hand', 'snowman', 'motorbike'][class_idx]
                class_samples = samples[labels == class_idx]
                class_labels = np.full(class_samples.shape[0], class_idx)
                
                # Reshape samples for classifier
                class_samples_flat = class_samples.reshape(class_samples.shape[0], -1)
                
                # Get predictions and confidence
                predictions = classifier.predict(class_samples_flat)
                probs = classifier.predict_proba(class_samples_flat)
                confidence = np.max(probs, axis=1)
                
                accuracy = np.mean(predictions == class_labels)
                avg_confidence = np.mean(confidence)
                
                print(f"\nClass: {class_name}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Average confidence: {avg_confidence:.4f}")
                
                # Print individual predictions and confidence
                for i in range(min(5, len(predictions))):
                    pred_class = ['rabbit', 'yoga', 'hand', 'snowman', 'motorbike'][predictions[i]]
                    print(f"Sample {i+1}: Predicted as {pred_class} with confidence {confidence[i]:.4f}")
    
    except Exception as e:
        print(f"Error evaluating with classifier: {e}")
        
    print("\nTraining and evaluation complete. Results saved in 'vae_results' directory.")

if __name__ == "__main__":
    main() 