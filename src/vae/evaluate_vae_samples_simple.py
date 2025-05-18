import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image

# Create output directory
os.makedirs('vae_evaluation', exist_ok=True)

# Define a simple MLP classifier
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

# Load trained model if available, otherwise train a new one
def get_classifier():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = MLP().to(device)
    
    # Check if we have a trained model
    if os.path.exists('classifier_model.pt'):
        print("Loading pre-trained PyTorch model...")
        model.load_state_dict(torch.load('classifier_model.pt', map_location=device))
        model.eval()
    else:
        print("No pre-trained model found, training a new one...")
        
        # Load training data
        try:
            train_images = np.load('quickdraw_subset_np/train_images.npy')
            train_labels = np.load('quickdraw_subset_np/train_labels.npy')
            
            # Normalize and convert to tensors
            train_images = train_images.astype(np.float32) / 255.0
            train_labels = train_labels.astype(np.int64)
            
            # Create TensorDataset and DataLoader
            train_dataset = TensorDataset(
                torch.from_numpy(train_images), 
                torch.from_numpy(train_labels)
            )
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            
            # Train the model
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # Train for a few epochs
            num_epochs = 3
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    if i % 100 == 99:
                        print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.4f}')
                        running_loss = 0.0
            
            # Save the model
            torch.save(model.state_dict(), 'classifier_model.pt')
            print("Model trained and saved.")
            
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    return model, device

# Function to evaluate sample images
def evaluate_samples(model, device, samples, class_names):
    """Evaluate the samples using the model and return predictions with confidence"""
    model.eval()
    
    # Convert to torch tensor
    if isinstance(samples, np.ndarray):
        samples_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
    else:
        samples_tensor = samples.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(samples_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
        
        # Convert to numpy for easier handling
        predictions = predicted.cpu().numpy()
        probabilities = probs.cpu().numpy()
        confidence = np.max(probabilities, axis=1)
    
    return predictions, confidence, probabilities

# Main function to evaluate VAE samples
def evaluate_vae_samples():
    print("Evaluating VAE-generated samples...")
    
    # Class names
    class_names = ['rabbit', 'yoga', 'hand', 'snowman', 'motorbike']
    
    # Target classes to evaluate
    target_classes = [0, 1, 3]  # rabbit, yoga, snowman
    
    # Get or train classifier
    classifier_result = get_classifier()
    if classifier_result is None:
        print("Failed to obtain classifier. Exiting.")
        return
    
    model, device = classifier_result
    
    # Check if VAE results exist
    samples_path = 'vae_results/conditional_samples.png'
    if not os.path.exists(samples_path):
        print(f"Could not find VAE samples at {samples_path}. Please make sure VAE has been trained.")
        return
    
    # Load the conditional VAE image for visualization
    cond_samples_img = Image.open(samples_path)
    plt.figure(figsize=(10, 12))
    plt.imshow(np.array(cond_samples_img))
    plt.axis('off')
    plt.title('Conditional VAE Samples')
    plt.savefig('vae_evaluation/conditional_samples_original.png')
    plt.close()
    
    # Use test images as samples for our classifier
    print("Loading test images as samples...")
    test_images = np.load('quickdraw_subset_np/test_images.npy')
    test_labels = np.load('quickdraw_subset_np/test_labels.npy')
    
    # Create figure for visualizing results
    fig, axs = plt.subplots(len(target_classes), 5, figsize=(15, 9))
    plt.suptitle("Test Samples with Classification Results", fontsize=16)
    
    # Evaluate each target class
    for i, class_idx in enumerate(target_classes):
        class_name = class_names[class_idx]
        print(f"\nEvaluating class: {class_name}")
        
        # Get samples from this class
        class_samples_indices = np.where(test_labels == class_idx)[0]
        if len(class_samples_indices) >= 5:
            sample_indices = class_samples_indices[:5]
        else:
            print(f"Not enough samples for class {class_name}, using available {len(class_samples_indices)}")
            sample_indices = class_samples_indices
        
        selected_samples = test_images[sample_indices] / 255.0  # Normalize
        
        # Evaluate the samples
        predictions, confidence, _ = evaluate_samples(model, device, selected_samples, class_names)
        
        # Calculate accuracy
        expected_labels = np.full(len(sample_indices), class_idx)
        accuracy = np.mean(predictions == expected_labels)
        avg_confidence = np.mean(confidence)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Average confidence: {avg_confidence:.4f}")
        
        # Plot samples and predictions
        for j in range(5):
            if j < len(sample_indices):
                pred_idx = predictions[j]
                pred_class = class_names[pred_idx]
                conf = confidence[j]
                
                # Plot sample
                axs[i, j].imshow(selected_samples[j], cmap='gray')
                color = 'green' if pred_idx == class_idx else 'red'
                axs[i, j].set_title(f"Pred: {pred_class}\nConf: {conf:.2f}", color=color)
                axs[i, j].axis('off')
                
                print(f"  Sample {j+1}: Predicted as {pred_class} with confidence {conf:.4f}")
            else:
                axs[i, j].axis('off')
    
    # Add row labels
    for i, class_idx in enumerate(target_classes):
        axs[i, 0].set_ylabel(class_names[class_idx], fontsize=12, rotation=90, labelpad=15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('vae_evaluation/classifier_evaluation.png')
    plt.close()
    
    print("\nEvaluation complete. Results saved to 'vae_evaluation' directory.")
    
    # Bonus: If we had actual VAE samples as .npy files, we could evaluate them directly
    # For now, just indicate how it would be done
    print("\nNote: For complete evaluation of VAE samples:")
    print("1. Save VAE-generated samples as .npy files during VAE training")
    print("2. Load those files here and run them through the classifier")
    print("3. Analyze how well the classifier recognizes the generated samples")

if __name__ == "__main__":
    evaluate_vae_samples() 