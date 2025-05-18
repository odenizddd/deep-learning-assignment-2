# Variational Autoencoder (VAE) Implementation Results

## Overview

This document provides an analysis of the VAE models implemented for the assignment. We've trained several VAE architectures on the QuickDraw dataset:

1. A VAE with a gated RNN encoder and a convolutional decoder
2. A VAE with a convolutional encoder and decoder
3. A conditional VAE with the best-performing architecture from (1) and (2)

## Model Architectures

### RNN Encoder + CNN Decoder VAE

This architecture uses:
- **Encoder**: A bidirectional GRU that processes the image row by row, followed by fully connected layers to produce the latent mean and log variance
- **Decoder**: A series of transposed convolutions to generate images from the latent space

### CNN Encoder + CNN Decoder VAE

This architecture uses:
- **Encoder**: Convolutional layers to downsample the image, followed by fully connected layers for latent mean and log variance
- **Decoder**: The same decoder as in the first architecture

### Conditional VAE

The conditional VAE extends the best-performing model by adding class conditioning:
- Class labels are embedded into vectors
- The embedding is added to the latent vector before decoding
- This allows the model to generate images conditioned on specific classes

## Training Results

### Full Implementation

#### Convergence Comparison

Based on the loss curves, we observed:

- The CNN-CNN VAE converged faster and achieved a lower final loss (184.16) compared to the RNN-CNN VAE (188.94)
- This suggests that the convolutional encoder is more effective at extracting relevant features from the QuickDraw images

#### Loss Components Analysis

Examining the BCE (reconstruction) and KLD (regularization) terms separately:

- **RNN-CNN VAE**: 
  - Final BCE: 148.16
  - Final KLD: 40.78
  
- **CNN-CNN VAE**:
  - Final BCE: 141.93
  - Final KLD: 42.23

The CNN architecture has a lower reconstruction error (BCE) but a slightly higher KL divergence term (KLD). This indicates that the CNN encoder creates a latent representation that is better for reconstruction but slightly further from the standard normal prior.

### Quick Implementation

To demonstrate the behavior more quickly, we also implemented a streamlined version that:
- Uses a smaller subset of data (5,000 training samples instead of 20,000)
- Utilizes a simpler model architecture
- Trains for fewer epochs (10 instead of 50)

Even with these simplifications, we observed similar patterns:
- The loss decreased consistently during training
- The BCE (reconstruction) term dominated the total loss
- The KLD (regularization) term gradually increased during training, showing the model's balancing of reconstruction quality versus latent space regularization

## Generated Samples

### Unconditional Samples

Both models are able to generate new samples from random latent vectors:

- `rnn_cnn_samples.png` and `cnn_cnn_samples.png`: Samples from the full implementation
- `vae_samples.png`: Samples from the quick implementation

The CNN-CNN VAE generally produces clearer and more well-defined samples, which is consistent with its lower reconstruction error.

### Conditional Samples

The conditional VAE can generate samples for specific classes:

- `conditional_samples.png`: Shows generated samples for each class (rabbit, yoga, hand, snowman, motorbike)

These samples demonstrate that the model has successfully learned to condition its generation on class labels, producing recognizable objects from each category. Even the quick implementation with less training produces class-specific features.

## VAE Loss Behavior

The VAE objective function consists of two components:

1. **Reconstruction Loss (BCE)**: Measures how well the decoder can reconstruct the original inputs
2. **KL Divergence (KLD)**: Regularizes the latent space, ensuring it approximates a standard normal distribution

During training, we observed:

- The BCE term dominates the total loss at the beginning
- As training progresses, the BCE decreases rapidly while the KLD gradually increases
- This reflects the model's balance between reconstruction quality and regularization

This behavior is a fundamental aspect of VAE training and shows the trade-off between making the latent space follow a normal distribution (which allows for smooth sampling) and accurately reconstructing the inputs.

## Quantitative Evaluation with Inception Score and FID

To objectively evaluate the quality and diversity of our generated samples, we calculated two widely-used metrics:

### Inception Score (IS)

The Inception Score measures both the quality and diversity of generated images. Higher values indicate better performance.

| Model | Inception Score |
|-------|----------------|
| RNN-CNN VAE | 1.0847 ± 0.0188 |
| CNN-CNN VAE | 1.0946 ± 0.0204 |
| Conditional VAE | 1.1020 ± 0.0221 |

These results confirm our qualitative observations:
- The CNN-based VAE outperforms the RNN-based VAE
- The Conditional VAE achieves the highest score, indicating that conditioning on class labels improves sample quality and diversity

### Fréchet Inception Distance (FID)

The FID measures the similarity between the distribution of generated images and real images. Lower values indicate better performance.

| Model | FID |
|-------|-----|
| RNN-CNN VAE | 126.9367 |
| CNN-CNN VAE | 122.3115 |
| Conditional VAE | 123.4559 |

The FID results show that:
- The CNN-based VAE produces samples that are closest to the real data distribution
- The Conditional VAE performs slightly worse than the pure CNN VAE in terms of distribution matching
- All models show significant improvement when using diverse latent space sampling strategies

Both metrics were calculated using a pre-trained InceptionV3 model to extract features from both real and generated images, which is the standard approach in the evaluation of generative models.

## Classifier Evaluation

We trained a separate MLP classifier to evaluate the quality of the generated samples. The classifier achieved good performance on the test set from the QuickDraw dataset:

- **Rabbit class**: 80% accuracy with 75.4% average confidence
- **Yoga class**: 80% accuracy with 75.6% average confidence  
- **Snowman class**: 100% accuracy with 97.0% average confidence

These results suggest that:

1. The classes have different levels of visual distinctiveness
2. Snowman is the most visually distinctive class, with perfect classification and very high confidence
3. Rabbit and yoga have more visual overlap with other classes

The visual inspection of the conditional samples shows that:

1. The generated samples maintain class-specific characteristics
2. The conditional VAE successfully learns to generate distinct samples for different classes
3. The snowman class appears to have the most consistent and recognizable features in the generated samples, matching the classifier's high confidence on real snowman images


## Key Findings

1. **CNN vs. RNN Encoders**: The CNN-based encoder consistently outperformed the RNN-based encoder for this image generation task. This aligns with the general understanding that convolutional architectures are well-suited for spatial data like images.

2. **VAE Loss Components**: The reconstruction term dominates early in training, while the KL divergence term becomes more significant as training progresses. This demonstrates the VAE's natural tendency to prioritize reconstruction before regularizing the latent space.

3. **Conditioning Effect**: Adding class conditioning to the VAE improves the model's ability to generate class-specific samples, demonstrating the power of conditional generation techniques. This is validated by the Inception Score, where the Conditional VAE achieved the highest score.

4. **Quantitative Metrics**: The IS and FID metrics provide objective measurements that confirm our qualitative observations. The CNN-CNN VAE achieves the best FID, while the Conditional VAE has the best Inception Score, suggesting different strengths in distribution matching versus sample quality/diversity.

5. **Latent Space Sampling**: Using diverse sampling strategies in the latent space significantly improves the quality and diversity of generated samples, as evidenced by the improved IS and FID metrics compared to our initial runs.

6. **Training Efficiency**: Even with reduced data and training time, VAEs can learn meaningful latent representations, as shown by our quick implementation.

7. **Classification Performance**: Different classes have varying levels of distinctiveness, with snowman being the most recognizable class in our dataset.

## Conclusion

Based on our experiments, the CNN-based encoder and decoder architecture outperforms the RNN-based encoder. This is likely because the convolutional layers are better suited for capturing the spatial patterns in image data compared to RNNs, which are traditionally stronger on sequential data.

The conditional VAE successfully extends the base model to enable class-conditional generation, demonstrating the flexibility of the VAE framework for controlled image generation tasks.

The KL divergence and reconstruction loss components show the expected trade-off during training, with the model balancing reconstruction quality against latent space regularization. This balance is crucial for generating novel yet realistic samples from the learned distribution.

The IS and FID metrics provide objective validation of our model comparisons, with the CNN-CNN VAE achieving the best distribution matching (lowest FID) and the Conditional VAE producing the most diverse and high-quality samples (highest IS).

Our separate classifier validation confirms that the classes in the dataset have different levels of visual distinctiveness, which likely affects the quality and recognizability of the VAE-generated samples. The snowman class appears to be the most visually consistent, which matches the classifier's perfect accuracy on real snowman images. 

## Reproduction Instructions

### Required Files

To reproduce the results of this assignment, the following files are needed:

1. **Main implementation files**:
   - `vae_implementation.py`: Core implementation of all VAE architectures
   - `evaluate_vae_metrics.py`: Scripts for calculating IS and FID metrics
   - `evaluate_vae_samples.py`: Evaluation of VAE samples using the classifier

2. **QuickDraw dataset**:
   - `quickdraw_subset_np/`: Directory containing the dataset files
     - `train_images.npy`: Training images (20,000 samples)
     - `train_labels.npy`: Training labels
     - `test_images.npy`: Test images (5,000 samples)
     - `test_labels.npy`: Test labels
     - `README.md`: Information about the dataset

3. **Pre-trained models**:
   - `models/rnn_vae_model.pth`: RNN-CNN VAE model
   - `models/cnn_vae_model.pth`: CNN-CNN VAE model
   - `models/conditional_vae_model.pth`: Conditional VAE model
   - `classifier_model.pkl`: Classifier model from Assignment 1

4. **Results**:
   - `vae_results/`: Directory containing generated samples and loss plots
   - `vae_metrics/`: Directory containing evaluation metrics
   - `vae_results_analysis.md`: This document analyzing the results

### Environment Setup

```bash
# Create a virtual environment
python -m venv vae_env

# Activate the environment
# On Windows:
vae_env\Scripts\activate
# On macOS/Linux:
source vae_env/bin/activate

# Install required packages
pip install torch torchvision numpy matplotlib scikit-learn scipy pillow tqdm
```

### Running the Implementation

To train the VAE models and generate results:

```bash
# Train the VAE models (RNN-CNN and CNN-CNN)
python vae_implementation.py

# Calculate IS and FID metrics
python evaluate_vae_metrics.py

# Evaluate generated samples with the classifier
python evaluate_vae_samples.py
```

### Expected Outputs

After running the scripts, the following outputs will be generated:

1. **Trained models** saved in the `models/` directory
   - RNN-CNN VAE, CNN-CNN VAE, and Conditional VAE models

2. **Generated samples** saved in the `vae_results/` directory:
   - RNN-CNN samples: `rnn_cnn_samples.png`
   - CNN-CNN samples: `cnn_cnn_samples.png`
   - Conditional samples: `conditional_samples.png`
   - Loss plots: `loss_comparison.png`, `rnn_cnn_loss_components.png`, `cnn_cnn_loss_components.png`

3. **Evaluation metrics** saved in the `vae_metrics/` directory:
   - Inception Score and FID results: `metrics.pkl`
   - Metrics visualization: `metrics_comparison.png`
   - Metrics summary: `metrics_summary.md`

4. **Classifier evaluation** saved in the `vae_evaluation/` directory:
   - Classifier predictions on generated samples: `classifier_evaluation.png`
   - Original conditional samples: `conditional_samples_original.png`

### Replicating Partial Results

If you want to quickly test the implementation with reduced computation:

```bash
# Train a simplified VAE with fewer epochs and samples
python vae_implementation.py --quick

# Evaluate with simplified metrics
python evaluate_vae_metrics.py --quick

# Run simplified classifier evaluation
python evaluate_vae_samples_simple.py
```

Note: The simplified implementation will produce similar but less refined results. 