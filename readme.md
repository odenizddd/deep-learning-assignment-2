# Bogazici University Deep Learning Course Assignment 2

This repository contains all the work that [Özgür Deniz Demir](https://github.com/odenizddd) and [Osman Yasin Baştuğ](https://github.com/yasinbastug) has done for the second assignment of the CmpE 597 Deep Learning course given at Bogazici University in Spring 2025.

## 1. Setup

We recommend that you create a virtual environment in order to use this repository.

To do this you can clone the repository and in the root of this project run the following commands:

```bash
python3 -m venv env
source ./env/bin/activate
pip install -r requirements.txt
```

## 2. Project Overview

This assignment implements two types of generative models for the QuickDraw dataset:

### AutoEncoders (AE)
1. LSTM-based autoencoder
2. CNN-based autoencoder

### Variational AutoEncoders (VAE)
1. VAE with RNN encoder and CNN decoder
2. VAE with CNN encoder and CNN decoder
3. Conditional VAE (based on the best-performing architecture)

The models are trained to reconstruct and generate sketches from the QuickDraw dataset (28×28 grayscale images).

## 3. Directory Structure

- `src/scripts/`: AutoEncoder implementation files
  - `cnnae.py`: CNN-based autoencoder
  - `lstmae.py`: LSTM-based autoencoder

- `src/vae/`: VAE implementation files
  - `vae_implementation.py`: Core implementation of all VAE architectures
  - `evaluate_vae_metrics.py`: Scripts for IS and FID metrics
  - `evaluate_vae_samples.py`: Evaluation with the classifier
  - `models/`: Trained VAE model files
  - `results/`: Generated samples and plots
  - `vae_metrics/`: Evaluation metrics
  - `vae_evaluation/`: Classifier evaluation results

- `data/`: Dataset information
  - `README.md`: Information about the QuickDraw dataset

## 4. Running the Implementation

### Dataset Setup

Before running the scripts, you need to have the QuickDraw dataset. For the VAE implementation, the scripts expect to find the dataset in specific locations:

```bash
# While in the src/vae directory:
ln -s /path/to/your/quickdraw_subset_np quickdraw_subset_np
ln -s results vae_results  # The scripts also expect a directory named vae_results
```

### Running AutoEncoder Models

To train and evaluate the autoencoder models:

```bash
# Run LSTM-based autoencoder
cd src/scripts
python lstmae.py

# Run CNN-based autoencoder
python cnnae.py
```

### Running VAE Models

To train all VAE models (RNN-CNN, CNN-CNN, and Conditional VAE):

```bash
cd src/vae
python vae_implementation.py
```

For a quicker test with reduced computation:

```bash
python vae_implementation.py --quick
```

### Evaluate VAE Models with Metrics

To calculate Inception Score (IS) and Fréchet Inception Distance (FID):

```bash
python evaluate_vae_metrics.py
```

For a quicker evaluation with fewer samples:

```bash
python evaluate_vae_metrics.py --quick
```

### Evaluate VAE Samples with Classifier

To evaluate generated samples using the pre-trained classifier:

```bash
python evaluate_vae_samples.py
```

For a simplified evaluation:

```bash
python evaluate_vae_samples_simple.py
```

## 5. Results

The comprehensive results and analysis are available in the `report.md` file, which includes:

### AutoEncoder Results
- Reconstruction quality comparison between LSTM and CNN autoencoders
- Training loss curves
- TSNE visualization of the latent space

### VAE Results
- Comparison of different VAE architectures
- Loss component analysis (BCE vs. KLD)
- Visual evaluation of generated samples
- Quantitative evaluation using IS and FID metrics
- Classifier-based evaluation of conditional samples

For detailed reproduction instructions, please refer to the "Reproduction Instructions" section in the report.
