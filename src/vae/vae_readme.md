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

This assignment implements Variational Autoencoders (VAEs) for the QuickDraw dataset, with three main architectures:

1. VAE with RNN encoder and CNN decoder
2. VAE with CNN encoder and CNN decoder
3. Conditional VAE (based on the best-performing architecture)

The models are trained to generate sketches from the QuickDraw dataset (28×28 grayscale images) and evaluated using metrics like Inception Score (IS) and Fréchet Inception Distance (FID).

## 3. Directory Structure

- `src/vae/`: Main implementation files
  - `vae_implementation.py`: Core implementation of all VAE architectures
  - `evaluate_vae_metrics.py`: Scripts for IS and FID metrics
  - `evaluate_vae_samples.py`: Evaluation with the classifier
  - `models/`: Trained model files
  - `results/`: Generated samples and plots
  - `vae_metrics/`: Evaluation metrics
  - `vae_evaluation/`: Classifier evaluation results

- `data/`: Dataset information
  - `README.md`: Information about the QuickDraw dataset

## 4. Running the Implementation

### Dataset Setup

Before running the scripts, you need to set up the proper directory structure. The scripts expect specific folder names:

```bash
# While in the src/vae directory:

# 1. Create a symbolic link to the QuickDraw dataset
ln -s /path/to/your/quickdraw_subset_np quickdraw_subset_np

# 2. Create a symbolic link from results to vae_results (if needed)
# The scripts expect a folder named 'vae_results' but the actual results are stored in 'results'
ln -s results vae_results
```

### Train VAE Models

To train all VAE models (RNN-CNN, CNN-CNN, and Conditional VAE):

```bash
cd src/vae
python vae_implementation.py
```

For a quicker test with reduced computation:

```bash
python vae_implementation.py --quick
```

### Evaluate Models with Metrics

To calculate Inception Score (IS) and Fréchet Inception Distance (FID):

```bash
python evaluate_vae_metrics.py
```

For a quicker evaluation with fewer samples:

```bash
python evaluate_vae_metrics.py --quick
```

### Evaluate Samples with Classifier

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

- Comparison of different VAE architectures
- Loss component analysis
- Visual evaluation of generated samples
- Quantitative evaluation using IS and FID metrics
- Classifier-based evaluation

For detailed reproduction instructions, please refer to the "Reproduction Instructions" section in the report.
