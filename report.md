# CmpE 597 Assignment 2 Report

This report is an explanation of all the work done by Özgür Deniz Demir and Osman yasin Baştuğ for the second assignment of CmpE 597 Deep Learning course given at the Boğaziçi University in Spring 2025. The relevant code can be found at this repository: [https://github.com/odenizddd/deep-learning-assignment-2](https://github.com/odenizddd/deep-learning-assignment-2).

The purpose of this work is to implement an autoencoder (AE) and a variational autoencoder (VAE) to reconstruct 28x28 grayscale images.

## AutoEncoder Implementation

The autoencoder is implemented in two different approaches.

### LSTM AutoEncoder

We used two different LSTM networks to act as the encoder and the decoder of the autoencoder. Each LSTM treats the 28x28 grayscale image as a multivariate time series. Each row of the image is considered to be a multidimensional feature vector. The encoder network predicts the pixel values of the next row given the previous hidden state and the pixels values of the previous row. The final hidden state is projected onto the latent space via a fully connected layer. The latent vector is projected once more onto the input space of the decoder LSTM, which predicts a sequence of pixel values that correspond to the rows of the final reconstructed image.

The hyperparameters of the network are as follows:

- Input Dimension: 28
- Hidden Dimension: 128
- Latent Dimension: 128
- Num of Layers Stacked: 2

We used MSE loss to train the network for 10.000 steps, with an Adam Optimizer and a learning rate of 1e-3. The training loss can be seen below.

### CNN AutoEncoder
