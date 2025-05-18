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

We used MSE loss to train the network for 10.000 steps, with an Adam Optimizer, a learning rate of 1e-3, and a batch size of 256. The training loss can be seen below.

![Figure_1](https://github.com/user-attachments/assets/51146b2b-5fc3-48b4-9523-ffb54a868b02)

Some of the reconstructions can be seen below, along with comparisons with the outputs from an untrained AE.

![Figure_2](https://github.com/user-attachments/assets/77a6bbca-bda2-4029-88c3-40fb774bdd67)



### CNN AutoEncoder


We used two CNNs similar to the LSTM AE above. The encoder is a CNN with 2 convolutional layers. The first layers maps single-channel input image onto a 16-channel feature map using 16 3x3 kernels with a stride of 2. The second convolutional layer maps the 16 features onto 32 features, which gives us 16x7x7 features. We map those onto a 64 dimensional latent space. Decoder has the same architecture in reverse.

We train the CNN for 1.000 steps with an Adam optimizer, using a learning rate of 1e-3 and a batch size of 256. The training loss can be seen below.

![Figure_4](https://github.com/user-attachments/assets/b5a571de-bd95-4bad-966c-29310e51ceb8)

Some of the reconstructions can be seen below, along with comparisons with the outputs from an untrained AE.

![Figure_5](https://github.com/user-attachments/assets/3f0bea3e-c743-4a80-bb9f-57e7c0d78cdf)


### Visualizing Latent Space Embeddings With TSNE

We fed the whole dataset through the trained LSTM AE to get the latent space representation of every image. Then we used TSNE method to find the two most prominent features and project the latent space vectors onto that 2D space. The results are visualized below.

![Figure_3](https://github.com/user-attachments/assets/fd395434-07c4-41e4-829d-76aa09623c5a)

Some clusters are reasonably well-separated, for example blue, brown, gray, and cyan classes look very distinct from the rest of the data. However, boundaries are not very sharp, and there is a significant degree of overlap. Especially between the green class and the rest.


