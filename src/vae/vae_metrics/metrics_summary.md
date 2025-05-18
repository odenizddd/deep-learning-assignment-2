# VAE Evaluation Metrics Summary

## Inception Score (IS) and Fréchet Inception Distance (FID)

| Model | Inception Score | FID |
|-------|----------------|-----|
| rnn_cnn | 1.0847 ± 0.0188 | 126.9367 |
| cnn_cnn | 1.0946 ± 0.0204 | 122.3115 |
| conditional | 1.1020 ± 0.0221 | 123.4559 |

## Interpretation

- **Inception Score (IS)**: Higher is better. Measures both quality and diversity of generated images.
- **Fréchet Inception Distance (FID)**: Lower is better. Measures similarity between real and generated image distributions.

