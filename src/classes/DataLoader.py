import os
import numpy as np


class DataLoader:
    def __init__(self, inputs, labels):

        self.inputs = inputs
        self.labels = labels

        self.num_samples = self.inputs.shape[0]

    def get_batch(self, batch_size: int):
        assert (
            batch_size <= self.num_samples
        ), "Batch size must be less than or equal to the number of samples."

        indices = np.random.choice(self.num_samples, batch_size, replace=False)

        batch_inputs = self.inputs[indices]
        batch_labels = self.labels[indices]

        return batch_inputs, batch_labels


class DataLoaderFactory:
    @staticmethod
    def build(inputs_path: str, labels_path: str, val_ratio: float | None = None):

        assert val_ratio is None or (
            0 <= val_ratio < 1
        ), "val_ratio must be either None or a float between 0 and 1."

        inputs = np.load(inputs_path)
        labels = np.load(labels_path)

        assert (
            inputs.shape[0] == labels.shape[0]
        ), "Inputs and labels must have the same number of samples."

        num_samples = inputs.shape[0]

        shuffle_indices = np.random.permutation(num_samples)

        inputs = inputs[shuffle_indices]
        labels = labels[shuffle_indices]

        if val_ratio is None or val_ratio == 0:
            loader = DataLoader(inputs, labels)
            return loader

        n = int(num_samples * (1 - val_ratio))

        train_inputs = inputs[:n]
        train_labels = labels[:n]

        val_inputs = inputs[n:]
        val_labels = labels[n:]

        train_loader = DataLoader(train_inputs, train_labels)
        val_loader = DataLoader(val_inputs, val_labels)

        return train_loader, val_loader


if __name__ == "__main__":

    data_dir = "../../data/quickdraw_subset_np"

    input_file_name = "train_images.npy"
    label_file_name = "train_labels.npy"

    inputs_path = os.path.join(data_dir, input_file_name)
    labels_path = os.path.join(data_dir, label_file_name)

    train_loader, val_loader = DataLoaderFactory.build(
        inputs_path, labels_path, val_ratio=0.1
    )
    batch_size = 32
    batch_inputs, batch_labels = train_loader.get_batch(batch_size)
    print(f"Batch inputs shape: {batch_inputs.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Batch inputs: {batch_inputs}")
    print(f"Batch labels: {batch_labels}")
