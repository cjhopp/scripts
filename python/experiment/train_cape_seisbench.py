import seisbench
import logging

import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from obspy.clients.fdsn import Client
from obspy import UTCDateTime


logger = seisbench.logger
logger.setLevel(logging.INFO)

dataset_root = Path('/media/chopp/HDD1/chet-meq/cape_modern/seisbench/cape_v1/dataset')

model_root = Path('/media/chopp/HDD1/chet-meq/cape_modern/seisbench/cape_v1/PN_test1/')



class CustomWaveformDataset(sbd.WaveformDataset):
    def __init__(self, metadata, **kwargs):
        super().__init__(metadata, **kwargs)

    def cross_fold(self, fold_number, total_folds=5):
        """
        Returns the datasets for the specified cross-validation fold.

        :param fold_number: Index of the fold (0 to total_folds - 1)
        :param total_folds: Total number of folds for cross-validation
        :return: Training dataset and validation dataset
        """
        if "split" not in self.metadata.columns:
            raise ValueError("Cross-fold requested but no split defined in metadata")

        unique_splits = sorted(self.metadata["split"].unique())
        
        # Assuming that the unique splits are evenly distributed across the folds
        if fold_number < 0 or fold_number >= total_folds:
            raise IndexError(f"Fold number must be between 0 and {total_folds - 1}")

        # Create a mask for the validation split and training splits
        valid_split = unique_splits[fold_number]
        train_splits = unique_splits[:fold_number] + unique_splits[fold_number + 1:]

        valid_mask = (self.metadata["split"] == valid_split).values
        train_mask = self.metadata["split"].isin(train_splits)

        # Return datasets for the specified fold
        train_dataset = self.filter(train_mask, inplace=False)
        valid_dataset = self.filter(valid_mask, inplace=False)

        return train_dataset, valid_dataset


def loss_fn(y_pred, y_true, eps=1e-5):
    # vector cross entropy loss
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h

def train_loop(dataloader):
    size = len(dataloader.dataset)
    for batch_id, batch in enumerate(dataloader):
        # Compute prediction and loss
        x = batch["X"].to(model.device)
        x_preproc = model.annotate_batch_pre(
            x, {}
        )  # Remove mean and normalize amplitude
        pred = model(x_preproc)
        loss = loss_fn(pred, batch["y"].to(model.device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 5 == 0:
            loss, current = loss.item(), batch_id * batch["X"].shape[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader):
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()  # close the model for evaluation

    with torch.no_grad():
        for batch in dataloader:
            x = batch["X"].to(model.device)
            x_preproc = model.annotate_batch_pre(
                x, {}
            )  # Remove mean and normalize amplitude
            pred = model(x_preproc)
            test_loss += loss_fn(pred, batch["y"].to(model.device)).item()

    model.train()  # re-open model for training stage
    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    # Instantiate your Custom Dataset
    custom_dataset = CustomWaveformDataset(dataset_root)

    # Specify parameters
    total_folds = 5
    epochs = 5
    batch_size = 256
    learning_rate = 1e-2
    num_workers = 4  # Number of threads for data loading

    # Initialize a PhaseNet model
    model = sbm.PhaseNet(phases="PSN", norm="std", default_args={"blinding": (200, 200)})
    model.to_preferred_device(verbose=True)
    # Iterate through each fold
    for fold in range(total_folds):
        print(f"Starting Fold {fold + 1}")

        # Get training and validation datasets for the current fold
        train_data, valid_data = custom_dataset.cross_fold(fold, total_folds)
        
        # Setup data generators
        train_generator = sbg.GenericGenerator(train_data)
        valid_generator = sbg.GenericGenerator(valid_data)

        augmentations = [
            sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"),
            sbg.RandomWindow(windowlen=3001, strategy="pad"),
            sbg.ChangeDtype(np.float32),
            sbg.ProbabilisticLabeller(label_columns=phase_dict, model_labels=model.labels, sigma=30, dim=0),
        ]

        train_generator.add_augmentations(augmentations)
        valid_generator.add_augmentations(augmentations)

        # Create DataLoaders
        train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_seeding)
        valid_loader = DataLoader(valid_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model for the specified number of epochs
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_loop(train_loader)
            test_loop(valid_loader)

    model.save(model_root)
    print("Cross-validation completed!")
