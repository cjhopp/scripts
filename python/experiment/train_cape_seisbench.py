
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API")

import seisbench
import logging
import pandas as pd

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
import os


logger = seisbench.logger
logger.setLevel(logging.INFO)

dataset_root = Path('/media/chopp/HDD1/chet-meq/cape_modern/seisbench/cape_v1/dataset')
model_root = Path('/media/chopp/HDD1/chet-meq/cape_modern/seisbench/cape_v1/models')

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
        
        if fold_number < 0 or fold_number >= total_folds:
            raise IndexError(f"Fold number must be between 0 and {total_folds - 1}")

        valid_split = unique_splits[fold_number]
        train_splits = unique_splits[:fold_number] + unique_splits[fold_number + 1:]

        valid_mask = (self.metadata["split"] == valid_split).values
        train_mask = self.metadata["split"].isin(train_splits)

        train_dataset = self.filter(train_mask, inplace=False)
        valid_dataset = self.filter(valid_mask, inplace=False)

        return train_dataset, valid_dataset


def loss_fn(y_pred, y_true, eps=1e-5):
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1) 
    h = h.mean()
    return -h


def train_loop(dataloader, model, optimizer):
    for batch in dataloader:
        x = batch["X"].to(model.device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, batch["y"].to(model.device))
        loss.backward()
        optimizer.step()

def test_loop(dataloader, model):
    total_loss = 0
    num_batches = len(dataloader)
    model.eval()
    
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["X"].to(model.device)
            pred = model(x)
            total_loss += loss_fn(pred, batch["y"].to(model.device)).item()

            all_preds.append(pred.cpu().numpy())
            all_true.append(batch["y"].to(model.device).cpu().numpy())

    avg_loss = total_loss / num_batches
    print(f"Validation loss: {avg_loss:.4f}")

    return np.concatenate(all_preds), np.concatenate(all_true), avg_loss


if __name__ == "__main__":
    # Instantiate your Custom Dataset
    custom_dataset = CustomWaveformDataset(dataset_root)

    # Specify parameters
    total_folds = 5
    epochs = 5
    batch_size = 256
    learning_rate = 1e-2
    num_workers = 4

    # Variables to keep track of validation losses for each fold
    validation_losses = []

    # Create a directory to save figures
    output_dir = "/media/chopp/HDD1/chet-meq/cape_modern/seisbench/cape_v1/output_figures"
    os.makedirs(output_dir, exist_ok=True)
    
    phase_dict = {
        "trace_p_arrival_sample": "P",
        "trace_pP_arrival_sample": "P",
        "trace_P_arrival_sample": "P",
        "trace_P1_arrival_sample": "P",
        "trace_Pg_arrival_sample": "P",
        "trace_Pn_arrival_sample": "P",
        "trace_PmP_arrival_sample": "P",
        "trace_pwP_arrival_sample": "P",
        "trace_pwPm_arrival_sample": "P",
        "trace_s_arrival_sample": "S",
        "trace_S_arrival_sample": "S",
        "trace_S1_arrival_sample": "S",
        "trace_Sg_arrival_sample": "S",
        "trace_SmS_arrival_sample": "S",
        "trace_Sn_arrival_sample": "S",
    }

    # Iterate through each fold for cross-validation
    for fold in range(total_folds):
        print(f"Starting Fold {fold + 1}")

        # Initialize the PhaseNet model
        model = sbm.PhaseNet(phases="PSN", norm="std", default_args={"blinding": (200, 200)})
        model.to_preferred_device(verbose=True)

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

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        fold_val_losses = []

        # Train the model for the specified number of epochs
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            train_loop(train_loader, model, optimizer)
            preds, true, val_loss = test_loop(valid_loader, model)
            fold_val_losses.append(val_loss)

            # Plotting the validation results for this fold
            plt.figure(figsize=(15, 5))
            plt.plot(true.flatten(), label='True Values', alpha=0.5)
            plt.plot(preds.flatten(), label='Predictions', alpha=0.5)
            plt.title(f'Fold {fold + 1} - Epoch {epoch + 1}')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'fold_{fold + 1}_epoch_{epoch + 1}.png'))
            plt.close()
        
        validation_losses.append(np.mean(fold_val_losses))


    # After k-fold cross-validation, select the best hyperparameters based on validation losses
    best_fold_index = np.argmin(validation_losses)
    print(f"Best fold based on validation losses: {best_fold_index + 1}")

    # Train a final model on the entire dataset using the best hyperparameters
    final_model = sbm.PhaseNet(phases="PSN", norm="std", default_args={"blinding": (200, 200)})
    final_model.to_preferred_device(verbose=True)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)

    # Prepare data loaders for the complete dataset
    full_generator = sbg.GenericGenerator(custom_dataset)
    full_generator.add_augmentations(augmentations)
    full_loader = DataLoader(full_generator, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_seeding)

    # Final training loop on the entire dataset
    for epoch in range(epochs):
        print(f"Final Training Epoch {epoch + 1}")
        train_loop(full_loader, final_model, optimizer)

    # Save the final model using the model's native save method
    final_model.save(str(model_root / 'final_phase_net_model'))

    # Plot predictions on the entire dataset using the final model
    all_preds = []
    all_true = []

    final_model.eval()
    with torch.no_grad():
        for batch in full_loader:
            x = batch["X"].to(final_model.device)
            pred = final_model(x)
            all_preds.append(pred.cpu().numpy())
            all_true.append(batch["y"].to(final_model.device).cpu().numpy())

    # Concatenate all predictions and true labels
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    # Plotting results for the final model
    plt.figure(figsize=(15, 5))
    plt.plot(all_true.flatten(), label='True Values', alpha=0.5)
    plt.plot(all_preds.flatten(), label='Final Model Predictions', alpha=0.5)
    plt.title('Final Model Predictions vs True Values')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'final_model_performance.png'))
    plt.close()

    # Generate waveform plots for a few examples
    num_examples = 5  # Number of waveform examples to plot
    sample_indices = np.random.choice(len(custom_dataset), size=num_examples, replace=False)

    for i in sample_indices:
        sample_waveform = custom_dataset[i]  # Fetch the waveform sample
        true_label = sample_waveform['y']
        input_waveform = sample_waveform['X'].flatten()  # Flatten to plot 1D
        final_model_prediction = final_model(torch.tensor(sample_waveform['X']).unsqueeze(0).to(final_model.device)).cpu().detach().numpy().flatten()

        plt.figure(figsize=(15, 5))
        plt.plot(input_waveform, label='Input Waveform', alpha=0.5)
        plt.plot(true_label.flatten(), label='True Label', alpha=0.5)
        plt.plot(final_model_prediction, label='Final Model Prediction', alpha=0.75)
        plt.title(f'Waveform {i + 1} - True vs Predicted')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'waveform_example_{i + 1}.png'))
        plt.close()

    print("Training, evaluation, and waveform plots generation completed!")
