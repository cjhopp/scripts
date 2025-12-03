
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API")

import seisbench
import logging

import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os

# Set up logging
logger = seisbench.logger
logger.setLevel(logging.INFO)

# --- Configuration ---
dataset_root = Path('/media/chopp/HDD1/chet-meq/cape_modern/seisbench/cape_v1/dataset')
model_root = Path('/media/chopp/HDD1/chet-meq/cape_modern/seisbench/cape_v1/models')
output_dir = Path('/media/chopp/HDD1/chet-meq/cape_modern/seisbench/cape_v1/output_figures')
num_examples = 50

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# This class definition is needed to load the dataset metadata
class CustomWaveformDataset(sbd.WaveformDataset):
    def __init__(self, metadata, **kwargs):
        super().__init__(metadata, **kwargs)


def generate_plots():
    # --- Load Model ---
    model_path = model_root / 'final_phase_net_model'
    logger.info(f"Loading model from {model_path}...")
    try:
        final_model = sbm.PhaseNet.load(str(model_path))
    except FileNotFoundError:
        logger.error(f"Model not found at {model_path}. Please ensure the model was trained and saved correctly.")
        return

    final_model.to_preferred_device(verbose=True)
    final_model.eval()

    # --- Load Dataset ---
    logger.info(f"Loading dataset from {dataset_root}...")
    custom_dataset = CustomWaveformDataset(dataset_root)

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
    
    augmentations = [
        sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"),
        sbg.RandomWindow(windowlen=3001, strategy="pad"),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, model_labels=final_model.labels, sigma=30, dim=0),
    ]

    eval_generator = sbg.GenericGenerator(custom_dataset)
    eval_generator.add_augmentations(augmentations)

    # --- Generate Plots ---
    logger.info(f"Generating {num_examples} evaluation plots...")
    for i in range(num_examples):
        sample_idx = np.random.randint(len(eval_generator))
        sample = eval_generator[sample_idx]
        
        # Robustly get trace name from metadata dataframe index
        trace_name = eval_generator.dataset.metadata.iloc[sample_idx].name

        fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1, 1]})
        
        axs[0].plot(sample["X"].T)
        axs[0].set_ylabel("Waveform")
        axs[0].legend(["E", "N", "Z"])

        axs[1].plot(sample["y"].T)
        axs[1].set_ylabel("True Labels")
        axs[1].legend(final_model.labels)

        with torch.no_grad():
            x = torch.from_numpy(sample["X"]).to(final_model.device).unsqueeze(0)
            x_preproc = final_model.annotate_batch_pre(x, {})
            pred = final_model(x_preproc)[0].cpu().numpy()

        axs[2].plot(pred.T)
        axs[2].set_ylabel("Predicted Labels")
        axs[2].set_xlabel("Samples")
        axs[2].legend(final_model.labels)

        # Use the corrected trace identifier for the title
        fig.suptitle(f"Final Model - Trace: {trace_name}", y=0.92)
        save_path = output_dir / f'final_model_example_{i + 1}.png'
        plt.savefig(save_path)
        plt.close(fig)
        if (i + 1) % 10 == 0:
            logger.info(f"  ... {i + 1}/{num_examples} plots saved.")

    logger.info(f"\nFigure generation completed! {num_examples} plots saved to {output_dir}")

if __name__ == "__main__":
    generate_plots()
