#!/bin/bash
#SBATCH -J CJH_HITP2_MF_Array
#SBATCH --partition=lr6
#SBATCH --account=pc_mqb
#SBATCH --qos=lr_normal
#SBATCH --time=4:00:00
#SBATCH --mem=96000
#SBATCH --nodes=1
#SBATCH --output=HITP2-MF_out_%a.txt
#SBATCH --error=HITP2-MF_err_%a.txt
#SBATCH --cpus-per-task=32
#SBATCH --array=0-34
#SBATCH --mail-user=chopp@lbl.gov

module load miniforge3/25.9.1
source $(conda info --base)/etc/profile.d/conda.sh  # Initialize Conda
conda activate eqcorrscan_miniforge

# Debugging: Check if ObsPy is available
echo "Checking if ObsPy is available:"
python -c "import obspy; print('ObsPy is available')"

# Define the global start and end dates
START_DATE="2025-10-23"
END_DATE="2026-2-5"

# Run the Python script with SLURM task-specific arguments
srun python /global/home/users/chopp/scripts/python/workflow/Lawrencium/Lawrencium_HITP_MF_from-client.py \
    --splits 35 \
    --instance $SLURM_ARRAY_TASK_ID \
    --start $START_DATE \
    --end $END_DATE