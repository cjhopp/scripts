#!/bin/bash
#SBATCH -J CJH_HITP_MF_Array
#SBATCH --partition=lr4
#SBATCH --account=lr_geop
#SBATCH --qos=lr_normal
#SBATCH --time=4:00:00
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --output=HITP-MF_out_%a.txt
#SBATCH --error=HITP-MF_err_%a.txt
#SBATCH --cpus-per-task=24
#SBATCH --array=0-107
#SBATCH --mail-user=chopp@lbl.gov

module load miniforge3/25.9.1
conda activate eqcorrscan

# Define the global start and end dates
START_DATE="2010-01-01"
END_DATE="2026-01-27"

# Run the Python script with SLURM task-specific arguments
srun python /global/home/users/chopp/scripts/python/workflow/Lawrencium/run_HITP_matched_filtering.py \
    --splits 108 \
    --instance $SLURM_ARRAY_TASK_ID \
    --start $START_DATE \
    --end $END_DATE