#!/bin/bash
#SBATCH -J CJH_Smackover_MF_1-30-26
#SBATCH --partition=lr4
#SBATCH --account=lr_geop
#SBATCH --qos=lr_lowprio
#SBATCH --time=4:00:00
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --output=Smackover-MF_out_%a.txt
#SBATCH --error=Smackover-MF_err_%a.txt
#SBATCH --cpus-per-task=24
#SBATCH --array=0-2
#SBATCH --mail-user=chopp@lbl.gov

module load miniforge3/25.9.1
conda activate eqcorrscan

# Define start and end dates
START_DATE="2025-12-31"
END_DATE="2026-1-30"

# Run the Python script with SLURM task-specific arguments
srun python /global/home/users/chopp/scripts/python/workflow/Lawrencium/Lawrencium_Smackover_MF_from-client.py \
    --splits 3 \
    --instance $SLURM_ARRAY_TASK_ID \
    --start $START_DATE \
    --end $END_DATE