#!/bin/bash
#SBATCH -J CJH_Smackover_MF_1-27-26
#SBATCH --partition=lr4
#SBATCH --account=pc_seisproc
#SBATCH --qos=lr_normal
#SBATCH --time=4:00:00
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --output=Smackover-MF_out_%a.txt
#SBATCH --error=Smackover-MF_err_%a.txt
#SBATCH --cpus-per-task=24
#SBATCH --array=0-107
#SBATCH --mail-user=chopp@lbl.gov

module load python/3.7
module load miniforge3/25.9.1


# Define start and end dates
START_DATE="2010-01-01"
END_DATE="2026-1-27"

# Run the Python script with SLURM task-specific arguments
srun python /global/home/users/chopp/scripts/python/workflow/Lawrencium/Lawrencium_Smackover_MF_from-client.py \
    --splits 108 \
    --instance $SLURM_ARRAY_TASK_ID \
    --start $START_DATE \
    --end $END_DATE