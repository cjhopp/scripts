#!/bin/bash
#SBATCH -J CJH_Smackover_MF_1-27-26
#SBATCH --partition=lr4
#SBATCH --account=lr_geop
#SBATCH --qos=lr_lowprio
#SBATCH --time=4:30:00
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --output=Smackover-MF_out_%a.txt
#SBATCH --error=Smackover-MF_err_%a.txt
#SBATCH --cpus-per-task=24
#SBATCH --array=0-238
#SBATCH --mail-user=chopp@lbl.gov

module load miniforge3/25.9.1
source $(conda info --base)/etc/profile.d/conda.sh  # Initialize Conda
conda activate eqcorrscan_miniforge

# Debugging: Check if ObsPy is available
echo "Checking if ObsPy is available:"
python -c "import obspy; print('ObsPy is available')"

# Define start and end dates
START_DATE="2001-03-22"
END_DATE="2026-2-1"

# Run the Python script with SLURM task-specific arguments
srun python /global/home/users/chopp/scripts/python/workflow/Lawrencium/Lawrencium_Smackover_MF_from-client.py \
    --splits 239 \
    --instance $SLURM_ARRAY_TASK_ID \
    --start $START_DATE \
    --end $END_DATE