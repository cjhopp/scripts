#!/bin/bash
#SBATCH -J CJH_Smackover_MF_4-17-26
#SBATCH --partition=lr6
#SBATCH --account=pc_mqb
#SBATCH --qos=lr_normal
#SBATCH --time=8:00:00
#SBATCH --mem=96000
#SBATCH --nodes=1
#SBATCH --output=Smackover-MF_analyzed_out_%a.txt
#SBATCH --error=Smackover-MF_analyzed_err_%a.txt
#SBATCH --cpus-per-task=32
#SBATCH --array=0-367
#SBATCH --mail-user=chopp@lbl.gov

module load miniforge3/25.9.1
source $(conda info --base)/etc/profile.d/conda.sh  # Initialize Conda
conda activate eqcorrscan_miniforge

# Debugging: Check if ObsPy is available
echo "Checking if ObsPy is available:"
python -c "import obspy; print('ObsPy is available')"

# Define start and end dates
START_DATE="2009-02-12"
END_DATE="2026-03-31"

# Run the Python script with SLURM task-specific arguments
srun python /global/home/users/chopp/scripts/python/workflow/Lawrencium/Lawrencium_Smackover_MF_from-client.py \
    --splits 368 \
    --instance $SLURM_ARRAY_TASK_ID \
    --start $START_DATE \
    --end $END_DATE