#!/bin/bash
#SBATCH -J CJH_Smackover_MF_1-27-26
#SBATCH --partition=lr6
#SBATCH --account=pc_mqb
#SBATCH --qos=lr_normal
#SBATCH --time=6:30:00
#SBATCH --mem=96000
#SBATCH --nodes=1
#SBATCH --output=Smackover-MF_out_%a.txt
#SBATCH --error=Smackover-MF_err_%a.txt
#SBATCH --cpus-per-task=32
#SBATCH --array=197,199,200,204,209,237,300,53,134,173,185,192,202,207,208,210,211,213,215,216,217,218,219,222,224,230,232,233,235,301,302,307,309,312,313,314,315,316,317,318,319,320,321,322,323,324,326,327,328,331,333,334,336,337,338,339,340,344,346,348,350,352,353,356,357,358,360,362,364,366,367,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,392,214,234,240,242,250,259,260,268,273,274,284,288,292,295,311,325,329,330,332,335,345,347,349,351,354,355,359,363,365,368
#SBATCH --mail-user=chopp@lbl.gov

module load miniforge3/25.9.1
source $(conda info --base)/etc/profile.d/conda.sh  # Initialize Conda
conda activate eqcorrscan_miniforge

# Debugging: Check if ObsPy is available
echo "Checking if ObsPy is available:"
python -c "import obspy; print('ObsPy is available')"

# Define start and end dates
START_DATE="1992-09-23"
END_DATE="2026-01-31"

# Run the Python script with SLURM task-specific arguments
srun python /global/home/users/chopp/scripts/python/workflow/Lawrencium/Lawrencium_Smackover_MF_from-client.py \
    --splits 393 \
    --instance $SLURM_ARRAY_TASK_ID \
    --start $START_DATE \
    --end $END_DATE