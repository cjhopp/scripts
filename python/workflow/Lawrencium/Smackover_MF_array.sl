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
#SBATCH --array=29,30,31,32,33,37,86,103,115,121,128,131,136,139,141,142,143,144,146,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,289,290,291,293,294,295,296,297,298,299,300,303,304,308,311,312,316,330,343,364,69,72,98,104,107,108,109,110,118,129,130,133,137,138,140,147,166,80,73,74,75,77,78,79,81,82,84,124,132,135,248,321
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