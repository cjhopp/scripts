#!/bin/bash
#SBATCH -J CJH_Match_2014
#SBATCH -A nesi00228
#SBATCH --time=24:00:00
#SBATCH --mem=90000
#SBATCH --nodes=1
#SBATCH --exclude=compute-chem-001
#SBATCH --output=match_12-15_test_out_%a.txt
#SBATCH --error=match_12-15_test_err_%a.txt
#SBATCH --cpus-per-task=12
#SBATCH --array=0

module load ObsPy/1.0.2-foss-2015a-Python-3.5.1
module load OpenCV/3.1.0-foss-2015a-Python-3.5.1

srun python3.5 /projects/nesi00228/scripts/python/workflow/PAN_scripts/PAN_MF_mseed.py --instance $SLURM_ARRAY_TASK_ID --start 2/1/2014 --end 2/1/2014
