#!/bin/bash
#SBATCH -J CJH_Match_1core
#SBATCH -A nesi00228
#SBATCH --time=55:00:00
#SBATCH --mem=70000
#SBATCH --nodes=1
#SBATCH --exclude=compute-chem-001
#SBATCH --output=match_12-15_1core_nodups_out_%a.txt
#SBATCH --error=match_12-15_1core_nodups_err_%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1460

module load ObsPy/1.0.2-foss-2015a-Python-3.5.1
module load OpenCV/3.1.0-foss-2015a-Python-3.5.1

srun python3.5 /projects/nesi00228/scripts/python/workflow/PAN_scripts/PAN_MF_mseed.py --splits 1461 --instance $SLURM_ARRAY_TASK_ID --start 1/1/2012 --end 31/12/2015