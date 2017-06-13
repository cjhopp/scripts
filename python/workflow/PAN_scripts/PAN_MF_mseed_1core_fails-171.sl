#!/bin/bash
#SBATCH -J CJH_Match_1core_fails-171
#SBATCH -A nesi00228
#SBATCH --time=60:00:00
#SBATCH --mem=750000
#SBATCH --nodes=1
#SBATCH --exclude=compute-bigmem-[001,003,004]
#SBATCH --output=match_12-15_1core_nodups_fails-171_out_%a.txt
#SBATCH --error=match_12-15_1core_nodups_fails-171_err_%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --array=171

module load ObsPy/1.0.2-foss-2015a-Python-3.5.1
module load OpenCV/3.1.0-foss-2015a-Python-3.5.1

srun python3.5 /projects/nesi00228/scripts/python/workflow/PAN_scripts/PAN_MF_mseed.py --splits 1461 --instance $SLURM_ARRAY_TASK_ID --start 1/1/2012 --end 31/12/2015