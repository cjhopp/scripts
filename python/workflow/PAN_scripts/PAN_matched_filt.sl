#!/bin/bash
#SBATCH -J CJH_Match_2013
#SBATCH -A nesi00228
#SBATCH --time=05:00:00
#SBATCH --mem=90000
#SBATCH --nodes=1
#SBATCH --exclude=compute-chem-001
#SBATCH --output=match2013_%a.txt
#SBATCH --error=match2013_%a.txt
#SBATCH --cpus-per-task=12
#SBATCH --array=0-364

module load OpenCV/2.4.9-intel-2015a
module load ObsPy/0.10.3rc1-intel-2015a-Python-2.7.9
module load joblib/0.8.4-intel-2015a-Python-2.7.9

srun python2.7 /projects/nesi00228/scripts/python/workflow/PAN_scripts/PAN_matched_filt.py --splits 365 --instance $SLURM_ARRAY_TASK_ID --start 1/1/2013 --end 31/12/2013
