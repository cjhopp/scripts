#!/bin/bash
#SBATCH -J CJH_Match_Test
#SBATCH -A nesi00228
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=7500
#SBATCH --nodes=1
#SBATCH --output=matchout_%a.txt
#SBATCH --error=matcherr_%a.txt
#SBATCH --cpus-per-task=12
#SBATCH --array=0-1

module load OpenCV/2.4.9-intel-2015a
module load ObsPy/0.10.3rc1-intel-2015a-Python-2.7.9
module load joblib/0.8.4-intel-2015a-Python-2.7.9

srun python2.7 /projects/nesi00228/scripts/6_PAN_match_filt_299-300.py --splits 3 --instance $SLURM_ARRAY_TASK_ID
