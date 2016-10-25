#!/bin/bash
#SBATCH -J CJH_det_2_cats
#SBATCH -A nesi00228
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=7500
#SBATCH --nodes=1
#SBATCH --output=cat_out_%a.txt
#SBATCH --error=cat_err_%a.txt
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2

module load OpenCV/2.4.9-intel-2015a
module load ObsPy/0.10.3rc1-intel-2015a-Python-2.7.9
module load joblib/0.8.4-intel-2015a-Python-2.7.9

srun python2.7 /projects/nesi00228/scripts/6_PAN_match_filt_136-144.py --splits 3 --instance $SLURM_ARRAY_TASK_ID
