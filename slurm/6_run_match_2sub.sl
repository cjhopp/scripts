#!/bin/bash
#SBATCH -J CJH_Match_Test
#SBATCH -A nesi00228
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=7500
#SBATCH --nodes=1
#SBATCH --exclude=compute-chem-001
#SBATCH --output=matchout_%a_2sub.txt
#SBATCH --error=matcherr_%a_2sub.txt
#SBATCH --cpus-per-task=12
#SBATCH --array=0-182

module load OpenCV/2.4.9-intel-2015a
module load ObsPy/0.10.3rc1-intel-2015a-Python-2.7.9
module load joblib/0.8.4-intel-2015a-Python-2.7.9

srun python2.7 /projects/nesi00228/scripts/6.1_detect_2_cat_PAN.py --splits 183 --instance $SLURM_ARRAY_TASK_ID
