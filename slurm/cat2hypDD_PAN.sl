#!/bin/bash
#SBATCH -J CJH_cat2hypDD
#SBATCH -A nesi00228
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --nodes=1
#SBATCH --exclude=compute-chem-001
#SBATCH --output=cat2hypDDout_%a.txt
#SBATCH --error=cat2hypDDerr_%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --array=0-301

module load OpenCV/2.4.9-intel-2015a
module load ObsPy/0.10.3rc1-intel-2015a-Python-2.7.9
module load joblib/0.8.4-intel-2015a-Python-2.7.9

srun python2.7 /projects/nesi00228/scripts/cat2hypDD_PAN.py --splits 302 --instance $SLURM_ARRAY_TASK_ID
