#!/bin/bash
#SBATCH -J CJH_pyasdf2temps
#SBATCH -A nesi00228
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=7500
#SBATCH --nodes=1
#SBATCH --output=pyasdf2temps_%a.txt
#SBATCH --error=pyasdf2temps_%a.txt
#SBATCH --cpus-per-task=8
#SBATCH --array=0-35

module load OpenCV/2.4.9-intel-2015a
module load ObsPy/0.10.3rc1-intel-2015a-Python-2.7.9
module load joblib/0.8.4-intel-2015a-Python-2.7.9

srun python2.7 /projects/nesi00228/scripts/workflow/PAN_scripts/pyasdf_2_templates_PAN.py --splits 36 --instance $SLURM_ARRAY_TASK_ID
