#!/bin/bash
#SBATCH -J CJH_pyasdf2temps_2013
#SBATCH -A nesi00228
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=7500
#SBATCH --nodes=1
#SBATCH --output=pyasdf2temps_2013__out_%a.txt
#SBATCH --error=pyasdf2temps_2013_err_%a.txt
#SBATCH --cpus-per-task=8
#SBATCH --array=0-35

module load OpenCV/3.1.0-foss-2015a

srun python3.5 /projects/nesi00228/scripts/python/workflow/PAN_scripts/pyasdf_2_templates_PAN.py --splits 36 --instance $SLURM_ARRAY_TASK_ID
