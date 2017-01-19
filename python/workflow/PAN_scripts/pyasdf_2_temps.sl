#!/bin/bash
#SBATCH -J CJH_pyasdf2temps_2014
#SBATCH -A nesi00228
#SBATCH --time=04:00:00
#SBATCH --mem=80000
#SBATCH --nodes=1
#SBATCH --output=pyasdf2temps_2014__out_%a.txt
#SBATCH --error=pyasdf2temps_2014_err_%a.txt
#SBATCH --cpus-per-task=4
#SBATCH --array=0-364

module load OpenCV/3.1.0-foss-2015a-Python-3.5.1
module load Python/3.5.1-intel-2015a

srun python3.5 /projects/nesi00228/scripts/python/workflow/PAN_scripts/pyasdf_2_templates_PAN.py --splits 365 --instance $SLURM_ARRAY_TASK_ID
