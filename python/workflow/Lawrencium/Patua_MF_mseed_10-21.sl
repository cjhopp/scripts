#!/bin/bash
#SBATCH -J CJH_Patua_MF_10-21
#SBATCH --partition=lr3
#SBATCH --account=pc_seisproc
#SBATCH --time=2:00:00
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --output=patua-MF_out_%a.txt
#SBATCH --error=patua-MF_err_%a.txt
#SBATCH --cpus-per-task=16
#SBATCH --array=0-200
#SBATCH --mail-user=chopp@lbl.gov

module load python/3.7
conda activate eqcorrscan

srun python3.7 /global/home/users/chopp/scripts/python/workflow/PAN_scripts/PAN_MF_mseed.py --splits 201 --instance $SLURM_ARRAY_TASK_ID --start 1/1/2010 --end 25/4/2021
