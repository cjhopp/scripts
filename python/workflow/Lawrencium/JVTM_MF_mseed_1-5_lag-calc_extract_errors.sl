#!/bin/bash
#SBATCH -J CJH_JVTM_MF_1-5
#SBATCH --partition=lr3
#SBATCH --account=pc_seisproc
#SBATCH --qos=lr_normal
#SBATCH --time=4:00:00
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --output=jvtm-MF_out_%a.txt
#SBATCH --error=jvtm-MF_err_%a.txt
#SBATCH --cpus-per-task=16
#SBATCH --array=118,140,153,156,164-179,181,184,190-192,195,196
#SBATCH --mail-user=chopp@lbl.gov

module load python/3.7
source activate eqcorrscan

srun python /global/home/users/chopp/scripts/python/workflow/Lawrencium/Lawrencium_MF_mseed_JVTM.py --splits 201 --instance $SLURM_ARRAY_TASK_ID --start 1/1/2010 --end 30/11/2021
