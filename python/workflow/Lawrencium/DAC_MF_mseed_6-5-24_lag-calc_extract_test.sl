#!/bin/bash
#SBATCH -J CJH_DAC_MF_6-5-24
#SBATCH --partition=lr4
#SBATCH --account=pc_seisproc
#SBATCH --qos=lr_normal
#SBATCH --time=4:00:00
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --output=dac-MF_out_%a.txt
#SBATCH --error=dac-MF_err_%a.txt
#SBATCH --cpus-per-task=24
#SBATCH --array=0-1
#SBATCH --mail-user=chopp@lbl.gov

module load python/3.7
source activate eqcorrscan_dev

srun python /global/home/users/chopp/scripts/python/workflow/Lawrencium/Lawrencium_MF_mseed_DAC.py --splits 2 --instance $SLURM_ARRAY_TASK_ID --start 19/4/2024 --end 22/4/2024
