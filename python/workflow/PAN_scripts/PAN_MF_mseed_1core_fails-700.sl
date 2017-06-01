#!/bin/bash
#SBATCH -J CJH_Match_1core_fails-700
#SBATCH -A nesi00228
#SBATCH --time=72:00:00
#SBATCH --mem=90000
#SBATCH --nodes=1
#SBATCH --exclude=compute-chem-001
#SBATCH --output=match_12-15_1core_nodups_fails-700_out_%a.txt
#SBATCH --error=match_12-15_1core_nodups_fails-700_err_%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --array=1106,177,256,331,335,350,364,396,403,414,420,428,451,455,459,464,474,484,487,492,494,496,499,500,502,506-508,512,513,519,526,528-536,538,540,543,549,552,555,557-559,561,564-566,568,570,571,573-575,579-584,589,590,592,594,595,597,601,604,607,611,613,615,623,627,632,642,660,663,666,671-674,676-681,686,687,690,691,695-697,699

module load ObsPy/1.0.2-foss-2015a-Python-3.5.1
module load OpenCV/3.1.0-foss-2015a-Python-3.5.1

srun python3.5 /projects/nesi00228/scripts/python/workflow/PAN_scripts/PAN_MF_mseed.py --splits 1461 --instance $SLURM_ARRAY_TASK_ID --start 1/1/2012 --end 31/12/2015