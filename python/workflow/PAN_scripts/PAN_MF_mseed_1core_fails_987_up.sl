#!/bin/bash
#SBATCH -J CJH_Match_1core_fails_987_up
#SBATCH -A nesi00228
#SBATCH --time=72:00:00
#SBATCH --mem=70000
#SBATCH --nodes=1
#SBATCH --exclude=compute-chem-001
#SBATCH --output=match_12-15_1core_nodups_987_up_out_%a.txt
#SBATCH --error=match_12-15_1core_nodups_987_up_err_%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --array=1011,1023,1057,1058,1061-1063,1069,1079,1085,1113,1115,1120,1131,1134,1135,1147,1150-1152,1157,1158,1163,1164,1169,1174-1177,1179-1182,1185,1192,1195,1264,1271,1281,1287,1291,1300,1381,1410

module load ObsPy/1.0.2-foss-2015a-Python-3.5.1
module load OpenCV/3.1.0-foss-2015a-Python-3.5.1

srun python3.5 /projects/nesi00228/scripts/python/workflow/PAN_scripts/PAN_MF_mseed.py --splits 1461 --instance $SLURM_ARRAY_TASK_ID --start 1/1/2012 --end 31/12/2015