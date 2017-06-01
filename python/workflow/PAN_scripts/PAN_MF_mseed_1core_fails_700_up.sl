#!/bin/bash
#SBATCH -J CJH_Match_1core
#SBATCH -A nesi00228
#SBATCH --time=55:00:00
#SBATCH --mem=70000
#SBATCH --nodes=1
#SBATCH --exclude=compute-chem-001
#SBATCH --output=match_12-15_1core_nodups_out_%a.txt
#SBATCH --error=match_12-15_1core_nodups_err_%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --array=703,704,706,710,711,716,717,719,721,723,729,730,732,738,745,747,749,750,755,759,762-766,768,769,771,777-779,783,784,788-792,797,798,804,807,808,813,823,831,834,835,845,847,849,850,854,855,858,861,862,870,872,873,883,885-895,899-901,903,904,906,910,912-917,919,920,922-924,932,935,937,950,965-968,972,974,979,987

module load ObsPy/1.0.2-foss-2015a-Python-3.5.1
module load OpenCV/3.1.0-foss-2015a-Python-3.5.1

srun python3.5 /projects/nesi00228/scripts/python/workflow/PAN_scripts/PAN_MF_mseed.py --splits 1461 --instance $SLURM_ARRAY_TASK_ID --start 1/1/2012 --end 31/12/2015