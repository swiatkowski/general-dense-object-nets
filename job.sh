#!/bin/bash
#
#SBATCH --job-name=train_gdon
#SBATCH --partition=common
#SBATCH --qos=8gpu3d
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodelist=asusgpu4,sylvester,arnold
#SBATCH --cpus-per-task 8
/bin/hostname
bash /results/$USER/general-dense-object-nets/singularity_exec.sh
