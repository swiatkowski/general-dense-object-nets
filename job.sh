#!/bin/bash
#
#SBATCH --job-name=train_gdon
#SBATCH --partition=common
#SBATCH --qos=8gpu3d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --exclude=asusgpu2,asusgpu5,steven,bruce
/bin/hostname
bash /results/$USER/general-dense-object-nets/singularity_exec.sh
