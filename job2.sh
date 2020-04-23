#!/bin/bash
#
#SBATCH --job-name=train_gdon
#SBATCH --partition=common
#SBATCH --qos=8gpu3d
#SBATCH --gres=gpu:1
#SBATCH -c 8
/bin/hostname
singularity exec --nv \
  --bind /scidatalg/mlp2020_descriptors:/home/$USER/data \
  --bind /results/$USER/general-dense-object-nets:/home/$USER/code \
  /results/$USER/gdon_latest.sif \
  ~/code/dense_correspondence/training/train2.sh
