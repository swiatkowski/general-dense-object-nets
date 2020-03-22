#!/bin/bash
#
#SBATCH --job-name=train_gdon
#SBATCH --partition=common
#SBATCH --qos=8gpu3d
#SBATCH --gres=gpu:1
/bin/hostname
singularity run --nv \
  --bind /scidatasm/dense_object_nets/data:/home/$USER/data \
  --bind /results/$USER/general-dense-object-nets:/home/$USER/code \
  /results/$USER/jakub-pytorch-dense-correspondence_latest.sif \
  ~/code/dense_correspondence/training/train.sh
