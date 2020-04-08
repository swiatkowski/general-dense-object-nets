#!/bin/bash
# TODO(swiatkowski): support reading SIFs from both results (fast) and scidatasm (large)
srun --partition=common --qos=8gpu3d --gres=gpu:1 \
        singularity exec --nv \
        --bind /scidatasm/dense_object_nets/data:/home/$USER/data \
        --bind /results/$USER/general-dense-object-nets:/home/$USER/code \
        /results/$USER/gdon_latest.sif \
        bash ~/code/dense_correspondence/training/train.sh
