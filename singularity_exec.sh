#!/usr/bin/env bash

# Default values
image=/results/$USER/gdon_latest.sif
data=/scidatalg/mlp2020_descriptors
entrypoint=dense_correspondence/training/training_script_gdon.py
train_config=config/dense_correspondence/training/training.yaml
data_config=config/dense_correspondence/dataset/composite/caterpillar_upright.yaml

while getopts i:d:e:t:c: option
do
  case "${option}" in
  i) image=${OPTARG};;
  d) data=${OPTARG};;
  e) entrypoint=${OPTARG};;
  t) train_config=${OPTARG};;
  c) data_config=${OPTARG};;
  *) echo "usage: $0 [-v] [-r]" >&2
       exit 1 ;;
  esac
done

set -x
singularity exec --nv \
  --bind $data:/home/$USER/data \
  --bind /results/$USER/general-dense-object-nets:/home/$USER/code \
  $image \
  ~/code/dense_correspondence/training/train.sh $entrypoint $train_config $data_config