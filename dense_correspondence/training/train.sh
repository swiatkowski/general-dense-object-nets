#!/usr/bin/env bash
source ~/code/docker/entrypoint.sh
use_pytorch_dense_correspondence
source ~/code/config/.env

if [ -n "$1" ]; then
  entrypoint="$1"
else
  entrypoint=dense_correspondence/training/training_script_gdon.py
fi
echo "Workdir: "
pwd
python $entrypoint "${@:2}"