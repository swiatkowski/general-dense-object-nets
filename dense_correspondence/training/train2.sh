#!/usr/bin/env bash
source ~/code/docker/entrypoint.sh
use_pytorch_dense_correspondence
source ~/code/config/.env
python dense_correspondence/training/training_script_gdon2.py