#!/usr/bin/env bash
source ~/code/docker/entrypoint.sh
use_pytorch_dense_correspondence
python dense_correspondence/training/training_script_gdon.py