#!/bin/bash
# Currently, all data was downloaded only to asusgpu4, sylvester, arnold.
srun --partition=common --qos=8gpu3d --gres=gpu:1 --ntasks=1 \
  --nodelist=asusgpu4,sylvester,arnold --cpus-per-task 8 \
  bash /results/$USER/general-dense-object-nets/singularity_exec.sh
