#!/bin/bash
# Currently, all data was downloaded only to asusgpu4, sylvester, arnold.
srun --partition=common --qos=8gpu3d --gres=gpu:1 --cpus-per-task 8 \
  --exclude=asusgpu1,asusgpu2,asusgpu3,asusgpu5,steven,bruce \
  bash /results/$USER/general-dense-object-nets/singularity_exec.sh
