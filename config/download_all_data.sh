#!/usr/bin/env bash
#
#SBATCH --job-name=gdon-data-download
#SBATCH --partition=common
#SBATCH --qos=8gpu3d
#SBATCH --gres=gpu:0
#SBATCH --nodelist=asusgpu4
#SBATCH --output=/results/data-download-log.txt

code_dir="/results/$USER/general-dense-object-nets"
config_files="/$code_dir/config/dense_correspondence/dataset/composite/*"
data_dir="/scidatalg/mlp2020_descriptors"

mkdir -p $data_dir

# Python env with pyyaml installed
source "/results/$USER/pyenv2/bin/activate"
command -v python
cd $code_dir || exit
pwd
for config in $config_files
do
  echo "Downloading data for $config"
  python "$code_dir/config/download_pdc_data.py" "$config" "$data_dir"
done