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

# Some of the data is downloaded directly to $data_dir instead of $data_dir/pdc/logs_proto/.
# Fix this by moving them after all data is downloaded.
mv $data_dir/* $data_dir/pdc/logs_proto/

# Give read permissions to others
find $data_dir -type d -print0 | xargs -0 chmod o=rx
find $data_dir -type f -print0 | xargs -0 chmod o=r
setfacl -R -m u:swiatkowski:rwx $data_dir
setfacl -R -m u:tomasz.gasior:rwx $data_dir