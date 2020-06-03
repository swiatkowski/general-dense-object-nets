# Prevent no $DISPLAY environment variable error from tkinter
import matplotlib
matplotlib.use('Agg')

import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.training.training import *
import sys
import logging
import os
from time import strftime

from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
logging.basicConfig(level=logging.INFO)

if len(sys.argv) != 3:
    print("{} requires two arguments: path to training config and path to dataset config.".format(
        sys.argv[0]))

train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), sys.argv[1])
data_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), sys.argv[2])

train_config = utils.getDictFromYamlFilename(train_config_file)
data_config = utils.getDictFromYamlFilename(data_config_file)
if '/expanded/' in data_config_file:
    dataset = SpartanDataset(config_expanded=data_config)
else:
    dataset = SpartanDataset(config=data_config)

# Cannot use %X. Neptune doesn't accept colons in tags.
time_string = strftime('%d-%m-%Y_%H-%M-%S')  # %X=clock time (%H:%M:%S), %d day, %m month, %Y year
train_config['training']['logging_dir_name'] = '{0}_{1}'.format(
    train_config['logging']['experiment'].replace(' ', '_'), time_string)

train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
train.run()
