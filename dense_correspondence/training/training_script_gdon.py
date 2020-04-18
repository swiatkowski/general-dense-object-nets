# Prevent no $DISPLAY environment variable error from tkinter
import matplotlib
matplotlib.use('Agg')

import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.training.training import *
import sys
import logging
import os

from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
logging.basicConfig(level=logging.INFO)

from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation

config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                               'dataset', 'composite', 'caterpillar_upright.yaml')
config = utils.getDictFromYamlFilename(config_filename)

train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                               'training', 'training.yaml')

train_config = utils.getDictFromYamlFilename(train_config_file)
dataset = SpartanDataset(config=config)

logging_dir = "trained_models/tutorials"
num_iterations = 500
d = 3 # the descriptor dimension
name = "caterpillar_%d" %(d)
train_config["training"]["logging_dir_name"] = name
train_config["training"]["logging_dir"] = logging_dir
train_config["dense_correspondence_network"]["descriptor_dimension"] = d
train_config["training"]["num_iterations"] = num_iterations

TRAIN = True
EVALUATE = True

# This statement is not true on Entropy
# All of the saved data for this network will be located in the
# code/data/pdc/trained_models/tutorials/caterpillar_3 folder

if TRAIN:
    print "training descriptor of dimension %d" %(d)
    train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
    train.run()
    print "finished training descriptor of dimension %d" %(d)


code_dir = os.environ['HOME'] + '/code'
model_folder = os.path.join(code_dir, logging_dir, name)
model_folder = utils.convert_data_relative_path_to_absolute_path(model_folder)

if EVALUATE:
    DCE = DenseCorrespondenceEvaluation
    num_image_pairs = 100
    DCE.run_evaluation_on_network(model_folder, num_image_pairs=num_image_pairs)