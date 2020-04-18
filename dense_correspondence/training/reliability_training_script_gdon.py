# Prevent no $DISPLAY environment variable error from tkinter
import matplotlib

matplotlib.use('Agg')

import dense_correspondence_manipulation.utils.utils as utils

utils.add_dense_correspondence_to_python_path()
from dense_correspondence.training.training import *
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

# Private account
# train_config["logging"]["namespace"] = "jkopanski"

# probabilistic
# train_config["loss_function"]["name"] = "probabilistic_loss"
# train_config["logging"]["namespace"] = "jkopanski"
# train_config["logging"]["experiment"] = "caterpillar"
# train_config["logging"]["description"] = "probabilistic_loss_add_conv_softplus_lr1e-3"
# train_config["logging"]["tags"] = ['general-dense-object-nets', 'jkopanski', 'probabilistic_loss']
# train_config["logging"]["qualitative_evaluation_logging_rate"] = 500

# pixelwise_contrastive_loss
# train_config["loss_function"]["name"] = "pixelwise_contrastive_loss"
# train_config["logging"]["namespace"] = "jkopanski"
# train_config["logging"]["experiment"] = "caterpillar"
# train_config["logging"]["description"] = "pixelwise_contrastive_loss"
# train_config["logging"]["tags"] = ['general-dense-object-nets', 'jkopanski', 'pixelwise_contrastive_loss']

# aploss
train_config["loss_function"]["name"] = "aploss"
train_config["logging"]["experiment"] = "caterpillar"
train_config["logging"]["description"] = 'aploss_reliability_4'
train_config["logging"]["tags"] = ['general-dense-object-nets', 'jkopanski', 'aploss']

# Common for all loss functions
# train_config["dense_correspondence_network"]["reliability"] = True
train_config["dense_correspondence_network"]["descriptor_dimension"] = 3
train_config["training"]["logging_dir_name"] = "{}_{}_{}".format(
    train_config["logging"]["experiment"],
    train_config["dense_correspondence_network"]["descriptor_dimension"],
    train_config["logging"]["description"])
# train_config["training"]["logging_dir"] = os.path.join(os.environ['HOME'], 'models', 'trained_models')
train_config["training"]["logging_dir"] = 'trained_models'
train_config["training"]["num_iterations"] = 5000
train_config["training"]["learning_rate"] = 1e-4

TRAIN = True
EVALUATE = True

if TRAIN:
    train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
    train.run()

code_dir = os.environ['HOME'] + '/code'
model_folder = os.path.join(code_dir, train_config["training"]["logging_dir"], train_config["training"]["logging_dir_name"])
model_folder = utils.convert_data_relative_path_to_absolute_path(model_folder)

# model_folder = os.path.join(train_config["training"]["logging_dir"], train_config["training"]["logging_dir_name"])

if EVALUATE:
    DCE = DenseCorrespondenceEvaluation
    num_image_pairs = 100
    DCE.run_evaluation_on_network(model_folder, num_image_pairs=num_image_pairs)
