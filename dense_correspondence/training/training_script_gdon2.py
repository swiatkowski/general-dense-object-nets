import os
import time

import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset

utils.set_cuda_visible_devices([0])


if __name__ == '__main__':
    src_dir = utils.getDenseCorrespondenceSourceDir()
    configs_dir = 'config/dense_correspondence'

    # create dataset
    config_filename = os.path.join(src_dir, configs_dir, 'dataset/composite/shoe_train_4_shoes.yaml')
    config = utils.getDictFromYamlFilename(config_filename)
    dataset = SpartanDataset(config=config)

    # training config
    train_config_file = os.path.join(src_dir, configs_dir, 'training/training.yaml')
    train_config = utils.getDictFromYamlFilename(train_config_file)

    # overwrite training config
    time_string = time.strftime('%X__%d-%m-%Y') # X=clock time ,%d day %m month %Y year
    train_config["training"]["logging_dir_name"] = 'shoes_{}'.format(time_string)
    train_config["training"]["logging_dir"] = "trained_models"
    train_config["dense_correspondence_network"]["descriptor_dimension"] = 3
    train_config["training"]["num_iterations"] = 3500

    train_config['loss_function']['name'] = 'aploss'
    train_config['loss_function']['nq'] = 25
    train_config['loss_function']['num_samples'] = 150
    train_config['loss_function']['sampler']['name'] = 'don'
    train_config['loss_function']['sampler']['mask_weight'] = 1
    train_config['loss_function']['sampler']['background_weight'] = 2

    # train
    train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
    train.run()
