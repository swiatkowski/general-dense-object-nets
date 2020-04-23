import matplotlib
matplotlib.use('Agg')

import os
import time
import argparse

import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset

utils.set_cuda_visible_devices([0])

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for train/test script')
    parser.add_argument('--loss', type=str, default='pixelwise_contrastive_loss')
    parser.add_argument('--bg-frac', type=float, default=0.5)
    parser.add_argument('--mask-frac', type=float, default=0.5)
    parser.add_argument('--num-iterations', type=int, default=3500)
    parser.add_argument('--sampler', type=str, default='don')
    parser.add_argument('--mask-weight', type=int, default=1)
    parser.add_argument('--bg-weight', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
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
    train_config["training"]["num_iterations"] = args.num_iterations

    train_config['loss_function']['name'] = args.loss
    train_config['loss_function']['nq'] = 25
    train_config['loss_function']['num_samples'] = 150
    train_config['loss_function']['sampler']['name'] = args.sampler
    train_config['loss_function']['sampler']['mask_weight'] = args.mask_weight
    train_config['loss_function']['sampler']['background_weight'] = args.bg_weight


    train_config["training"]["save_rate"] = 5000
    train_config["training"]["fraction_background_non_matches"] = float(args.bg_frac)
    train_config["training"]["fraction_masked_non_matches"] = float(args.bg_frac)

    # train
    train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
    train.run()
