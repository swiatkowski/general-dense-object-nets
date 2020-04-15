import os

import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
utils.add_dense_correspondence_to_python_path()




if __name__ == '__main__':
    src_dir = utils.getDenseCorrespondenceSourceDir()
    configs_dir = 'data/pdc/trained_models/shoes_consistent_M_background_0.500_3'

    # create dataset
    config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), configs_dir, 'dataset.yaml')
    config = utils.getDictFromYamlFilename(config_filename)
    dataset = SpartanDataset(config_expanded=config)

    # training config
    train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), configs_dir, 'training.yaml')
    train_config = utils.getDictFromYamlFilename(train_config_file)
    logging_dir = "trained_models"
    train_config["training"]["logging_dir_name"] = 'shoes_aploss_3d'
    train_config["training"]["logging_dir"] = logging_dir

    train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
    train.run()
