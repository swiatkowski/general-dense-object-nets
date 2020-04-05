from dense_correspondence.logging.NeptuneLogger import NeptuneLogger


def dispatch_logger(config):
    if config['logging']['backend'] == 'neptune':
        return NeptuneLogger(config)
    elif config['logging']['backend'] == 'tensorboard':
        raise NotImplementedError('No support for "Tensorboard" yet')
    elif config['logging']['backend'] == 'wandb':
        raise NotImplementedError('No support for "Weights & Biases" yet')
    elif config['logging']['backend'] == 'file':
        raise NotImplementedError('No support for files yet')
    else:
        raise Exception("Logging backend: {} not recognized. Supported types are: [neptune]".format(type))
