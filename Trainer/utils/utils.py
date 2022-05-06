import yaml
from torch import device
from torch import optim
import os
from easydict import EasyDict as edict

def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = edict(yaml.safe_load(rf))
        return cfg

def get_device(cfg):
    """ Get device based on configuration
    Args: 
        cfg (dict): a dict of configuration
    Returns:
        torch.device
    """
    result_device = None
    if cfg['device'] == 'cpu':
        result_device = device("cpu")
    elif cfg['device'] == 'cuda:0':
        result_device = device("cuda:0")
    elif cfg['device'] == 'cuda:1':
        result_device = device("cuda:1")
    else:
        raise NotImplementedError
    return result_device

def get_optimizer(cfg, network):
    """ Get optimizer based on the configuration
    Args:
        cfg (dict): a dict of configuration
        network: network to optimize
    Returns:
        optimizer 
    """
    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'])

    elif cfg['train']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(network.parameters(), lr=cfg['train']['lr'])

    elif cfg['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=cfg['train']['lr'])

    elif cfg['train']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(network.parameters(), lr=cfg['train']['lr']) 

    else:
        raise NotImplementedError
    
    return optimizer

def generate_hyperparameters(train_cfg:dict):
    """
    Funtion to generate hyperparameters to log all training process in Comet ML.
    Args:
        train_cfg (dict): a dict of configuration
    Return:
        hyperparameters that will be logged in CometML
    """
    hyperparameters = {
        'model': train_cfg.model.base,
        'model_input': train_cfg.model.input_size[0],
        'output_class': train_cfg.model.num_classes,
        'batch_size': train_cfg.train.batch_size,
        'optimizer': train_cfg.train.optimizer,
        'learning_rate': train_cfg.train.lr,
        'epoch': train_cfg.train.num_epochs
    }
    return hyperparameters

def generate_model_config(train_cfg:dict):
    """
    Function to generate model config file for inference engine.
    Args:
        train_cfg (dict): a dict of configuration
    """
    model_config = {
        'model':train_cfg.model.base,
        'model_input': train_cfg.model.input_size[0],
        'output_class': train_cfg.model,
        'model_file': 'best_model.pth'
    }
    save_dir = train_cfg['output_dir']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, 'model-config.yaml')
    with open(save_path, "w") as yaml_file:
        yaml.safe_dump(model_config, yaml_file, default_flow_style=None, sort_keys=False, explicit_start=True)