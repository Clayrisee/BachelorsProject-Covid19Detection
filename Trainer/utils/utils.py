import yaml
from torch import device
from torch import optim
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
