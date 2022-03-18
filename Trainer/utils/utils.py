import yaml
from torch import device

def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
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