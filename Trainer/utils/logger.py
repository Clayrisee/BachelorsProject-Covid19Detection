import logging
import logging.config
import yaml

try:
    with open('configs/logging_config.yaml', 'r') as f:
        conf = yaml.safe_load(f.read())
        logging.config.dictConfig(conf)
        logging.captureWarnings(True)
except:
    with open('Trainer/configs/logging_config.yaml', 'r') as f:
        conf = yaml.safe_load(f.read())
        logging.config.dictConfig(conf)
        logging.captureWarnings(True)

def get_logger(name: str):
    """Logs a Message
    Args:
        Name(str): name of logger
    Return:
        (logger): Named logger
    """
    logger = logging.getLogger(name=name)
    return logger