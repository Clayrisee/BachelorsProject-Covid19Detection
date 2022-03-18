import os
from comet_ml import Experiment
from data.covid_dataloader import CovidDataModule
from utils.utils import read_cfg, get_optimizer, get_device
from models.models import create_model
from trainer.Trainer import Trainer
from utils.schedulers import CosineAnealingWithWarmUp
from utils.callbacks import CustomCallback
from utils.logger import get_logger
import argparse
import torch.nn as nn



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument for train the model")
    parser.add_argument('-cfg', '--config', type=str, help="Path to config yaml file")
    args = parser.parse_args()
    cfg = read_cfg(cfg_file=args.config)
    LOG = get_logger(cfg['model']['base']) # create logger based on model name for Track each proses in console

    LOG.info("Training Process Start")
    logger = Experiment(api_key=cfg['logger']['api_key'], 
            project_name=cfg['logger']['project_name'],
            workspace=cfg['logger']['workspace']) # logger for track model in Comet ML
    LOG.info("Comet Logger has successfully loaded.")

    device = get_device(cfg)
    LOG.info(f"{str(device)} has choosen.")

    kwargs = dict(pretrained=cfg['model']['pretrained'], output_class=cfg['model']['output_class'])
    network = create_model(cfg['model']['base'], **kwargs)
    print(network)
    LOG.info(f"Network {cfg['model']['base']} succesfully loaded.")

    optimizer = get_optimizer(cfg, network)
    LOG.info(f"Optimizer has been defined.")

    lr_scheduler = CosineAnealingWithWarmUp(optimizer, 
        first_cycle_steps=300, 
        cycle_mult=1.0,
        max_lr=0.1, 
        min_lr=0.001, 
        warmup_steps=50, 
        gamma=0.5)
    LOG.info(f"Scheduler has been defined.")

    criterion = nn.CrossEntropyLoss()
    LOG.info(f"Criterion has been defined")

    dataset = CovidDataModule(cfg)
    LOG.info(f"Dataset successfully loaded.")
    cb_config = dict(
        checkpoint_path=cfg['output_dir'],
        patience=cfg['custom_cb']['patience'],
        metric=cfg['custom_cb']['metric'],
        mode=cfg['custom_cb']['mode']
    )
    custom_cb = CustomCallback(**cb_config)
    LOG.info(f"Custom CB Initialized")
    trainer = Trainer(cfg, network, optimizer, criterion, dataset, device, callbacks=custom_cb, lr_scheduler=lr_scheduler, logger=logger)

    trainer.train()
