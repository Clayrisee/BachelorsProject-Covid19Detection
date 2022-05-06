import os
from comet_ml import Artifact, Experiment
import torch
from data.covid_dataloader import CovidDataModule
from utils.utils import generate_model_config, read_cfg, get_optimizer, get_device, generate_hyperparameters
from models.models import create_model
from trainer.Trainer import Trainer
from utils.schedulers import CosineAnealingWithWarmUp
from utils.callbacks import CustomCallback
from utils.logger import get_logger
import argparse
import torch.nn as nn
import pandas as pd

def count_weighted(csv):
    df = pd.read_csv(csv)
    weight = list()
    total_class = len(df.groupby('folder').count().filename)
    total_files = len(df)
    for total_files_in_class in df.groupby('folder').count().filename:
        # print(total_files_in_class)
        # print(total_files)
        w = total_files / (total_class * total_files_in_class)
        # print(w)
        weight.append(w)
    return torch.Tensor(weight)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument for train the model")
    parser.add_argument('-cfg', '--config', type=str, help="Path to config yaml file")
    args = parser.parse_args()
    cfg = read_cfg(cfg_file=args.config)
    hyperparameters = generate_hyperparameters(cfg)
    LOG = get_logger(cfg['model']['base']) # create logger based on model name for Track each proses in console

    LOG.info("Training Process Start")
    logger = Experiment(api_key=cfg['logger']['api_key'], 
            project_name=cfg['logger']['project_name'],
            workspace=cfg['logger']['workspace']) # logger for track model in Comet ML
    artifact = Artifact("Covid-19 Artifact", "Model")
    LOG.info("Comet Logger has successfully loaded.")

    device = get_device(cfg)
    LOG.info(f"{str(device)} has choosen.")

    kwargs = dict(pretrained=cfg['model']['pretrained'], output_class=cfg['model']['num_classes'])
    network = create_model(cfg['model']['base'], **kwargs)
    print(network)
    LOG.info(f"Network {cfg['model']['base']} succesfully loaded.")

    optimizer = get_optimizer(cfg, network)
    LOG.info(f"Optimizer has been defined.")

    lr_scheduler = CosineAnealingWithWarmUp(optimizer, 
        first_cycle_steps=250, 
        cycle_mult=0.5,
        max_lr=1e-2, 
        min_lr=cfg['train']['lr'], 
        warmup_steps=100, 
        gamma=0.5)

    LOG.info(f"Scheduler has been defined.")
    weight = count_weighted(os.path.join(cfg.dataset.root_dir,cfg.dataset.train_csv))
    weight = weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
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
    logger.log_parameters(hyperparameters)
    LOG.info("Parameters has been Logged")
    generate_model_config(cfg)
    LOG.info("Model config has been generated")

    if cfg['model']['pretrained_path'] is not None:
        net_state_dict = torch.load(cfg['model']['pretrained_path'], map_location=device)
        network = network.load_state_dict(state_dict=net_state_dict)
    
    if cfg['optimizer']['pretrained_path'] is not None:
        opt_state_dict = torch.load(cfg['optimizer']['pretrained_path'], map_location=device)
        optimizer = optimizer.load_state_dict(opt_state_dict)
    
    trainer = Trainer(cfg, network, optimizer, criterion, dataset, device, callbacks=custom_cb, lr_scheduler=lr_scheduler, logger=logger)

    trainer.train()
    best_model_path = os.path.join(cfg['output_dir'], 'best_model.pth')
    best_optimizer_path = os.path.join(cfg['output_dir'], 'best_optimizer.pth')
    final_model_path = os.path.join(cfg['output_dir'], 'final_model.pth')
    final_optimizer_path = os.path.join(cfg['output_dir'], 'final_optimizer.pth')
    model_cfg_path = os.path.join(cfg['output_dir'], 'model-config.yaml')
    artifact.add(best_model_path)
    artifact.add(best_optimizer_path)
    artifact.add(final_model_path)
    artifact.add(final_optimizer_path)
    artifact.add(model_cfg_path)
    logger.log_artifact(artifact=artifact)
