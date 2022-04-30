

def generate_hyperparameters(train_cfg:dict):
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