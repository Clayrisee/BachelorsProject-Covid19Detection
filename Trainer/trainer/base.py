class BaseTrainer():
    def __init__(self, cfg, network, optimizer, criterion, dataset, lr_scheduler, device, logger):
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset = dataset
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.logger = logger
    
    def load_model(self):
        raise NotImplementedError
    
    def save_model(self):
        raise NotImplementedError
    
    def train_one_epoch(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError