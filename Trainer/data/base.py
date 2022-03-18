class DataModuleBase(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError