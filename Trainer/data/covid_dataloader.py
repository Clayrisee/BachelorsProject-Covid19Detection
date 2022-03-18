from data.base import DataModuleBase
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class CovidDataModule(DataModuleBase):
    def __init__(self, cfg):
        
        super().__init__(cfg)

        self.train_bs = cfg['train']['batch_size']
        self.val_bs = cfg['val']['batch_size']
        self.test_bs = cfg['test']['batch_size']
        self.map_size = cfg['model']['map_size']

        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(cfg['dataset']['rotation_degree']),
            transforms.Resize(cfg['model']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['std'])
        ])

        self.test_val_transforms = transforms.Compose([
            transforms.Resize(cfg['model']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['std'])
        ])

        self.prepare_dataset(cfg=cfg)
    
    def prepare_dataset(self, cfg):
        self.train_set = ImageFolder(root=cfg['dataset']['train_root'], transform=self.train_transforms)
        self.val_set = ImageFolder(root=cfg['dataset']['val_root'], transform=self.test_val_transforms)
        self.test_set = ImageFolder(root=cfg['dataset']['test_root'], transform=self.test_val_transforms)

    def train_dataloader(self):
        kwargs = dict(
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=2
        )
        self.train_dl = DataLoader(self.train_set, **kwargs)
        return self.train_dl

    def val_dataloader(self):
        kwargs = dict(
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=2
        )
        self.val_dl = DataLoader(self.val_set, **kwargs)
        return self.val_dl

    def test_dataloader(self):
        kwargs = dict(
            batch_size=self.test_bs,
            shuffle=True,
            num_workers=2
        )
        self.test_dl = DataLoader(self.test_set, **kwargs)
        return self.test_dl