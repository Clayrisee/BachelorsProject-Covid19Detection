import pandas as pd
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import os

class CovidDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        super(CovidDataset, self).__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.transform = transform
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) :

        img_name = self.data.iloc[index, 0]
        format = self.data.iloc[index, 1].lower()
        folder = self.data.iloc[index, 2]
        img_path = os.path.join(self.root_dir, folder, 'images', img_name, format)

        img = Image.open(img_path)
        label = self.data.iloc[index, -1].astype(np.float32)
        label = np.expand_dims(label, axis=0)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label