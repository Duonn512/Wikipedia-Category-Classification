
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class WikiDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_frame = pd.read_csv(data_path)
        self.data_frame = self.data_frame.dropna()

        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):

        data_at_index = self.data_frame.iloc[index]
        content = data_at_index['content']
        label = data_at_index['label']
        
        if self.transform:
            content = self.transform(content)
        
        return content, label

    def get_data_loader(self, batch_size=32, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)