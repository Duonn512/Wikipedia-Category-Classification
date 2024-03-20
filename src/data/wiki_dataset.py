
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class WikiDataset(Dataset):
    def __init__(self, data_path, train, transform=None):
        all_df = pd.read_csv(data_path)
        all_df = all_df.dropna()
        all_idx = np.arange(0, len(all_df))
        np.random.shuffle(all_idx)
        
        train_size = int(0.8 * len(all_df))

        idx = all_idx[:train_size] if train else all_idx[train_size:]
        self.data_frame = all_df.iloc[idx]
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):

        data_at_index = self.data_frame.iloc[index]
        description = data_at_index['content']
        label = data_at_index['label']
        
        if self.transform:
            description = self.transform(description)
        
        return description, label

    def get_data_loader(self, batch_size=32, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)