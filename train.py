# %%
import sys
sys.path.append('D:/supernova/utils')
import numpy as np
import utils.select_id as select_id
import utils.cluster as clst
import utils.slice as slc
import os
from os import listdir
import torch


# %%
data_path = 'event'
file_list = listdir(data_path)
file_list.sort()

# %%
from torch.utils.data import Dataset, DataLoader, TensorDataset
class MyDataset(Dataset):
    def __init__(self, data_path_, file_list_):
        self.data_path = data_path_
        self.file_list = file_list_
        self.len = len(file_list_)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        data = np.load( os.path.join( self.data_path, file_name ) )
        return torch.from_numpy(data['imxz'][None, :, :]).to(torch.float)/4096, torch.from_numpy(data['imyz'][None, :, :]).to(torch.float)/4096, torch.from_numpy(data['sig']).to(torch.long)
    def __len__(self):
        return self.len