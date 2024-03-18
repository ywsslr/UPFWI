import os
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils import data
import torch.utils.data.dataloader as DataLoader


# define my dataset
class FWI_Dataset(data.Dataset):
    def __init__(self, dir_path,transform_data=None,transfrom_label=None):
        self.dir_path = dir_path
        self.transform_data=transform_data
        self.transform_label=transfrom_label

    def __len__(self):
        return len(os.listdir(self.dir_path))

    def __getitem__(self, index):
        # Select sample
        file_name = f'data_{index}' 
        # Load data
        try:
            data = torch.load(os.path.join(self.dir_path,file_name))
        except:
            os.remove(os.path.join(self.dir_path,file_name))
        seis_data = data['seis_data']
        velocity = data['velocity']
        if self.transform_data:
            seis_data=self.transform_data(seis_data)
        if self.transform_label:
            velocity=self.transform_label(velocity)
        return seis_data, velocity
    
# load my data
def load_data(dir_path, batch_size, shuffle=None, transform_data=None, transform_label=None, drop_last=False):
    data = FWI_Dataset(dir_path, transform_data, transform_label)
    data_loader = DataLoader.DataLoader(data, batch_size, shuffle, num_workers=2, drop_last=drop_last)
    return data_loader


if __name__ == '__main__':
    #  train_data = FWI_Dataset('/data1/lr/FWIDATA/train/')
    dir_path = '/data1/lr/FWIDATA/train/'
    batch_size = 16
    train_loader = load_data(dir_path, batch_size)
    test_loader = load_data('/data1/lr/FWIDATA/test/',batch_size)
    (seis_data_batch, v_batch) = next(iter(train_loader))
    print(f'the length of the dataloader:{len(train_loader)}\n'
      f'the shape of the seis_data:{seis_data_batch.shape}\n'
      f'the shape of the v_batch:{v_batch.shape}')
