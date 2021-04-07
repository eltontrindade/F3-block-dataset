import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class BaseDatasetShape(Dataset):
    def __init__(self, data_root, num_data, canal):
        self.data_root = data_root
        self.len = num_data
        self.canal=canal
    def __getitem__(self, index):
        data = np.load(os.path.join(self.data_root,'shape_{}.npz'.format(index)))
        shape = data['shape']
        segs = data['segs']

        
        shape=shape[self.canal,:,:,:]
        
        segs= keras.utils.to_categorical(data['segs'], 6)
        segs=segs[self.canal,:,:,:]
        
        segs=np.moveaxis(segs,-1,0)

        
        
        return torch.from_numpy(shape).float(), torch.from_numpy(segs.astype(float)).float()
    def __len__(self):
        return self.len
        
class BaseDatasetVoxel(Dataset):
    def __init__(self, data_root, num_data):
        self.data_root = data_root
        self.len = num_data        
    def __getitem__(self, index):
        data = np.load(os.path.join(self.data_root,'voxel_{}.npz'.format(index)))
        voxel = data['voxel']
        segs = data['segs']
        segs= keras.utils.to_categorical(data['segs'], 6)
        segs=np.moveaxis(segs,-1,0)
        return torch.from_numpy(voxel).float(), torch.from_numpy(segs.astype(float)).float()
    def __len__(self):
        return self.len