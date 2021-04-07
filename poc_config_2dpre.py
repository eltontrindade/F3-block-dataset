from datetime import datetime
import os
import sys
import fire
import time
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from unet0 import UNet as UNet0
from unet1 import UNet as UNet1
from unet2 import UNet as UNet2 



class POCVoxelEnv():
    data_train = r'D:\Elton\F3\data\train_voxel'
    data_test = r'D:\Elton\F3\data\test_voxel'
    data_eval = r'D:\Elton\F3\data\eval_voxel'
    data_eval2 = r'D:\Elton\F3\data\eval_voxel2'
    shape_checkpoint= r'C:\Users\admin\Desktop\cnn-facies-classifier-master\tmp\shape_880samples\model.dat' # put the 2D model checkpoint here.

class POCShapeEnv():
    data_test = r'D:\Elton\F3\data\train2d'
    data_train = r'D:\Elton\F3\data\train2d'


class POCShapeConfig():
    # default config
    batch_size = 64
    n_epochs = 50
    drop_rate = 0.0
    seed = None
    num_workers = 0

    # optimizer
    lr = 0.01
    wd = 0.001
    momentum=0.9

    # scheduler
    milestones = [0.5 * n_epochs, 0.75 * n_epochs]
    gamma=0.1
    save_all = False

    # model 
    train_samples = len(os.listdir(r'D:\Elton\F3\data\train2d'))
    test_samples=len(os.listdir(r'D:\Elton\F3\data\train2d'))

    
    
    flag = '_{}samples'.format(train_samples)
    save = r'C:\Users\admin\Desktop\Elton\F3\ensemblemodels'

class POCVoxelConfig():
    # default config
    train_batch_size = 13
    test_batch_size = 13
    n_epochs = 100
    drop_rate = 0.0
    seed = 0
    num_workers = 4

    # optimizer
    lr = 0.01
    wd = 0.0001
    momentum=0.9

    bg_loss = 0.1
    focal_gamma = 2

    # scheduler
    milestones = [30]
    gamma=0.1
    save_all = False

    train_samples = len(os.listdir(r'D:\Elton\F3\data\train_voxel'))
    test_samples=len(os.listdir(r'D:\Elton\F3\data\test_voxel'))    
    eval_samples=len(os.listdir(r'D:\Elton\F3\data\eval_voxel'))
    eval_samples2=len(os.listdir(r'D:\Elton\F3\data\eval_voxel2'))
    noise = 0.5

    # conv = 'Conv3D'
    
    #conv = 'ACSConv'    
    #pretrained = True

    conv = 'Conv2_5D'
    pretrained = True

    flag = '_{}samples'.format(train_samples)

    save = os.path.join(r'C:\Users\admin\Desktop\Elton\F3\ensemblemodels', conv)



class mergemodels(): #carrega cada um dos treinamentos 2 D na sua rede geometricamente posicionados no seus eixos ortogonais
    
    groupmodel=[]

    redes=[UNet0, UNet1, UNet2]
    num_classes=6

    for i in range(3):

        shape_cp= r'C:\Users\admin\Desktop\Elton\F3\ensemblemodels\canal_'+ format(i)+ '\model.dat'
        shape_cp = torch.load(shape_cp)
        model= redes[i](num_classes)
        #print(shape_cp.keys())
        #state_dict_2d=shape_cp.state_dict()
        for key in list(shape_cp.keys()):
 
            if shape_cp[key].dim()==4:

                #print(key)
                shape_cp[key] = shape_cp[key].unsqueeze(i-3)
                #print(shape_cp[key].shape)    

        shape_cp.popitem()
        shape_cp.popitem()
        #print(shape_cp.keys)
        model.load_state_dict(shape_cp, strict=False)

        groupmodel.append(model)
    

