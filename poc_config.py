from datetime import datetime
import os
import sys
from random import randrange


class POCVoxelEnv():
    data_train = r'D:\Elton\F3\data\train_voxel'
    data_test = r'D:\Elton\F3\data\test_voxel'
    shape_checkpoint= r'C:\Users\admin\Desktop\cnn-facies-classifier-master\tmp\shape_880samples\model.dat' # put the 2D model checkpoint here.

class POCShapeEnv():
    data_test = r'D:\Elton\F3\data\train2d'
    data_train = r'D:\Elton\F3\data\train2d'

class POCShapeConfig():
        
    # default config
    batch_size = 64
    n_epochs = 5
    drop_rate = 0.0
    seed = None
    num_workers = 0

    # optimizer
    lr = 0.01
    wd = 0.0001
    momentum=0.9

    # scheduler
    milestones = [0.5 * n_epochs, 0.75 * n_epochs]
    gamma=0.1
    save_all = False

    # model 
    train_samples = len(os.listdir(r'D:\Elton\F3\data\train2d'))
    test_samples=len(os.listdir(r'D:\Elton\F3\data\train2d'))

    noise = randrange(5)
    
    flag = '_{}samples'.format(train_samples)
    save = r'C:\Users\admin\Desktop\Elton\F3\ensemblemodels'

class POCVoxelConfig():
    # default config
    train_batch_size = 4
    test_batch_size = 20
    n_epochs = 50
    drop_rate = 0.0
    seed = 0
    num_workers = 4

    # optimizer
    lr = 0.001
    wd = 0.0001
    momentum=0.9

    bg_loss = 0.1
    focal_gamma = 2

    # scheduler
    milestones = [5000]
    gamma=0.1
    save_all = False


    train_samples=len(os.listdir(r'D:\Elton\F3\data\train_voxel'))
    test_samples=len(os.listdir(r'D:\Elton\F3\data\test_voxel'))
    noise = 0.1

    # conv = 'Conv3D'
    
    conv = 'ACSConv'    
    pretrained = True

    # conv = 'Conv2_5D'
    # pretrained = True

    flag = '_{}samples'.format(test_samples)
    save = os.path.join(sys.path[0], './tmp', 'voxel', conv +flag)
