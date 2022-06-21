import os
import argparse
import scipy.io as sio
import numpy as np
import h5py

from tqdm import tqdm


import pandas as pd
from random import sample, randint

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import _init_paths
from mylib.voxel_transform import rotation

def save_dataset(path, X_train, Y_train, X_test, Y_test,X_train_ACS, Y_train_ACS):
    f = h5py.File(path, 'w')
    f.create_dataset('test/X', data=X_test)
    f.create_dataset('test/Y', data=Y_test)
    f.create_dataset('trainACS/Y', data=Y_train_ACS)
    f.create_dataset('train/X', data=X_train)
    f.create_dataset('train/Y', data=Y_train)
    f.create_dataset('trainACS/X', data=X_train_ACS)



def sample_well_locations(n_wells, x_range, y_range):
    well_locations = []
    x_coords = np.random.choice(x_range, n_wells)
    y_coords = np.random.choice(y_range, n_wells)

    for i in range(n_wells):
        well_locations.append((x_coords[i], y_coords[i]))
    return well_locations


def main(seed=randint(0,100), n_wells=10, image_size=32, out_filename='stanford6_32.h5'):
    print('Creating dataset...\n')

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'data\\raw\stanford6_truncated.mat')
    dataset_path = os.path.join(
        dirname, 'data\\interim', out_filename)
    
    pathtest=os.path.join(dirname, 'data\\test')
    pathtrain=os.path.join(dirname,'data\\train')
    pathtrain_voxel=os.path.join(dirname,'data\\train_voxel')
    pathtest_voxel=os.path.join(dirname,'data\\test_voxel')
    
    if not os.path.exists(pathtrain):
        os.makedirs(pathtrain)
    
    if not os.path.exists(pathtest):
        os.makedirs(pathtest)    
    if not os.path.exists(pathtrain_voxel):
        os.makedirs(pathtrain_voxel) 
    if not os.path.exists(pathtest_voxel):
        os.makedirs(pathtest_voxel)
      
        
    
    data = sio.loadmat(filename)
    seismic_cube = data['sismica_input']
    facies_cube = data['facies_output']

    height, width, depth = seismic_cube.shape

    x_range = width - image_size + 1
    y_range = depth - image_size + 1
    z_range = height - image_size + 1

    np.random.seed(seed)
    well_locations = sample_well_locations(n_wells, x_range, y_range)
    print(well_locations)
    m_train = n_wells * z_range
    m_test = z_range * x_range * y_range

    seismic_cube -= np.min(seismic_cube)
    seismic_cube /= np.max(seismic_cube)
    seismic_cube *= 255

    X_train = np.empty((m_train, image_size, image_size, 3), dtype='uint8')
    Y_train = np.empty((m_train, 1), dtype='int8')
    X_train_ACS= np.empty((m_train,3, image_size, image_size), dtype='uint8')
    Y_train_ACS= np.empty((m_train,3, image_size, image_size), dtype=bool)
    X_test = np.empty((m_test, image_size, image_size,3), dtype='uint8')
    Y_test = np.empty((m_test, 1), dtype='int8')
    #X_test_ACS= np.zeros((image_size,image_size,image_size),dtype=bool)
    #Y_test_ACS= np.zeros((image_size, image_size, image_size), dtype=bool)
    
    mid_idx = int(np.median(range(image_size)))

    i, j,w,v = 0, 0,0,0

    for x in range(0,x_range, image_size//8):
        for y in range(0,y_range,image_size//8):
            for z in range(0,z_range,image_size//8):
                voxel = seismic_cube[
                    z:z+image_size,
                    x:x+image_size,
                    y:y+image_size,
                    ]
                segs= facies_cube[
                    z:z+image_size,
                    x:x+image_size,
                    y:y+image_size,
                    ]
   
                    
                    
            np.savez(os.path.join(pathtest_voxel,'voxel_{}.npz'.format(w)),voxel=voxel,segs=segs)
            w +=1  
            

        for z in range(0,z_range,image_size//4):
                
            if (x, y) in well_locations:
        
                voxel = seismic_cube[
                z:z+image_size,
                x:x+image_size,                    
                y:y+image_size,
                ]
              
                segs = facies_cube[
                z:z+image_size,
                x:x+image_size,                    
                y:y+image_size,
                ]            
                    
                
            np.savez(os.path.join(pathtrain_voxel,'voxel_{}.npz'.format(v)),voxel=voxel,segs=segs)                 
            v+=1
    i, j = 0, 0            
    for x in range(0,x_range):
        for y in range(0,y_range):
            for z in range(0,z_range):
                sample = seismic_cube[
                    z:z+image_size,
                    x:x+image_size,
                    y:y+image_size,
                ]

                image = np.moveaxis(
                    np.array([
                        sample[mid_idx, :, :],  # Red    -> height
                        sample[:, mid_idx, :],  # Green  -> width
                        sample[:, :, mid_idx],  # Blue   -> depth
                    ]),
                    0,
                    -1
                )
                
                 
                imageACS =np.array([
                        sample[mid_idx, :, :],  # Red    -> height
                        sample[:, mid_idx, :],  # Green  -> width
                        sample[:, :, mid_idx],  # Blue   -> depth
                    ])
                
                sampleACS = facies_cube[
                    z:z+image_size,
                    x:x+image_size,
                    y:y+image_size,
                ]

                faciesACS = np.array([
                        sampleACS[mid_idx, :, :],  # Red    -> height
                        sampleACS[:, mid_idx, :],  # Green  -> width
                        sampleACS[:, :, mid_idx],  # Blue   -> depth
                    ])
                

                facies = facies_cube[
                    z:z+image_size,
                    x:x+image_size,
                    y:y+image_size,
                ][mid_idx, mid_idx, mid_idx]

                X_test[i] = image
                Y_test[i] = facies

                i += 1
                
                if (x, y) in well_locations:
                    X_train[j] = image
                    Y_train[j] = facies
                    X_train_ACS[j]=imageACS
                    Y_train_ACS[j]=faciesACS
                    
                    np.savez(os.path.join(pathtrain,'shape_{}.npz'.format(j)),shape=imageACS,segs=faciesACS)
                    j += 1
    print(f'Seed: {seed}')
    print(f'Input shape: {width, depth, height} (x, y, z)')
    print(f'#Wells sampled: {n_wells}')
    print(f'Sampled well locations:\n{well_locations}\n')

    print(f'X_train shape: {X_train.shape}')
    print(f'Y_train shape: {Y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'Y_test shape: {Y_test.shape}\n')

    print(f'Saving dataset as "{dataset_path}"\n...')
    save_dataset(
        dataset_path,
        X_train, Y_train,
        X_test, Y_test,
        X_train_ACS,Y_train_ACS
        
    )
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'n_wells',
        type=int,
        help='number of wells to sample training data'
    )
    parser.add_argument(
        'image_size',
        type=int,
        help='size of dataset images (image_size x image_size)'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        dest='seed',
        help='RNG initial seed'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='stanford6_truncated_rgb.h5',
        dest='out_filename',
        help='Output file name'
    )
    args = parser.parse_args()

    main(
        seed=args.seed,
        n_wells=args.n_wells,
        image_size=args.image_size,
        out_filename=args.out_filename
    )
