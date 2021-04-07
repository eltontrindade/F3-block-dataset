import os
import argparse
import scipy.io as sio
import numpy as np
import h5py

from tqdm import tqdm

from scipy import ndimage
import random
from random import sample, randint
import pandas as pd


import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import _init_paths
from mylib.voxel_transform import rotation




def sample_well_locations(n_wells, x_range, y_range):
    well_locations = []
    x_coords = np.random.choice(x_range, n_wells)
    y_coords = np.random.choice(y_range, n_wells)

    for i in range(n_wells):
        well_locations.append((x_coords[i], y_coords[i]))
    return well_locations


def main(seed=randint(0,100), n_wells=10,n_iter=1, image_size=32, out_folder=r'D:\Elton\F3'):
    print('Creating dataset...\n')

    dirname = os.path.dirname(__file__)

    dataset_path = os.path.join(
        out_folder, 'data')
   
    #pathtest=os.path.join(dirname, 'data\\test')
    #pathtrain=os.path.join(dirname,'data\\train')
    pathtest= os.path.join(out_folder,'data\\test2d')
    pathtrain=os.path.join(out_folder,'data\\train2d')
    
    pathtrain_voxel=os.path.join(out_folder,'data\\train_voxel')
    pathtest_voxel=os.path.join(out_folder,'data\\test_voxel')
    patheval_voxel=os.path.join(out_folder,'data\\eval_voxel')
    patheval_voxel2=os.path.join(out_folder,'data\\eval_voxel2')
    
    if not os.path.exists(pathtrain):
        os.makedirs(pathtrain)
    
    if not os.path.exists(pathtest):
        os.makedirs(pathtest)    
    if not os.path.exists(pathtrain_voxel):
        os.makedirs(pathtrain_voxel) 
    if not os.path.exists(pathtest_voxel):
        os.makedirs(pathtest_voxel)
    if not os.path.exists(patheval_voxel):
        os.makedirs(patheval_voxel)
    if not os.path.exists(patheval_voxel2):
        os.makedirs(patheval_voxel2)        
      
        
    #seismic_cube = np.load(os.path.join(out_folder,'data\\train', 'tecva.npy'))
    seismic_cube = np.load(os.path.join(out_folder,'data\\train', 'train_seismic.npy'))
    facies_cube = np.load(os.path.join(out_folder,'data\\train', 'train_labels.npy'))
    #seismic_test = np.load(os.path.join(out_folder,'data\\test_once', 'test1_seismic.npy'))
    facies_test = np.load(os.path.join(out_folder,'data\\test_once', 'test1_labels.npy'))    
    
    prof=[]
    for x in range(seismic_cube.shape[0]):
        for y in range(seismic_cube.shape[1]):
            for z in range(seismic_cube.shape[2]):
                prof.append(z)
    

    prof=np.array(prof).reshape(seismic_cube.shape)
    
    pad= tuple((image_size - x%image_size) for x in seismic_cube.shape) # vê o resto da divisão pelo image_size para definir o padding
    print(seismic_cube.shape)
    print(pad)
    seismic_cube=np.pad(seismic_cube, ((0,pad[0]),(0,pad[1]),(0,pad[2])), 'edge') #faz o padding da sismica no final
    print(seismic_cube.shape)
    facies_cube=np.pad(facies_cube, ((0,pad[0]),(0,pad[1]),(0,pad[2])), 'edge') #faz o padding dos labels no final
    prof=np.pad(prof, ((0,pad[0]),(0,pad[1]),(0,pad[2])), 'edge')

    width, depth,height = seismic_cube.shape

    x_range = width - image_size +1
    y_range = depth - image_size +1
    z_range = height - image_size +1

    np.random.seed(seed)
    
    well_locations = sample_well_locations(n_wells, x_range, y_range)    
    
    
    # inline, xline of F3 wells inside training data, corrected to np index
    well_locations = np.asarray([
        (62, 36),
        (18, 160),
        (152, 155),
        (126, 91),
        (53, 130),
        (56, 129),
        (50, 600)
    ])



    seismic_cube -= np.min(seismic_cube)
    seismic_cube /= np.max(seismic_cube)
    seismic_cube *= 255



    
    mid_idx = int(np.median(range(image_size)))

    i, j,w,v,e = 0, 0,0,0,0

    for x in range(0,x_range, image_size):
        for y in range(0,y_range,image_size):
            for z in range(0,z_range,image_size):
                voxel = seismic_cube[
                    x:x+image_size,
                    y:y+image_size,
                    z:z+image_size
                    ]
                
                dep=prof[
                    x:x+image_size,
                    y:y+image_size,
                    z:z+image_size
                    ]
                      
                segs= facies_cube[
                    x:x+image_size,
                    y:y+image_size,
                    z:z+image_size,
                    ]
                                 
    
                coin=random.random()
                if coin>0.5:
                


                    #segs1=ndimage.rotate(segs, float(((-1)**e) *20*coin), reshape=False, order=0, mode='reflect')
                    
                    #voxel1=ndimage.rotate(voxel,float(((-1)**e) *20*coin), reshape=False, order=0,mode='reflect')
                    
                    #dep1=ndimage.rotate(dep,float(((-1)**e) *20*coin), reshape=False, order=0,mode='reflect')
                    #e=e+1
                    
                    #voxel1=np.stack([voxel1,dep1], axis=0)
                    
                    #np.savez(os.path.join(pathtrain_voxel,'voxel_{}.npz'.format(v)),voxel=voxel1,segs=segs1)
                    #v=v+1;

                    #segs2=np.fliplr(segs)
                    #voxel2=np.fliplr(voxel)
                    #dep2=np.fliplr(dep)
                    #voxel2=np.stack([voxel2,dep2], axis=0)

                    #np.savez(os.path.join(pathtrain_voxel,'voxel_{}.npz'.format(v)),voxel=voxel2,segs=segs2)
                    #v=v+1; 
                    
                    voxel3=np.stack([voxel,dep], axis=0)
                 
                    
                    np.savez(os.path.join(pathtrain_voxel,'voxel_{}.npz'.format(v)),voxel=voxel3,segs=segs)
                    v +=1                            
                    
               
                voxel=np.stack([voxel,dep], axis=0)
                 
                    
                np.savez(os.path.join(pathtest_voxel,'voxel_{}.npz'.format(w)),voxel=voxel,segs=segs)
                w +=1 

    i, j = 0, 0            
    
    for x, y in well_locations:
        #x -= int(image_size/2) - 1
        #y -= int(image_size/2) - 1
        #print(f'x: {x}, y: {y}')
        for z in range(z_range):


              

            
            
            
            sample = seismic_cube[
                x:x+image_size,
                y:y+image_size,
                z:z+image_size,
            ]
            
            
            dep=prof[
                x:x+image_size,
                y:y+image_size,
                z:z+image_size
                ]
                
            sample=np.stack([sample,dep], axis=0)
            
            
            sampleACS = facies_cube[
                x:x+image_size,
                y:y+image_size,
                z:z+image_size,
            ]              

            image = np.array([
                    sample[:,mid_idx, :, :],  # Red    -> height
                    sample[:,:, mid_idx, :],  # Green  -> width
                    sample[:,:, :, mid_idx],  # Blue   -> depth
                ])


            facies = facies_cube[
                x:x+image_size,
                y:y+image_size,
                z:z+image_size,
            ][mid_idx, mid_idx, mid_idx]


            
            imageACS =np.array([
                    sample[:,mid_idx, :, :],  # Red    -> height
                    sample[:,:, mid_idx, :],  # Green  -> width
                    sample[:,:, :, mid_idx],  # Blue   -> depth
                ]) # 3 planos com dos dados sismicos

            faciesACS = np.array([
                    sampleACS[mid_idx, :, :],  # Red    -> height
                    sampleACS[:, mid_idx, :],  # Green  -> width
                    sampleACS[:, :, mid_idx],  # Blue   -> depth
                ])                 
            
            
            np.savez(os.path.join(pathtrain,'shape_{}.npz'.format(j)),shape=imageACS,segs=faciesACS)
            j += 1            
    
            
        for z in range(0,z_range,image_size//2):

      

                voxel = seismic_cube[
                x:x+image_size,
                y:y+image_size,                    
                z:z+image_size,
                ]
                
                dep=prof[
                x:x+image_size,
                y:y+image_size,
                z:z+image_size
                ]

                voxel=np.stack([voxel,dep], axis=0)

                segs = facies_cube[
                x:x+image_size,
                y:y+image_size,                    
                z:z+image_size,
                ]            


                #np.savez(os.path.join(pathtrain_voxel,'voxel_{}.npz'.format(v)),voxel=voxel,segs=segs)                 
                #v+=1

    np.random.seed(seed)
    well_locations = sample_well_locations(n_wells, x_range, y_range)                


    
    #Dado de test1
    
    seismic_cube = np.load(os.path.join(out_folder,'data\\test_once', 'test_1tecva.npy'))#com tecva
    #seismic_cube = np.load(os.path.join(out_folder,'data\\test_once', 'test1_seismic.npy')) 
    facies_cube = np.load(os.path.join(out_folder,'data\\test_once', 'test1_labels.npy'))    

    
    proftest=[]
    for x in range(seismic_cube.shape[0]):
        for y in range(seismic_cube.shape[1]):
            for z in range(seismic_cube.shape[2]):
                proftest.append(z)
    
    proftest=np.array(proftest).reshape(seismic_cube.shape)
    
    
    pad= tuple((image_size - x%image_size) for x in seismic_cube.shape) # vê o resto da divisão pelo image_size para definir o padding
    print(seismic_cube.shape)
    print(pad)
    seismic_cube=np.pad(seismic_cube, ((0,pad[0]),(0,pad[1]),(0,pad[2])), 'edge') #faz o padding da sismica no final
    print(seismic_cube.shape)
    facies_cube=np.pad(facies_cube, ((0,pad[0]),(0,pad[1]),(0,pad[2])), 'edge') #faz o padding dos labels no final
    
    proftest=np.pad(proftest, ((0,pad[0]),(0,pad[1]),(0,pad[2])), 'edge')
    width, depth,height = seismic_cube.shape

    x_range = width - image_size +1
    y_range = depth - image_size +1
    z_range = height - image_size +1
    seismic_cube -= np.min(seismic_cube)
    seismic_cube /= np.max(seismic_cube)
    seismic_cube *= 255
    
    
    
    
    i, j,w,v = 0, 0,0,0    
    for x in range(0,x_range, image_size):
        
        for y in range(0,y_range,image_size):
            for z in range(0,z_range,image_size):
                
                
                voxel = seismic_cube[
                    x:x+image_size,
                    y:y+image_size,
                    z:z+image_size,
                    ]
                
                dep=proftest[
                x:x+image_size,
                y:y+image_size,
                z:z+image_size
                ]

                voxel=np.stack([voxel,dep], axis=0)                
                
                segs= facies_cube[
                    x:x+image_size,
                    y:y+image_size,
                    z:z+image_size,
                    ]
   
                    
                
    
                np.savez(os.path.join(patheval_voxel,'voxel_{}.npz'.format(w)),voxel=voxel,segs=segs)
                w +=1
    
    
    #Dado de test2
    
    #seismic_cube = np.load(os.path.join(out_folder,'data\\test_once', 'test2_seismic.npy'))
    seismic_cube = np.load(os.path.join(out_folder,'data\\test_once', 'test_2tecva.npy'))
    facies_cube = np.load(os.path.join(out_folder,'data\\test_once', 'test2_labels.npy'))    

    
    proftest=[]
    for x in range(seismic_cube.shape[0]):
        for y in range(seismic_cube.shape[1]):
            for z in range(seismic_cube.shape[2]):
                proftest.append(z)
    
    proftest=np.array(proftest).reshape(seismic_cube.shape)
    
    
    pad= tuple((image_size - x%image_size) for x in seismic_cube.shape) # vê o resto da divisão pelo image_size para definir o padding
    print(seismic_cube.shape)
    print(pad)
    seismic_cube=np.pad(seismic_cube, ((0,pad[0]),(0,pad[1]),(0,pad[2])), 'edge') #faz o padding da sismica no final
    print(seismic_cube.shape)
    facies_cube=np.pad(facies_cube, ((0,pad[0]),(0,pad[1]),(0,pad[2])), 'edge') #faz o padding dos labels no final
    
    proftest=np.pad(proftest, ((0,pad[0]),(0,pad[1]),(0,pad[2])), 'edge')
    width, depth,height = seismic_cube.shape

    x_range = width - image_size +1
    y_range = depth - image_size +1
    z_range = height - image_size +1
    seismic_cube -= np.min(seismic_cube)
    seismic_cube /= np.max(seismic_cube)
    seismic_cube *= 255
    
    
    
    
    i, j,w,v = 0, 0,0,0    
    for x in range(0,x_range, image_size):
        
        for y in range(0,y_range,image_size):
            for z in range(0,z_range,image_size):
                
                
                voxel = seismic_cube[
                    x:x+image_size,
                    y:y+image_size,
                    z:z+image_size,
                    ]
                
                dep=proftest[
                x:x+image_size,
                y:y+image_size,
                z:z+image_size
                ]

                voxel=np.stack([voxel,dep], axis=0)                
                
                segs= facies_cube[
                    x:x+image_size,
                    y:y+image_size,
                    z:z+image_size,
                    ]
   
                    
                
    
                np.savez(os.path.join(patheval_voxel2,'voxel_{}.npz'.format(w)),voxel=voxel,segs=segs)
                w +=1    
    
    
    
    print(f'Seed: {seed}')
    print(f'Input shape: {width, depth, height} (x, y, z)')
    print(f'#Wells sampled: {n_wells}')
    print(f'Sampled well locations:\n{well_locations}\n')


    print(f'Saving dataset as "{dataset_path}"\n...')

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
        n_iter=args.n_iter,
        image_size=args.image_size,
        out_folder=args.out_filename
    )
