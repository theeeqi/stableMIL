import time 
import sys 
import h5py
from find_Region import find_region_ag
from token_fuse import fuse_labels
import os 
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
import pandas as pd
def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def fuse_token(coords,ratio = 2):

    coords = torch.from_numpy(coords)
    
    labels = fuse_labels(coords,ratio) 
    sorted_idx = torch.argsort(labels)
    coords = coords[sorted_idx]
    
    bin_count = torch.bincount(labels)

    bin_count = bin_count[bin_count>0]

    coords = coords.float()
    split_coords = torch.split(coords, bin_count.tolist())

    max_points = int(ratio**2)
    padded_regions = []

    for i, region in enumerate(split_coords):
        num_points = region.size(0)
        if num_points >= max_points:

            padded_region = region[:max_points]
        else:

            pad_value = region.mean(dim=0).expand(max_points-num_points,-1)

            padded_region = torch.concat([region,pad_value],dim=0)

        padded_regions.append(padded_region)


    padded_regions = torch.stack(padded_regions)

    coords = padded_regions.mean(dim=1)/ratio

    return labels.numpy(),sorted_idx.numpy(),coords.numpy()

    
def region_sort(coords,aggregate_num=256):
    N,C = coords.shape 
    if N < 5 :
        sorted_index = np.arange(N)
    else:
        region_indices = find_region_ag(torch.from_numpy(coords),aggregate_num)
    sorted_index = np.argsort(region_indices)
    return region_indices,sorted_index

def create_tokens(coords,k_neighbors = 8,max_dist = 6*np.sqrt(2)):

    N, C = coords.shape

    k_neighbors = min(k_neighbors,N - 1)


    coords_tree = coords
    nbrs = NearestNeighbors(n_neighbors= k_neighbors + 1, algorithm='kd_tree').fit(coords_tree)
    distance, indices = nbrs.kneighbors(coords_tree) 

    self_indices = np.arange(N)[:, np.newaxis] 
    neighbor_index_default = np.repeat(self_indices, k_neighbors + 1, axis=1)
        
    mask = distance < max_dist

    neighbor_index = np.where(mask, indices, neighbor_index_default)
    

    neighbor_out = np.where(mask, indices, -1)

    return neighbor_index,neighbor_out
    
save_path = r''
h5_path = r''
df = pd.read_csv(r'')

slide_list = df['slide_id'].tolist()
os.makedirs(save_path,exist_ok=True)

start_time = time.time()
counts = 0
N_ls =[]
N_ls_1 = []
for f in os.listdir(h5_path):
    if f.split('.')[0] not in slide_list:
        continue
    counts +=1
    print(f'processing f{f}',flush=True)
    
    full_path = os.path.join(h5_path,f)
    with h5py.File(full_path,'r') as hdf5_file:
        coords = hdf5_file['coords'][:] //512

    fuse_lb,fuse_sorted_idx,fuse_coords = fuse_token(coords,2)
    region_indices , region_sorted_index = region_sort(fuse_coords, 256)
    fuse_coords_reindex = fuse_coords[region_sorted_index,:]

    attention_mask_1 ,attention_mask_2 = create_tokens(fuse_coords_reindex, 8)
    asset_dict = {'fuse_labels': fuse_lb, 'fuse_coords': fuse_coords,'fuse_sorted_idx': fuse_sorted_idx,'region_indices': region_indices,
                  'region_sorted_index': region_sorted_index,'attention_mask_1': attention_mask_1,'attention_mask_2': attention_mask_2}
    
    N_ls.append(coords.shape[0])
    N_ls_1.append(fuse_coords_reindex.shape[0])
    save_hdf5(save_path +'/'+ f , asset_dict)
end_time = time.time()

print(np.median(N_ls))
print(np.median(N_ls_1))
print('time',end_time - start_time)
print(f"average time: {(end_time - start_time) / counts}",flush=True)