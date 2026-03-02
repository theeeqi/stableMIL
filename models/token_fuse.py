import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import h5py

def create_bounding_box(coords):
    min_x, min_y = torch.min(coords, dim=0).values
    max_x, max_y = torch.max(coords, dim=0).values
    return min_x, max_x, min_y, max_y

def assign_labels_to_regions(coords, min_x, max_x, min_y, max_y, m, n):

    region_width = (max_x - min_x) / m
    region_height = (max_y - min_y) / n
    x_idx = ((coords[:, 0] - min_x) // region_width).clamp(0, m - 1).long()
    y_idx = ((coords[:, 1] - min_y) // region_height).clamp(0, n - 1).long()


    labels = y_idx * m + x_idx
    return labels

def fuse_labels(coords,ratio = 2):
    min_x, max_x, min_y, max_y = create_bounding_box(coords)
    m = torch.ceil((max_x - min_x)/ratio).int() 
    n = torch.ceil((max_y-min_y)/ratio).int()  
    labels = assign_labels_to_regions(coords, min_x, max_x, min_y, max_y, m, n)
    return labels

