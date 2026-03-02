
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import h5py

def minimum_bounding_rectangle(coords):

    hull = ConvexHull(coords)
    hull_coords = coords[hull.vertices]
    min_area = float('inf')
    best_rect = None
    best_angle = 0

    for i in range(len(hull_coords)):
        p1 = hull_coords[i]
        p2 = hull_coords[(i + 1) % len(hull_coords)]
        edge_dir = p2 - p1
        edge_dir = edge_dir / np.linalg.norm(edge_dir)
        angle = np.arctan2(edge_dir[1], edge_dir[0])
        rotation_matrix = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        rotated_coords = np.dot(coords - p1, rotation_matrix)
        min_x, min_y = np.min(rotated_coords, axis=0)
        max_x, max_y = np.max(rotated_coords, axis=0)
        area = (max_x - min_x) * (max_y - min_y)

        if area < min_area:
            min_area = area
            rect_coords = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ])
            best_rect = np.dot(rect_coords, rotation_matrix.T) + p1
            best_angle = angle

    return best_rect, best_angle

def distance_to_line_vectorized(coords, line_start, line_end):

    line_vec = line_end - line_start
    point_vecs = coords - line_start
    cross = np.cross(line_vec, point_vecs)
    return np.abs(cross) / np.linalg.norm(line_vec)

def assign_coords_to_regions_vectorized(coords, mbr, m, n):

    bottom_left = mbr[0]  
    bottom_right = mbr[1]  
    top_left = mbr[3] 

 
    dist_to_bottom = distance_to_line_vectorized(coords, bottom_left, bottom_right)
    dist_to_left = distance_to_line_vectorized(coords, bottom_left, top_left)


    max_dist_bottom = np.max(dist_to_bottom)
    max_dist_left = np.max(dist_to_left)
    region_width = max_dist_bottom / n
    region_height = max_dist_left / m

    x_indices = np.clip((dist_to_bottom / region_width).astype(int), 0, m - 1)
    y_indices = np.clip((dist_to_left / region_height).astype(int), 0, n - 1)

    return np.column_stack((x_indices, y_indices))


def divide_mbr_by_aspect_ratio(mbr, total_regions=128):

    total_regions = max(1,total_regions)
    length = np.linalg.norm(mbr[1] - mbr[0])  
    width = np.linalg.norm(mbr[3] - mbr[0])  


    aspect_ratio = length / width


    n = int(np.sqrt(total_regions * aspect_ratio))
    n = max(1,n)
    m = int(total_regions / n)


    return m, n

def find_region(coords,region_num):
    mbr, angle = minimum_bounding_rectangle(coords.cpu().numpy())
    m = int(np.ceil(np.sqrt(region_num)))
    n = m
    region_indices = assign_coords_to_regions_vectorized(coords.cpu().numpy(), mbr, m, n)
    regindex = region_indices[:, 0] * n+ region_indices[:, 1]

    return regindex

def find_region_ag(coords,aggregate_num):
    mbr, angle = minimum_bounding_rectangle(coords.cpu().numpy())

    length = np.linalg.norm(mbr[1] - mbr[0]) 
    width = np.linalg.norm(mbr[3] - mbr[0]) 


    m = int(np.ceil(length/np.sqrt(aggregate_num)))
    n = int(np.ceil(width/np.sqrt(aggregate_num)))
    region_indices = assign_coords_to_regions_vectorized(coords.cpu().numpy(), mbr, m, n)
    regindex = region_indices[:, 0] * n+ region_indices[:, 1]

    return regindex
