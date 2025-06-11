import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, Counter

# --- Model Parameters Class ---
class ModelParams:
    def __init__(self,
                 base_k3=0.4,
                 base_k4=0.4,
                 base_k5=0.4,
                 building_k=8.0,
                 vegetation_weight=1.2,
                 density_weight=0.8,
                 shadow_gamma=1.5):
        self.base_k3 = base_k3
        self.base_k4 = base_k4
        self.base_k5 = base_k5
        self.building_k = building_k
        self.vegetation_weight = vegetation_weight
        self.density_weight = density_weight
        self.shadow_gamma = shadow_gamma
        # self.buffer_radius = buffer_radius
        # self.buffer_extinction = buffer_extinction

# --- Utility Functions ---
def estimate_point_spacing(points, k=2, sample_size=10000):
    sample = points[np.random.choice(points.shape[0], size=min(sample_size, len(points)), replace=False)]
    nbrs = NearestNeighbors(n_neighbors=k).fit(sample)
    distances, _ = nbrs.kneighbors(sample)
    avg_spacing = np.mean(distances[:, 1])
    return avg_spacing

def determine_voxel_size(points, multiplier=10):
    resolution = estimate_point_spacing(points)
    voxel_size = np.ceil(multiplier * resolution)
    # print(f"Estimated spacing: {resolution:.3f} → Voxel size: {voxel_size}")
    return voxel_size

# --- Voxelization ---
# def voxelize(points, classifications, voxel_size):
#     min_bounds = points.min(axis=0)
#     voxel_indices = np.floor((points - min_bounds) / voxel_size).astype(int)
#     voxel_dict = {}
#     for idx, cls in zip(voxel_indices, classifications):
#         key = tuple(idx)
#         voxel_dict.setdefault(key, []).append(cls)
#     voxel_grid = {}
#     for voxel_idx, cls_list in voxel_dict.items():
#         classes, counts = np.unique(cls_list, return_counts=True)
#         # print(classes, counts)
#         majority_class = classes[np.argmax(counts)]
#         voxel_grid[voxel_idx] = {
#             'class': majority_class,
#             'density': np.max(counts)  # For density weighting
#         }
#     return voxel_grid, min_bounds
def compute_voxel_class_weights(class_list):
    total = len(class_list)
    counter = Counter(class_list)
    return {int(cls): count / total for cls, count in counter.items()}

def voxelize(points, classifications, voxel_size):
    min_bounds = points.min(axis=0)
    voxel_indices = np.floor((points - min_bounds) / voxel_size).astype(int)

    voxel_dict = {}
    for idx, cls in zip(voxel_indices, classifications):
        key = tuple(idx)
        voxel_dict.setdefault(key, []).append(cls)

    voxel_grid = {}
    for voxel_idx, cls_list in voxel_dict.items():
        class_weights = compute_voxel_class_weights(cls_list)
        voxel_grid[voxel_idx] = {
            'classes': class_weights,
            'density': len(cls_list)
        }

    return voxel_grid, min_bounds

def fill_building_voxel_columns(voxel_grid):
    filled = set()
    for (i, j, k), class_weights in voxel_grid.items():
        if class_weights.get(6, 0) > 0.5:  # Mostly building
            for kk in range(0, k):  # Everything below
                idx = (i, j, kk)
                if idx in voxel_grid:
                    voxel_grid[idx][6] = max(voxel_grid[idx].get(6, 0), 0.9)
                else:
                    voxel_grid[idx] = {6: 1.0}
                filled.add(idx)
    return voxel_grid



def expand_target_area(voxel_grid, min_bounds, voxel_size, target_coords, target_z, target_voxel_radius=2):
    """
    Clears (empties) voxels around the target point to prevent immediate self-blocking.
    
    Args:
        voxel_grid (dict): Existing voxel grid.
        min_bounds (np.array): Minimum XYZ bounds of voxel grid.
        voxel_size (float): Size of each voxel.
        target_coords (np.array): XY coordinates of target.
        target_z (float): Z elevation of target.
        target_voxel_radius (int): Radius (in voxels) around target to clear.
        
    Returns:
        Updated voxel_grid with cleared target voxels.
    """
    target_idx = np.floor((np.array([target_coords[0], target_coords[1], target_z]) - min_bounds) / voxel_size).astype(int)

    for dx in range(-target_voxel_radius, target_voxel_radius + 1):
        for dy in range(-target_voxel_radius, target_voxel_radius + 1):
            for dz in range(-target_voxel_radius, target_voxel_radius + 1):
                voxel_key = tuple(target_idx + np.array([dx, dy, dz]))
                if voxel_key in voxel_grid:
                    del voxel_grid[voxel_key]  # Remove obstacles around the target

    return voxel_grid

# def build_voxel_grid(points, classifications, voxel_size):
#     voxel_grid = {}
#     min_bounds = points.min(axis=0)

#     for pt, cls in zip(points, classifications):
#         voxel_idx = tuple(np.floor((pt - min_bounds) / voxel_size).astype(int))
#         if voxel_idx not in voxel_grid:
#             voxel_grid[voxel_idx] = {'points': [], 'classes': Counter()}
#         voxel_grid[voxel_idx]['points'].append(pt)
#         voxel_grid[voxel_idx]['classes'][cls] += 1

#     return voxel_grid, min_bounds
