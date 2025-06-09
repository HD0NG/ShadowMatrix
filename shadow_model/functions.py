import numpy as np
import numpy as np
from sklearn.neighbors import NearestNeighbors
# import laspy
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd
# from pvlib.location import Location

def compute_direction_vector(azimuth, altitude):
    x_dir = np.cos(np.radians(azimuth)) * np.cos(np.radians(altitude))
    y_dir = np.sin(np.radians(azimuth)) * np.cos(np.radians(altitude))
    z_dir = np.sin(np.radians(altitude))
    vec = np.array([x_dir, y_dir, z_dir])
    return vec / np.linalg.norm(vec)

def find_z_at_target(points, target_coords):
    distances = np.linalg.norm(points[:, :2] - target_coords[:2], axis=1)
    closest_point_index = np.argmin(distances)
    return points[closest_point_index, 2]

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
def voxelize(points, classifications, voxel_size):
    min_bounds = points.min(axis=0)
    voxel_indices = np.floor((points - min_bounds) / voxel_size).astype(int)
    voxel_dict = {}
    for idx, cls in zip(voxel_indices, classifications):
        key = tuple(idx)
        voxel_dict.setdefault(key, []).append(cls)
    voxel_grid = {}
    for voxel_idx, cls_list in voxel_dict.items():
        classes, counts = np.unique(cls_list, return_counts=True)
        # print(classes, counts)
        majority_class = classes[np.argmax(counts)]
        voxel_grid[voxel_idx] = {
            'class': majority_class,
            'density': np.max(counts)  # For density weighting
        }
    return voxel_grid, min_bounds

# --- Beer-Lambert Ray Marching with Tunable Parameters ---
def beer_lambert_ray_march(voxel_grid, min_bounds, voxel_size, ray_origin, direction_vector, max_distance, params, step=0.5):
    transmission = 1.0
    position = np.array(ray_origin)

    for distance in np.arange(0, max_distance, step):
        position = ray_origin + direction_vector * distance
        voxel_idx = tuple(np.floor((position - min_bounds) / voxel_size).astype(int))
        
        if voxel_idx in voxel_grid:
            extinction = get_voxel_extinction(voxel_grid[voxel_idx], params)
            if extinction == np.inf:  # Hit building
                transmission = 0
                break
            else:
                transmission *= np.exp(-extinction * step)
    
    return transmission

def create_transmission_matrix(altitude_range, azimuth_range, voxel_grid, min_bounds, voxel_size, target_coords, target_z, radius, params):
    matrix = np.zeros((len(altitude_range), len(azimuth_range)))

    for i, altitude in enumerate(altitude_range):
        for j, azimuth in enumerate(azimuth_range):
            dir_vec = compute_direction_vector(azimuth, altitude)
            ray_origin = np.array([target_coords[0], target_coords[1], target_z])
            trans = beer_lambert_ray_march(voxel_grid, min_bounds, voxel_size, ray_origin, dir_vec, radius, params)
            matrix[i, j] = trans
    
    return matrix

def create_shadow_matrix(altitude_range, azimuth_range, voxel_grid, min_bounds, voxel_size, target_coords, target_z, radius, params):
    shadow_matrix = np.zeros((len(altitude_range), len(azimuth_range)))

    for i, altitude in enumerate(altitude_range):
        for j, azimuth in enumerate(azimuth_range):
            dir_vec = compute_direction_vector(azimuth, altitude)
            ray_origin = np.array([target_coords[0], target_coords[1], target_z])
            transmission = beer_lambert_ray_march(voxel_grid, min_bounds, voxel_size, ray_origin, dir_vec, radius, params)
            shadow_matrix[i, j] = 1.0 - transmission**params.shadow_gamma  # Shadow = 1 - Transmission

    return shadow_matrix

def create_dataframe(matrix, altitude_range, azimuth_range):
    altitudes = [f"Altitude_{int(alt)}" for alt in altitude_range]
    azimuths = [f"Azimuth_{int(azi)}" for azi in azimuth_range]
    return pd.DataFrame(matrix, index=altitudes, columns=azimuths)

def plot_shadow_polar(matrix, altitude_range, azimuth_range):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    for i, altitude in enumerate(altitude_range):
        for j, azimuth in enumerate(azimuth_range):
            shadow_intensity = matrix[i, j]
            if shadow_intensity > 0:
                color = plt.cm.gray_r(shadow_intensity)
                ax.plot(np.radians(azimuth), 90 - altitude, 'o', color=color, markersize=5)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_title('Shadow Intensity Polar Plot\n(Beer–Lambert Voxel Model)\n\n')

    # Add colorbar indicating shadow intensity
    sm = plt.cm.ScalarMappable(cmap='gray_r', norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label='Shadow Intensity (1 - Transmission)')
    
    plt.show()

def plot_shadow_polar_in(matrix, altitude_range, azimuth_range):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    for i, altitude in enumerate(altitude_range):
        for j, azimuth in enumerate(azimuth_range):
            shadow_intensity = matrix[i, j]
            if shadow_intensity > 0:
                color = plt.cm.gray_r(shadow_intensity)
                ax.plot(np.radians(azimuth), 90 - altitude, 'o', color=color, markersize=5)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_title('Shadow Intensity Polar Plot\n(Beer–Lambert Voxel Model)\n\n')

    # Add colorbar indicating shadow intensity
    sm = plt.cm.ScalarMappable(cmap='gray_r', norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label='Shadow Intensity (1 - Transmission)')
    
    return fig
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

# --- Extinction Coefficient Function ---
def get_voxel_extinction(voxel, params):
    if voxel['class'] == 6:
        return np.inf
    elif voxel['class'] == 'building_buffer':
        return params.buffer_extinction
    elif voxel['class'] == 3:  # Vegetation classes
        # Density weighting (normalize if needed)
        density = voxel.get('density', 1)
        return params.base_k3 * params.vegetation_weight * (1 + params.density_weight * density)
    elif voxel['class'] == 4:  # Building classes
        # Density weighting (normalize if needed)
        density = voxel.get('density', 1)
        return params.base_k4 * params.vegetation_weight * (1 + params.density_weight * density)
    elif voxel['class'] == 5:  # Building classes
        # Density weighting (normalize if needed)
        density = voxel.get('density', 1)
        return params.base_k5 * params.vegetation_weight * (1 + params.density_weight * density)
        # return 0.6
    else:
        return 0.0