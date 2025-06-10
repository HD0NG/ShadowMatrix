import numpy as np


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