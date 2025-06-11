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
# def get_voxel_extinction(voxel, params):
#     if voxel['class'] == 6:
#         return np.inf
#     elif voxel['class'] == 'building_buffer':
#         return params.buffer_extinction
#     elif voxel['class'] == 3:  # Vegetation classes
#         # Density weighting (normalize if needed)
#         density = voxel.get('density', 1)
#         return params.base_k3 * params.vegetation_weight * (1 + params.density_weight * density)
#     elif voxel['class'] == 4:  # Building classes
#         # Density weighting (normalize if needed)
#         density = voxel.get('density', 1)
#         return params.base_k4 * params.vegetation_weight * (1 + params.density_weight * density)
#     elif voxel['class'] == 5:  # Building classes
#         # Density weighting (normalize if needed)
#         density = voxel.get('density', 1)
#         return params.base_k5 * params.vegetation_weight * (1 + params.density_weight * density)
#         # return 0.6
#     else:
#         return 0.0

def get_voxel_extinction(voxel, params):
    class_weights = voxel.get('classes', {})
    if not class_weights:
        return 0.0

    extinction = 0.0
    for cls, weight in class_weights.items():
        if cls == 6:  # Building
            k = params.building_k
        elif cls == 3:
            k = params.base_k3
        elif cls == 4:
            k = params.base_k4
        elif cls == 5:
            k = params.base_k5
        else:
            k = 0.0
        extinction += weight * k

    # Apply global density scaling (optional)
    # if params.density_weight > 0:
    #     extinction *= (1 + params.density_weight * voxel.get('density', 1))

    return extinction

def voxel_weighting(pos, voxel_idx, voxel_size, min_bounds):
    center = min_bounds + (np.array(voxel_idx) + 0.5) * voxel_size
    dist = np.linalg.norm(pos - center)
    return np.clip(1 - dist / (np.sqrt(3) * voxel_size / 2), 0, 1)  # normalized

# --- Beer-Lambert Ray Marching with Tunable Parameters ---
def beer_lambert_ray_march(voxel_grid, min_bounds, voxel_size, ray_origin, direction_vector, max_distance, params, step=0.5):
    transmission = 1.0
    position = np.array(ray_origin)

    for distance in np.arange(0, max_distance, step):
        position = ray_origin + direction_vector * distance
        voxel_idx = tuple(np.floor((position - min_bounds) / voxel_size).astype(int))
        
        if voxel_idx in voxel_grid:
            extinction = get_voxel_extinction(voxel_grid[voxel_idx], params)
            # 🔥 Apply distance-based weight
            weight = voxel_weighting(position, voxel_idx, voxel_size, min_bounds)
            k_weighted = extinction * weight
            if extinction == np.inf:  # Hit building
                transmission = 0
                break
            else:
                transmission *= np.exp(-extinction * step)
    
    return transmission

def jittered_directions(base_dir, num_samples=5, spread_deg=1.0):
    directions = []
    for _ in range(num_samples):
        jitter = np.random.normal(0, np.radians(spread_deg), size=3)
        jittered_dir = base_dir + jitter
        jittered_dir /= np.linalg.norm(jittered_dir)
        directions.append(jittered_dir)
    return directions

def create_transmission_matrix(altitude_range, azimuth_range, voxel_grid, min_bounds, voxel_size, target_coords, target_z, radius, params):
    matrix = np.zeros((len(altitude_range), len(azimuth_range)))

    for i, altitude in enumerate(altitude_range):
        for j, azimuth in enumerate(azimuth_range):
            dir_vec = compute_direction_vector(azimuth, altitude)
            ray_origin = np.array([target_coords[0], target_coords[1], target_z])
            directions = jittered_directions(dir_vec, num_samples=7)
            transmissions = [beer_lambert_ray_march(voxel_grid, min_bounds, voxel_size,ray_origin, d, radius, params)for d in directions]
            # trans = beer_lambert_ray_march(voxel_grid, min_bounds, voxel_size, ray_origin, dir_vec, radius, params)
            matrix[i, j] = np.mean(transmissions)
    
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