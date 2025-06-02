import open3d as o3d
import numpy as np
from collections import defaultdict

def vis_pcd(point_cloud, name):
  vis = o3d.visualization.Visualizer()
  vis.create_window(window_name=name, width=800, height=600)
  vis.add_geometry(point_cloud)
  opt = vis.get_render_option()
  opt.point_size = 1.0 
  vis.run()
  vis.destroy_window()

def pointcloud_to_voxel_grid(point_cloud, voxel_size=0.02):
    """
    Convert point cloud to binary 3D voxel grid.
    
    Args:
        point_cloud: open3d.geometry.PointCloud
        voxel_size: Size of each voxel in meters
    
    Returns:
        voxel_grid: 3D binary numpy array
        offset: Origin offset for coordinate conversion
        grid_size: Dimensions of the voxel grid
    """
    points = np.asarray(point_cloud.points)
    
    # Calculate bounds and grid dimensions
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    
    # Add small padding to avoid edge issues
    padding = voxel_size * 2
    min_bounds -= padding
    max_bounds += padding
    
    grid_size = np.ceil((max_bounds - min_bounds) / voxel_size).astype(int)
    
    # Convert points to voxel indices
    voxel_indices = np.floor((points - min_bounds) / voxel_size).astype(int)
    
    # Create binary voxel grid
    voxel_grid = np.zeros(grid_size, dtype=bool)
    
    # Mark occupied voxels
    for idx in voxel_indices:
        if np.all(idx >= 0) and np.all(idx < grid_size):
            voxel_grid[tuple(idx)] = True
    
    return voxel_grid, min_bounds, grid_size

def voxel_grid_to_pointcloud(voxel_grid, offset, voxel_size=0.02):
    """
    Convert binary voxel grid back to point cloud.
    
    Args:
        voxel_grid: 3D binary numpy array
        offset: Origin offset from pointcloud_to_voxel_grid
        voxel_size: Size of each voxel in meters
    
    Returns:
        open3d.geometry.PointCloud
    """
    # Find occupied voxels
    occupied_voxels = np.argwhere(voxel_grid)
    
    # Convert voxel indices back to 3D coordinates (use voxel centers)
    points = (occupied_voxels + 0.5) * voxel_size + offset
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

def calculate_density_center(point_cloud, voxel_size=0.02):
    """
    Calculate the global center of density by finding the centroid of voxel centers weighted by point count.
    
    Args:
        point_cloud: open3d.geometry.PointCloud
        voxel_size: Size of each voxel in meters
    
    Returns:
        numpy array with 3D coordinates of density center
    """
    points = np.asarray(point_cloud.points)
    
    if len(points) == 0:
        return np.array([0, 0, 0])
    
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Count points per voxel and calculate voxel centers
    voxel_counts = defaultdict(int)
    for voxel_idx in voxel_indices:
        voxel_key = tuple(voxel_idx)
        voxel_counts[voxel_key] += 1
    
    # Calculate weighted centroid
    weighted_sum = np.zeros(3)
    total_weight = 0
    
    for voxel_key, count in voxel_counts.items():
        voxel_center = np.array(voxel_key) * voxel_size + voxel_size / 2
        weighted_sum += voxel_center * count
        total_weight += count
    
    return weighted_sum / total_weight if total_weight > 0 else np.array([0, 0, 0])

def create_sphere_marker(center, radius=0.05, color=[1, 0, 0]):
    """Create a sphere marker at the specified center position."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere
