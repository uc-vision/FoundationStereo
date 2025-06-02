import open3d as o3d
import numpy as np
import argparse
from collections import defaultdict
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from pcd_filter_utils import (
    pointcloud_to_voxel_grid,
    voxel_grid_to_pointcloud,
    calculate_density_center,
    create_sphere_marker,
    vis_pcd
)

def voxel_density_filter(point_cloud, voxel_size=0.02, min_points_threshold=10):
    """
    Filter point cloud based on voxel density.
    
    Args:
        point_cloud: open3d.geometry.PointCloud
        voxel_size: Size of each voxel in meters (default 2cm)
        min_points_threshold: Minimum points per voxel to keep
    
    Returns:
        Filtered open3d.geometry.PointCloud
    """
    points = np.asarray(point_cloud.points)
    
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Count points per voxel
    voxel_counts = defaultdict(int)
    point_to_voxel = {}
    
    for i, voxel_idx in enumerate(voxel_indices):
        voxel_key = tuple(voxel_idx)
        voxel_counts[voxel_key] += 1
        point_to_voxel[i] = voxel_key
    
    # Filter based on voxel density
    keep_indices = []
    for i, point in enumerate(points):
        voxel_key = point_to_voxel[i]
        if voxel_counts[voxel_key] >= min_points_threshold:
            keep_indices.append(i)
    
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[keep_indices])
    
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[keep_indices])
    
    return filtered_pcd

def create_3d_sphere_kernel(radius_voxels):
    """
    Create a 3D spherical structuring element for morphological operations.
    
    Args:
        radius_voxels: Radius in voxel units
    
    Returns:
        3D binary numpy array representing sphere
    """
    size = 2 * radius_voxels + 1
    center = radius_voxels
    
    # Create coordinate grids
    x, y, z = np.ogrid[:size, :size, :size]
    
    # Create sphere mask
    sphere = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius_voxels**2
    
    return sphere

def test_incremental_density_filter(point_cloud, args):
  thresholds = range(args.step, args.max_threshold + 1, args.step)
  
  for threshold in thresholds:
    print(f"\nTesting threshold: {threshold} points per voxel")
    
    filtered_pcd = voxel_density_filter(point_cloud, args.voxel_size, threshold)
    
    remaining_points = len(filtered_pcd.points)
    retention_rate = (remaining_points / len(point_cloud.points)) * 100
    
    print(f"Filtered point cloud has {remaining_points} points ({retention_rate:.1f}% retained)")
    
    if remaining_points == 0:
      print("No points remaining - threshold too high!")
      continue
    
    filtered_density_center = calculate_density_center(filtered_pcd, args.voxel_size)
    filtered_marker = create_sphere_marker(filtered_density_center, radius=0.02, color=[0, 1, 0])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Filtered (threshold={threshold})", width=800, height=600)
    vis.add_geometry(filtered_pcd)
    vis.add_geometry(filtered_marker)
    vis.run()
    vis.destroy_window()
    
    print(f"Filtered density center: [{filtered_density_center[0]:.3f}, {filtered_density_center[1]:.3f}, {filtered_density_center[2]:.3f}]")
    
    response = input("Continue to next threshold? (y/n/save): ").lower().strip()
    if response == 'n':
      break
    elif response == 'save':
      output_file = args.ply_file.replace('.ply', f'_filtered_t{threshold}_v{args.voxel_size}.ply')
      o3d.io.write_point_cloud(output_file, filtered_pcd)
      print(f"Saved filtered point cloud to {output_file}")
      break

def dbscan_outlier_removal(point_cloud, eps=0.01, min_samples=200):
    """
    Use DBSCAN clustering to identify and remove outliers.
    
    Args:
        point_cloud: open3d.geometry.PointCloud
        eps: Maximum distance between two samples for clustering
        min_samples: Minimum number of samples in a cluster
    
    Returns:
        Filtered point cloud (largest cluster) and outlier indices
    """
    points = np.asarray(point_cloud.points)
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    # Find largest cluster 
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    
    if len(unique_labels) > 0:
        largest_cluster_label = unique_labels[np.argmax(counts)]
        inlier_mask = labels == largest_cluster_label
    else:
        # If no clusters found, keep all points
        inlier_mask = np.ones(len(points), dtype=bool)
    
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points[inlier_mask])
    
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        result.colors = o3d.utility.Vector3dVector(colors[inlier_mask])
    
    outlier_indices = np.where(~inlier_mask)[0]
    
    return result, outlier_indices

def morphological_closing_density_preserving(
    point_cloud, 
    radius_meters=0.005, 
    voxel_size=0.003,
    density_preservation_factor=1.5,
    max_points_per_voxel=3
):
    """
    Apply morphological closing while preserving point density.
    Revised to avoid blockiness and uncontrolled point growth.

    Args:
        point_cloud: open3d.geometry.PointCloud
        radius_meters: Radius of structuring element in meters (default: 0.02)
        voxel_size: Voxel size for morphological operations (default: 0.01)
        density_preservation_factor: Controls how aggressively gaps are filled (default: 1.5)
        max_points_per_voxel: Maximum synthetic points added per voxel (default: 2)

    Returns:
        Filtered open3d.geometry.PointCloud with natural density
    """
    def poisson_sample_voxel(voxel_center, voxel_size, tree, min_spacing, max_attempts=10):
        """Generate points with Poisson disk sampling in a voxel."""
        for _ in range(max_attempts):
            offset = np.random.uniform(-voxel_size/3, voxel_size/3, 3)
            candidate = voxel_center + offset
            if len(tree.query_ball_point(candidate, min_spacing)) == 0:
                return candidate
        return None  # failed to place point

    # store original data so we dont lose resolution from voxelisation
    original_points = np.asarray(point_cloud.points)
    has_colors = point_cloud.has_colors()
    if has_colors:
        original_colors = np.asarray(point_cloud.colors)

    # 3D closing on pcd using 3D kernel 
    voxel_grid, offset, grid_size = pointcloud_to_voxel_grid(point_cloud, voxel_size)
    radius_voxels = max(1, int(np.ceil(radius_meters / voxel_size)))
    kernel = create_3d_sphere_kernel(radius_voxels)
    
    # smooth to reduce blocking from voxels
    smoothed = ndimage.gaussian_filter(voxel_grid.astype(float), sigma=0.7)
    closed_voxels = ndimage.binary_closing(smoothed > 0.5, structure=kernel)

    # convert back to points
    closed_points = voxel_grid_to_pointcloud(closed_voxels, offset, voxel_size)
    closed_coords = np.asarray(closed_points.points)
    
    if len(closed_coords) == 0:
        return point_cloud  

    # preserve points based on the closed_coords
    closed_tree = cKDTree(closed_coords)
    distances, _ = closed_tree.query(original_points)
    keep_mask = distances <= (voxel_size * np.sqrt(3) / 2)  # Half voxel diagonal
    kept_points = original_points[keep_mask]
    if has_colors:
        kept_colors = original_colors[keep_mask]

    new_points = []
    new_colors = []
    
    # Estimate original density
    orig_tree = cKDTree(original_points)
    sample_size = min(1000, len(original_points))
    sample_indices = np.random.choice(len(original_points), sample_size, replace=False)
    sample_dists, _ = orig_tree.query(original_points[sample_indices], k=2)
    avg_spacing = np.mean(sample_dists[:, 1])  # Avg distance to nearest neighbor

    # Add points only where very undersampled
    for voxel_center in closed_coords:
        dist_to_nearest, _ = orig_tree.query(voxel_center)
        
        if dist_to_nearest > avg_spacing * density_preservation_factor:
            for _ in range(max_points_per_voxel):
                new_point = poisson_sample_voxel(
                    voxel_center, 
                    voxel_size, 
                    orig_tree, 
                    avg_spacing * 0.8  # Slightly tighter spacing
                )
                if new_point is not None:
                    new_points.append(new_point)
                    if has_colors:
                        _, nearest_idx = orig_tree.query(new_point)
                        new_colors.append(original_colors[nearest_idx])

    # combine
    if len(new_points) > 0:
        all_points = np.vstack([kept_points, np.array(new_points)])
        if has_colors:
            all_colors = np.vstack([kept_colors, np.array(new_colors)])
    else:
        all_points = kept_points
        if has_colors:
            all_colors = kept_colors

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(all_points)
    if has_colors and len(all_colors) > 0:
        result.colors = o3d.utility.Vector3dVector(all_colors)

    return result

def main():
    parser = argparse.ArgumentParser(description='Filter grapevine point cloud using voxel density')
    parser.add_argument('ply_file', help='Path to input .ply file')
    parser.add_argument('--voxel_size', type=float, default=0.02, help='Voxel size in meters (default: 0.02)')
    parser.add_argument('--max_threshold', type=int, default=300, help='Maximum threshold to test (default: 300)')
    parser.add_argument('--step', type=int, default=30, help='Threshold increment step (default: 30)')
    parser.add_argument('--method', type=str, default='all', help='select from "opening", "closing", "density" or "all". (default: "all")')
    
    args = parser.parse_args()
    
    print(f"Loading point cloud from {args.ply_file}")
    original_pcd = o3d.io.read_point_cloud(args.ply_file)
    
    if len(original_pcd.points) == 0:
        print("Error: Could not load point cloud or file is empty")
        return
    
    print(f"Original point cloud has {len(original_pcd.points)} points")
    vis_pcd(original_pcd, name='original')

    def calc_remaining_points(new, old):
      remaining_points = len(new.points)
      retention_rate = (remaining_points / len(old.points)) * 100
      print(f"new point cloud has {remaining_points} points ({retention_rate:.1f}% retained)")
        
    closed_pcd = morphological_closing_density_preserving(original_pcd)
    calc_remaining_points(closed_pcd, original_pcd)
    vis_pcd(closed_pcd, name="closed pcd")

    descan_pcd, mask = dbscan_outlier_removal(closed_pcd)
    calc_remaining_points(descan_pcd, closed_pcd)
    vis_pcd(descan_pcd, name='descan_pcd')

    filtered_pcd = voxel_density_filter(descan_pcd, args.voxel_size, 170)
    calc_remaining_points(filtered_pcd, descan_pcd)
    vis_pcd(filtered_pcd, name='voxel density filter')

if __name__ == "__main__":
    main()