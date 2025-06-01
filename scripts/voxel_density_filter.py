import open3d as o3d
import numpy as np
import argparse
from collections import defaultdict

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
    
    # calc voxel density
    voxel_counts = defaultdict(int)
    point_to_voxel = {}
    
    for i, voxel_idx in enumerate(voxel_indices):
        voxel_key = tuple(voxel_idx)
        voxel_counts[voxel_key] += 1
        point_to_voxel[i] = voxel_key
    
    # filter based on thresh 
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
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Count points per voxel
    voxel_counts = defaultdict(int)
    for voxel_idx in voxel_indices:
        voxel_key = tuple(voxel_idx)
        voxel_counts[voxel_key] += 1
    
    # calc centroid
    weighted_sum = np.zeros(3)
    total_weight = 0
    
    for voxel_key, count in voxel_counts.items():
        voxel_center = np.array(voxel_key) * voxel_size + voxel_size / 2
        weighted_sum += voxel_center * count
        total_weight += count
    
    return weighted_sum / total_weight

def create_sphere_marker(center, radius=0.05, color=[1, 0, 0]):
    """Create a sphere marker at the specified center position."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere

def main():
    parser = argparse.ArgumentParser(description='Filter grapevine point cloud using voxel density')
    parser.add_argument('ply_file', help='Path to input .ply file')
    parser.add_argument('--voxel_size', type=float, default=0.02, help='Voxel size in meters (default: 0.02)')
    parser.add_argument('--max_threshold', type=int, default=300, help='Maximum threshold to test (default: 50)')
    parser.add_argument('--step', type=int, default=30, help='Threshold increment step (default: 5)')
    
    args = parser.parse_args()
    
    print(f"Loading point cloud from {args.ply_file}")
    original_pcd = o3d.io.read_point_cloud(args.ply_file)
    
    if len(original_pcd.points) == 0:
        print("Error: Could not load point cloud or file is empty")
        return
    
    print(f"Original point cloud has {len(original_pcd.points)} points")
    
    density_center = calculate_density_center(original_pcd, args.voxel_size)
    print(f"Global density center: [{density_center[0]:.3f}, {density_center[1]:.3f}, {density_center[2]:.3f}]")
    
    print("\nShowing original point cloud with density center marker...")
    density_marker = create_sphere_marker(density_center, radius=0.02, color=[1, 0, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Original Point Cloud", width=800, height=600)
    vis.add_geometry(original_pcd)
    vis.add_geometry(density_marker)
    vis.run()
    vis.destroy_window()
    
    thresholds = range(args.step, args.max_threshold + 1, args.step)
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold} points per voxel")
        
        filtered_pcd = voxel_density_filter(original_pcd, args.voxel_size, threshold)
        
        remaining_points = len(filtered_pcd.points)
        retention_rate = (remaining_points / len(original_pcd.points)) * 100
        
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

if __name__ == "__main__":
    main()