import numpy as np
import open3d as o3d
import cv2
from typing import List, Tuple, Optional
import pandas as pd

import time


from geometry_msgs.msg import TransformStamped, PoseStamped

import os
import re

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_timestamps_from_pngs(folder_path: str) -> List[str]:
    pattern = re.compile(r'^(\d{8}_\d{6})_depth\.png$')
    timestamps = []

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            timestamps.append(match.group(1))

    return timestamps

def pose_stamped_to_matrix(pose_stamped: PoseStamped) -> np.ndarray:
    p = pose_stamped.pose.position
    q = pose_stamped.pose.orientation

    x, y, z, w = q.x, q.y, q.z, q.w

    # Normalize the quaternion (in case it's not already normalized)
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    
    # Convert to rotation matrix using the standard formula
    matrix = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y), p.x],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x), p.y],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y), p.z],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return matrix

def depth_to_pointcloud(depth_image: np.ndarray, 
                       intrinsic_matrix: np.ndarray,
                       pose_matrix: np.ndarray = None,
                       color_image: np.ndarray = None,
                       depth_scale: float = 1.0,
                       max_depth: float = 10000.0) -> o3d.geometry.PointCloud:
    """
    Convert a depth image to a point cloud in world coordinates.
    
    Args:
        depth_image: Depth image (H x W) in millimeters or meters
        intrinsic_matrix: Camera intrinsic matrix (3x3)
        pose_matrix: Camera pose matrix (4x4) - transforms from camera to world coords
        color_image: Optional RGB image (H x W x 3)
        depth_scale: Scale factor to convert depth to meters (1000.0 for mm->m)
        max_depth: Maximum depth threshold in meters
    
    Returns:
        Open3D point cloud
    """
    height, width = depth_image.shape

    print(height)
    print(width)

    print(intrinsic_matrix.shape)
    
    # Create camera intrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        intrinsic_matrix[0][0],  # fx
        intrinsic_matrix[1][1],  # fy
        intrinsic_matrix[0][2],  # cx
        intrinsic_matrix[1][2]   # cy
    )
    
    # Convert depth image to Open3D format
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.uint16))
    
    # Create point cloud from depth image
    if color_image is not None:
        color_o3d = o3d.geometry.Image(color_image.astype(np.uint8))
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, 
                depth_scale=depth_scale,
                depth_trunc=max_depth
            ),
            intrinsic
        )
    else:
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, intrinsic,
            depth_scale=depth_scale,
            depth_trunc=max_depth
        )
    
    # Transform to world coordinates if pose is provided
    if pose_matrix is not None:
        pcd.transform(pose_matrix)

    o3d.visualization.draw_geometries([pcd])
    
    return pcd

def combine_pointclouds(depth_images: List[np.ndarray],
                       poses: List[np.ndarray],
                       intrinsic_matrix: np.ndarray,
                       color_images: List[np.ndarray] = None,
                       depth_scale: float = 1000.0,
                       max_depth: float = 10.0,
                       voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Combine multiple depth images with poses into a single point cloud.
    
    Args:
        depth_images: List of depth images
        poses: List of 4x4 pose matrices (camera to world transform)
        intrinsic_matrix: Camera intrinsic matrix (3x3)
        color_images: Optional list of RGB images
        depth_scale: Scale factor for depth values
        max_depth: Maximum depth threshold
        voxel_size: Voxel size for downsampling (0 to disable)
    
    Returns:
        Combined point cloud
    """
    combined_pcd = o3d.geometry.PointCloud()
    
    for i, (depth_img, pose) in enumerate(zip(depth_images, poses)):
        color_img = color_images[i] if color_images else None
        
        # Convert depth image to point cloud
        pcd = depth_to_pointcloud(
            depth_img, intrinsic_matrix, pose, color_img, depth_scale, max_depth
        )
        
        # Add to combined point cloud
        combined_pcd += pcd
        print(f"Processed frame {i+1}/{len(depth_images)}, points: {len(pcd.points)}")
    
    print(f"Total points before filtering: {len(combined_pcd.points)}")
    
    # Remove statistical outliers
    combined_pcd, _ = combined_pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    
    # Downsample if requested
    if voxel_size > 0:
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
        print(f"Points after downsampling: {len(combined_pcd.points)}")
    
    return combined_pcd

def load_pose_matrix(pose_data) -> np.ndarray:
    """
    Convert pose data to 4x4 transformation matrix.
    
    Args:
        pose_data: Can be:
            - 4x4 matrix
            - 7-element array [x,y,z,qx,qy,qz,qw] (translation + quaternion)
            - 6-element array [x,y,z,rx,ry,rz] (translation + euler angles)
    
    Returns:
        4x4 transformation matrix
    """
    if isinstance(pose_data, (list, tuple)):
        pose_data = np.array(pose_data)
    
    if pose_data.shape == (4, 4):
        return pose_data
    elif len(pose_data) == 7:  # Translation + quaternion
        t = pose_data[:3]
        q = pose_data[3:]  # [qx, qy, qz, qw]
        
        # Convert quaternion to rotation matrix
        qx, qy, qz, qw = q
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
        ])
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    elif len(pose_data) == 6:  # Translation + euler angles
        t = pose_data[:3]
        angles = pose_data[3:]
        
        # Convert euler angles to rotation matrix
        rx, ry, rz = angles
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    else:
        raise ValueError(f"Unsupported pose format: {pose_data.shape}")

def plot_transforms_3d(transforms: list[np.ndarray], axis_length: float = 0.1) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for T in transforms:
        origin = T[0:3, 3]
        x_axis = T[0:3, 0] * axis_length
        y_axis = T[0:3, 1] * axis_length
        z_axis = T[0:3, 2] * axis_length

        ax.quiver(*origin, *x_axis, color='r', linewidth=1)
        ax.quiver(*origin, *y_axis, color='g', linewidth=1)
        ax.quiver(*origin, *z_axis, color='b', linewidth=1)

    positions = np.array([T[0:3, 3] for T in transforms])
    ax.plot(positions[:,0], positions[:,1], positions[:,2], color='k', linestyle='--', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    plt.show()

if __name__ == "__main__":
    
  transforms = pd.read_pickle('data/transforms.pkl')

  data_path = 'data'

  timestamps = extract_timestamps_from_pngs(folder_path=data_path)

  pose_array = []

  for timestamp in timestamps[:-1]:
    pose_array.append(pose_stamped_to_matrix(pose_stamped=transforms[timestamp]))

  plot_transforms_3d(transforms=pose_array)

  n_imgs = len(timestamps)
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')

  print(cam_intr)

  depth_images = []
  color_images = []  
  
  for i in range(n_imgs):  
      color_img = cv2.cvtColor(cv2.imread(f"data/left_{timestamp}.png"), cv2.COLOR_BGR2RGB)

      cv2.imshow("color image", color_img)

      print(f"data/depth_cv_{timestamp}.npy")
      depth_img= np.load(f"data/depth_cv_{timestamp}.npy")

      print(np.count_nonzero(~np.isnan(depth_img)))
      depth_images.append(depth_img)
      color_images.append(color_img)
  
  # combined_pcd = combine_pointclouds(
  #     depth_images=depth_images,
  #     poses=pose_array,
  #     intrinsic_matrix=cam_intr,
  #     color_images=None,  
  #     depth_scale=1.0,  
  #     voxel_size=0.000001     
  # )
  
  # o3d.io.write_point_cloud("combined_pointcloud.ply", combined_pcd)
  
  # o3d.visualization.draw_geometries([combined_pcd])