"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion
import pandas as pd 

import numpy as np
from geometry_msgs.msg import TransformStamped, PoseStamped

import os
import re
from typing import List

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

def transform_stamped_to_matrix(transform_stamped: TransformStamped) -> np.ndarray:
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation

    x, y, z, w = q.x, q.y, q.z, q.w

    r00 = 1 - 2 * (y**2 + z**2)
    r01 = 2 * (x*y - z*w)
    r02 = 2 * (x*z + y*w)

    r10 = 2 * (x*y + z*w)
    r11 = 1 - 2 * (x**2 + z**2)
    r12 = 2 * (y*z - x*w)

    r20 = 2 * (x*z - y*w)
    r21 = 2 * (y*z + x*w)
    r22 = 1 - 2 * (x**2 + y**2)

    matrix = np.array([
        [r00, r01, r02, t.x],
        [r10, r11, r12, t.y],
        [r20, r21, r22, t.z],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return matrix

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_transforms_3d(transforms: list[np.ndarray], axis_length: float = 0.1) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for T in transforms:
        origin = T[0:3, 3]
        x_axis = T[0:3, 0] * axis_length
        y_axis = T[0:3, 1] * axis_length
        z_axis = T[0:3, 2] * axis_length

        # Check orthogonality (dot product should be ~0 for perpendicular vectors)
        xy_dot = np.dot(x_axis, y_axis)
        yz_dot = np.dot(y_axis, z_axis)
        zx_dot = np.dot(z_axis, x_axis)

        print("checking orthogonality")
        print(f"Dot products: XY={xy_dot:.2e}, YZ={yz_dot:.2e}, ZX={zx_dot:.2e}")
        
        tol = 1e-6  # Tolerance for floating point comparison
        if not (abs(xy_dot) < tol and abs(yz_dot) < tol and abs(zx_dot) < tol):
            print(f"Transform: Axes are not orthogonal! "
                  f"Dot products: XY={xy_dot:.2e}, YZ={yz_dot:.2e}, ZX={zx_dot:.2e}")

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

from mpl_toolkits.mplot3d import Axes3D

def quaternion_to_rotation_matrix(orientation):
    """Convert quaternion to 3x3 rotation matrix."""
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    
    # Normalize quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Convert to rotation matrix
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

def visualize_poses_3d(pose_stamped_list, scale=0.1, show_trajectory=True, 
                      axis_labels=True, title="3D Pose Visualization"):
    """
    Visualize a list of PoseStamped messages in 3D.
    
    Args:
        pose_stamped_list: List of PoseStamped messages
        scale: Scale factor for the coordinate frame arrows (default: 0.1)
        show_trajectory: Whether to draw lines connecting poses (default: True)
        axis_labels: Whether to show axis labels (default: True)
        title: Plot title (default: "3D Pose Visualization")
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    positions = []
    
    for i, pose_stamped in enumerate(pose_stamped_list):
        pose = pose_stamped.pose
        
        # Extract position
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        positions.append(pos)
        
        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix(pose.orientation)
        
        # Define coordinate frame vectors (scaled)
        x_axis = R[:, 0] * scale  # Red arrow (X-axis)
        y_axis = R[:, 1] * scale  # Green arrow (Y-axis)  
        z_axis = R[:, 2] * scale  # Blue arrow (Z-axis)
        
        # Plot coordinate frame arrows
        ax.quiver(pos[0], pos[1], pos[2], 
                 x_axis[0], x_axis[1], x_axis[2], 
                 color='red', alpha=0.8, arrow_length_ratio=0.1)
        
        ax.quiver(pos[0], pos[1], pos[2], 
                 y_axis[0], y_axis[1], y_axis[2], 
                 color='green', alpha=0.8, arrow_length_ratio=0.1)
        
        ax.quiver(pos[0], pos[1], pos[2], 
                 z_axis[0], z_axis[1], z_axis[2], 
                 color='blue', alpha=0.8, arrow_length_ratio=0.1)
        
        # Plot position as a point
        ax.scatter(pos[0], pos[1], pos[2], c='black', s=20)
        
        # Optionally add pose number labels
        ax.text(pos[0], pos[1], pos[2], f'  {i}', fontsize=8)
    
    # Draw trajectory line connecting all poses
    if show_trajectory and len(positions) > 1:
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               'k--', alpha=0.6, linewidth=1, label='Trajectory')
    
    # Set labels and title
    if axis_labels:
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
    
    ax.set_title(title)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='X-axis'),
        Line2D([0], [0], color='green', lw=2, label='Y-axis'),
        Line2D([0], [0], color='blue', lw=2, label='Z-axis'),
        Line2D([0], [0], color='black', marker='o', linestyle='None', label='Position')
    ]
    if show_trajectory:
        legend_elements.append(Line2D([0], [0], color='black', linestyle='--', label='Trajectory'))
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Show grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

  transforms = pd.read_pickle('data/transforms.pkl')

  data_path = 'data'

  timestamps = extract_timestamps_from_pngs(folder_path=data_path)

  pose_array = []
  pose_stamped_array = []

  for timestamp in timestamps[:-1]:
    pose_array.append(pose_stamped_to_matrix(pose_stamped=transforms[timestamp]))
    pose_stamped_array.append(transforms[timestamp])

  # visualize_poses_3d(pose_stamped_array)

  plot_transforms_3d(transforms=pose_array)

  print(len(pose_array))

  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = len(timestamps)
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  vol_bnds = np.zeros((3,2))

  for timestamp in timestamps[:-1]:
    # Read depth image and camera pose
    depth_im = cv2.imread(f"data/{timestamp}_depth.png").astype(np.float32)[:, :, 0]
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    # print(depth_im)
    # depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    cam_pose = pose_stamped_to_matrix(pose_stamped=transforms[timestamp])  # 4x4 rigid transformation matrix
    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  # print(f"volume bounds {vol_bnds}")

  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.001)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  count = 0
  for timestamp in timestamps[:-1]:
    print("Fusing frame %d/%d"%(count+1, n_imgs))
    count+=1

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread(f"data/left_{timestamp}.png"), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(f"data/{timestamp}_depth.png").astype(np.float32)[:, :, 0]

    depth_im /= 1000.
    # depth_im[depth_im == 65.535] = 0
    cam_pose = pose_stamped_to_matrix(pose_stamped=transforms[timestamp])

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)