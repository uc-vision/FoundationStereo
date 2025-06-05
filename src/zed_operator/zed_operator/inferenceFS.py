import os
import sys
import imageio.v2 as imageio 
import cv2
import numpy as np
import torch
import onnxruntime as ort
import open3d as o3d 

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8, force_square=False):
        self.ht, self.wd = dims[-2:]
        if force_square:
          max_side = max(self.ht, self.wd)
          pad_ht = ((max_side // divis_by) + 1) * divis_by - self.ht
          pad_wd = ((max_side // divis_by) + 1) * divis_by - self.wd
        else:
          pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
          pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
def toOpen3dCloud(points,colors=None,normals=None):
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
  if colors is not None:
    if colors.max()>1:
      colors = colors/255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  if normals is not None:
    cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
  return cloud

def depth2xyzmap(depth:np.ndarray, K, uvs:np.ndarray=None, zmin=0.1):
  invalid_mask = (depth<zmin)
  H,W = depth.shape[:2]
  if uvs is None:
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
  else:
    uvs = uvs.round().astype(int)
    us = uvs[:,0]
    vs = uvs[:,1]
  zs = depth[vs,us]
  xs = (us-K[0,2])*zs/K[0,0]
  ys = (vs-K[1,2])*zs/K[1,1]
  pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
  xyz_map = np.zeros((H,W,3), dtype=np.float32)
  xyz_map[vs,us] = pts
  if invalid_mask.any():
    xyz_map[invalid_mask] = 0
  return xyz_map

def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output={}):
  """
  @disp: np array (H,W)
  @invalid_thres: > thres is invalid
  """
  disp = disp.copy()
  H,W = disp.shape[:2]
  invalid_mask = disp>=invalid_thres
  if (invalid_mask==0).sum()==0:
    other_output['min_val'] = None
    other_output['max_val'] = None
    return np.zeros((H,W,3))
  if min_val is None:
    min_val = disp[invalid_mask==0].min()
  if max_val is None:
    max_val = disp[invalid_mask==0].max()
  other_output['min_val'] = min_val
  other_output['max_val'] = max_val
  vis = ((disp-min_val)/(max_val-min_val)).clip(0,1) * 255
  if cmap is None:
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[...,::-1]
  else:
    vis = cmap(vis.astype(np.uint8))[...,:3]*255
  if invalid_mask.any():
    vis[invalid_mask] = 0
  return vis.astype(np.uint8)

def onnx_inference(ort_session, img0=None, img1=None):
    input_details = ort_session.get_inputs()
    output_details = ort_session.get_outputs()

    input_left_name = input_details[0].name
    input_right_name = input_details[1].name
    output_disparity_name = output_details[0].name

    img0 = imageio.imread('assets/left_1.png')
    img1 = imageio.imread('assets/right_1.png')

    # resize images
    img0 = cv2.resize(img0, (960, 736), interpolation=cv2.INTER_LINEAR)
    img1 = cv2.resize(img1, (960, 736), interpolation=cv2.INTER_LINEAR)

    img0_ori = img0.copy()
    H,W = img0.shape[:2]

    img0_torch = torch.as_tensor(img0).float()[None].permute(0, 3, 1, 2) 
    img1_torch = torch.as_tensor(img1).float()[None].permute(0, 3, 1, 2) 

    padder = InputPadder(img0_torch.shape, divis_by=32, force_square=False)
    img0_padded_torch, img1_padded_torch = padder.pad(img0_torch, img1_torch)

    img0_onnx_input = img0_padded_torch.cpu().numpy().astype(np.float32)
    img1_onnx_input = img1_padded_torch.cpu().numpy().astype(np.float32)

    ort_inputs = {
        input_left_name: img0_onnx_input,
        input_right_name: img1_onnx_input
    }

    print("Running inference...")
    disp = ort_session.run([output_disparity_name], ort_inputs)[0]
    print("Inference done.")

    disp = torch.from_numpy(disp).float()
    disp = padder.unpad(disp)
    disp = disp.data.cpu().numpy().reshape(H,W)
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)

    yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx-disp
    invalid = us_right<0
    disp[invalid] = np.inf

    intrinsic_file = '/home/canterbury/FoundationStereo/assets/K.txt'
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
    # K[:2] *= scale
    depth = K[0,0]*baseline/disp
    # np.save(f'{out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=2)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    # o3d.io.write_point_cloud(f'{out_dir}/cloud.ply', pcd)
    # print(f"PCL saved to {out_dir}")


    print("[Optional step] denoise point cloud...")
    cl, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
    inlier_cloud = pcd.select_by_index(ind)
    # o3d.io.write_point_cloud(f'{out_dir}/cloud_denoise.ply', inlier_cloud)
    pcd = inlier_cloud

    # print("Visualizing point cloud. Press ESC to exit.")
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.get_render_option().point_size = 1.0
    # vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    # vis.run()
    # vis.destroy_window()

    return pcd