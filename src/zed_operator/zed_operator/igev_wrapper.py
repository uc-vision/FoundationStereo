import sys
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2
import time 
import open3d as o3d
import imageio.v2 as imageio 

class IGEVArgs:
    def __init__(self, **kwargs):
        self.restore_ckpt = kwargs.get('restore_ckpt', './pretrained_models/igev_plusplus/sceneflow.pth')
        self.save_numpy = kwargs.get('save_numpy', False)
        self.left_imgs = kwargs.get('left_imgs', "./demo-imgs/*/im0.png")
        self.right_imgs = kwargs.get('right_imgs', "./demo-imgs/*/im1.png")
        self.output_directory = kwargs.get('output_directory', "demo_output")
        
        self.mixed_precision = kwargs.get('mixed_precision', True)
        self.precision_dtype = kwargs.get('precision_dtype', 'float16')
        self.valid_iters = kwargs.get('valid_iters', 16)
        
        self.hidden_dims = kwargs.get('hidden_dims', [128] * 3)
        self.corr_levels = kwargs.get('corr_levels', 2)
        self.corr_radius = kwargs.get('corr_radius', 4)
        self.n_downsample = kwargs.get('n_downsample', 2)
        self.n_gru_layers = kwargs.get('n_gru_layers', 3)
        self.max_disp = kwargs.get('max_disp', 768)
        
        self.s_disp_range = kwargs.get('s_disp_range', 48)
        self.m_disp_range = kwargs.get('m_disp_range', 96)
        self.l_disp_range = kwargs.get('l_disp_range', 192)
        
        self.s_disp_interval = kwargs.get('s_disp_interval', 1)
        self.m_disp_interval = kwargs.get('m_disp_interval', 2)
        self.l_disp_interval = kwargs.get('l_disp_interval', 4)
        
        if self.precision_dtype not in ['float16', 'bfloat16', 'float32']:
            raise ValueError("precision_dtype must be one of: float16, bfloat16, float32")
        
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


class IGEVWrapper:
    def __init__(self):
        self.igev_path = os.path.join(os.path.dirname(__file__), '..', '..', 'IGEV-plusplus')
        # self.model_path = model_path
        self.DEVICE = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        self._import_igev_classes()

        self.output_dir = '/local/zed_angles/again/igev_out'
        os.makedirs(self.output_dir, exist_ok=True)

    
    def _import_igev_classes(self):
        sys.path.insert(0, self.igev_path)
        try:
            from core.igev_stereo import IGEVStereo
            from core.utils.utils import InputPadder
            from core.utils.frame_utils import readPFM
            
            # Store classes as instance variables
            self.IGEVStereo = IGEVStereo
            self.InputPadder = InputPadder
            self.readPFM = readPFM
            
        finally:
            sys.path.remove(self.igev_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def instantiate_model(self):
        self.args = IGEVArgs(
            restore_ckpt='/local/igev_models/middlebury.pth',
        )

        self.model = torch.nn.DataParallel(self.IGEVStereo(self.args), device_ids=[0])
        self.model.load_state_dict(torch.load(self.args.restore_ckpt))

        self.model = self.model.module
        self.model.to(self.DEVICE)
        self.model.half()
        self.model.eval()

    def run_inference_from_file(self, left_img, right_img, save_path):
        def load_image(imfile):
            img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
            img = torch.from_numpy(img).permute(2, 0, 1).half()
            return img[None].to(self.DEVICE)
        with torch.no_grad():
            image1 = load_image(left_img)
            image2 = load_image(right_img)
            self.img1ori = imageio.imread(left_img)
            self.padder = self.InputPadder(image1.shape, divis_by=32)
            image1, image2 = self.padder.pad(image1, image2)
            disp = self.model(image1, image2, iters=self.args.valid_iters, test_mode=True)
            disp = self.padder.unpad(disp)

            # filename = os.path.join(self.output_dir, '_depth.png')
            filename = save_path + '_depth.png'
            disp = disp.cpu().numpy().squeeze()
            plt.imsave(filename, disp.squeeze(), cmap='jet')

        pcd, depth = self.make_point_cloud(disp)
        filename = save_path + '_cloud.ply'
        o3d.io.write_point_cloud(filename, pcd)
        filename = save_path + '_depth_meter.npy'
        np.save(filename, depth)


    def make_point_cloud(self, disp, save_path=None):
        disp = torch.from_numpy(disp).float()
        # disp = self.padder.unpad(disp)
        H,W = self.img1ori.shape[:2]
        disp = disp.data.cpu().numpy().reshape(H,W)
        vis = vis_disparity(disp)
        vis = np.concatenate([self.img1ori, vis], axis=1)

        yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx-disp
        invalid = us_right<0
        disp[invalid] = np.inf

        # intrinsic_file = '/home/canterbury/FoundationStereo/assets/K.txt'
        intrinsic_file = '/local/FoundationStereo/assets/K.txt'
        with open(intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
            baseline = float(lines[1])
        # K[:2] *= scale
        depth = K[0,0]*baseline/disp

        # np.save(f'{out_dir}/depth_meter.npy', depth)
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1,3), self.img1ori.reshape(-1,3))
        keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=2.0)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        # o3d.io.write_point_cloud(f'{out_dir}/cloud.ply', pcd)
        # print(f"PCL saved to {out_dir}")


        # print("[Optional step] denoise point cloud...")
        # cl, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
        # inlier_cloud = pcd.select_by_index(ind)
        # # o3d.io.write_point_cloud(f'{out_dir}/cloud_denoise.ply', inlier_cloud)
        # pcd = inlier_cloud

        # print("Visualizing point cloud. Press ESC to exit.")
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd)
        # vis.get_render_option().point_size = 1.0
        # vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
        # vis.run()
        # vis.destroy_window()

        return pcd, depth


if __name__ == "__main__":
    wrapper = IGEVWrapper()
    wrapper.instantiate_model()
    print("IGEVWrapper initialized successfully.")

    dirs = [
    "/local/zed_angles/again/20250528_112244",
    "/local/zed_angles/again/20250528_112256",
    "/local/zed_angles/again/20250528_112308",
    "/local/zed_angles/again/20250528_112321",
    "/local/zed_angles/again/20250528_112332",
    "/local/zed_angles/again/20250528_112342",
    "/local/zed_angles/again/20250528_112356",
    "/local/zed_angles/again/20250528_112411",
    "/local/zed_angles/again/20250528_112429"
    ]

    for dir in dirs:
        id = dir.split('/')[-1]
        left_img = os.path.join(dir, f'left_{id}.png')
        right_img = os.path.join(dir, f'right_{id}.png')

        strt = time.time()
        wrapper.run_inference_from_file(left_img, right_img, f'/local/zed_angles/again/igev_out/{id}')
        print(f"Inference completed in {time.time() - strt} seconds.")


    