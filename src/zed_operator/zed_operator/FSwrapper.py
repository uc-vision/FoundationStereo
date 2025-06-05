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
from dataclasses import dataclass, field
from omegaconf import OmegaConf

class StereoArgs:
    def __init__(self, code_dir='.', **kwargs):
        self.code_dir = code_dir

        self.left_file = kwargs.get('left_file', f'{self.code_dir}/../assets/left.png')
        self.right_file = kwargs.get('right_file', f'{self.code_dir}/../assets/right.png')
        self.intrinsic_file = kwargs.get('intrinsic_file', f'/local/FoundationStereo/assets/K.txt')
        self.ckpt_dir = kwargs.get('ckpt_dir', f'{self.code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth')
        self.out_dir = kwargs.get('out_dir', f'{self.code_dir}/../output/')
        
        self.scale = kwargs.get('scale', 1.0)
        self.hiera = kwargs.get('hiera', 1)
        self.z_far = kwargs.get('z_far', 1.5)
        self.valid_iters = kwargs.get('valid_iters', 16)
        self.get_pc = kwargs.get('get_pc', 1)
        self.remove_invisible = kwargs.get('remove_invisible', 1)
        self.denoise_cloud = kwargs.get('denoise_cloud', 1)
        self.denoise_nb_points = kwargs.get('denoise_nb_points', 30)
        self.denoise_radius = kwargs.get('denoise_radius', 0.03)


class FSWrapper:
    def __init__(self):
        self.fs_path = os.path.join(os.path.dirname(__file__), '..', '..', 'FoundationStereo')
        self.DEVICE = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        self._import_FS_stuff()

        self.output_dir = '/local/zed_angles/FS_out'
        os.makedirs(self.output_dir, exist_ok=True)

    
    def _import_FS_stuff(self):
        sys.path.insert(0, self.fs_path)
        try:
            from core.utils.utils import InputPadder
            from Utils import toOpen3dCloud, depth2xyzmap, vis_disparity, set_seed
            from core.foundation_stereo import FoundationStereo
            
            # Store classes as instance variables
            self.FoundationStereo = FoundationStereo
            self.InputPadder = InputPadder
            self.toOpen3dCloud = toOpen3dCloud
            self.depth2xyzmap = depth2xyzmap
            self.vis_disparity = vis_disparity
            self.set_seed = set_seed
            
        finally:
            sys.path.remove(self.fs_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def instantiate_model(self):
        # set_logging_format()
        self.set_seed(0)
        torch.autograd.set_grad_enabled(False)

        self.args = StereoArgs(
            ckpt_dir='/local/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth',
            out_dir=self.output_dir,
            left_file='/local/zed_angles/again/20250528_112244/left_20250528_112244.png',
            right_file='/local/zed_angles/again/20250528_112244/right_20250528_112244.png',
            scale = 0.7
        )
        ckpt_dir = self.args.ckpt_dir
        cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        for k in self.args.__dict__:
            cfg[k] = self.args.__dict__[k]
        self.args = OmegaConf.create(cfg)

        print(f"args:\n{self.args}")
        print(f"Using pretrained model from {ckpt_dir}")

        self.model = self.FoundationStereo(self.args)

        ckpt = torch.load(ckpt_dir, weights_only=False)
        # logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
        self.model.load_state_dict(ckpt['model'])

        self.model.cuda()
        self.model.eval()
        self.model = self.model.half()

    def run_inference(self, left_img=None, right_img=None, denoise_cloud=False):
        if left_img is not None and right_img is not None:
            img0 = left_img
            img1 = right_img
        else:
            print(f"Test on imaged from file")
            img0 = imageio.imread(self.args.left_file)
            img1 = imageio.imread(self.args.right_file)
        scale = self.args.scale
        assert scale<=1, "scale must be <=1"
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
        # img0 = cv2.resize(img0, (1440, 960), interpolation=cv2.INTER_LINEAR)
        # img1 = cv2.resize(img1, (1440, 960), interpolation=cv2.INTER_LINEAR)

        H,W = img0.shape[:2]
        img0_ori = img0.copy()
        print(f"img0: {img0.shape}")

        img0 = torch.as_tensor(img0).cuda().half()[None].permute(0,3,1,2)
        img1 = torch.as_tensor(img1).cuda().half()[None].permute(0,3,1,2)
        padder = self.InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        str = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast(True):
                if not self.args.hiera:
                    disp = self.model.forward(img0, img1, iters=self.args.valid_iters, test_mode=True)
                else:
                    disp = self.model.run_hierachical(img0, img1, iters=self.args.valid_iters, test_mode=True, small_ratio=0.5)
                print(f"forward time: {time.time()-str:.2f}s")
            disp = padder.unpad(disp.float())
            disp = disp.data.cpu().numpy().reshape(H,W)
            vis = self.vis_disparity(disp)
            vis = np.concatenate([img0_ori, vis], axis=1)
            imageio.imwrite(f'{self.args.out_dir}/vis.png', vis)
            print(f"Output saved to {self.args.out_dir}")

        if self.args.remove_invisible:
            yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
            us_right = xx-disp
            invalid = us_right<0
            disp[invalid] = np.inf

        if self.args.get_pc:
            with open(self.args.intrinsic_file, 'r') as f:
                lines = f.readlines()
                K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
                baseline = float(lines[1])
            K[:2] *= scale
            depth = K[0,0]*baseline/disp
            # np.save(f'{self.args.out_dir}/depth_meter.npy', depth)
            xyz_map = self.depth2xyzmap(depth, K)
            pcd = self.toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
            keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=self.args.z_far)
            keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
            pcd = pcd.select_by_index(keep_ids)
            # o3d.io.write_point_cloud(f'{self.args.out_dir}/cloud.ply', pcd)
            # print(f"PCL saved to {self.args.out_dir}")

            if denoise_cloud:
                print("[Optional step] denoise point cloud...")
                cl, ind = pcd.remove_radius_outlier(nb_points=self.args.denoise_nb_points, radius=self.args.denoise_radius)
                inlier_cloud = pcd.select_by_index(ind)
                # o3d.io.write_point_cloud(f'{self.args.out_dir}/cloud_denoise.ply', inlier_cloud)
                pcd = inlier_cloud

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
    wrapper = FSWrapper()
    wrapper.instantiate_model()
    print("FS wrapper initialized successfully.")

    wrapper.run_inference()

    # dirs = [
    # "/local/zed_angles/again/20250528_112244",
    # "/local/zed_angles/again/20250528_112256",
    # "/local/zed_angles/again/20250528_112308",
    # "/local/zed_angles/again/20250528_112321",
    # "/local/zed_angles/again/20250528_112332",
    # "/local/zed_angles/again/20250528_112342",
    # "/local/zed_angles/again/20250528_112356",
    # "/local/zed_angles/again/20250528_112411",
    # "/local/zed_angles/again/20250528_112429"
    # ]

    # for dir in dirs:
    #     id = dir.split('/')[-1]
    #     left_img = os.path.join(dir, f'left_{id}.png')
    #     right_img = os.path.join(dir, f'right_{id}.png')

    #     strt = time.time()
    #     wrapper.run_inference_from_file(left_img, right_img, f'/local/zed_angles/fs_out/{id}')
    #     print(f"Inference completed in {time.time() - strt} seconds.")
