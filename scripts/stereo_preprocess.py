import os
from PIL import Image
from beartype import beartype
from typing import Tuple, Optional


import cv2
import numpy as np
from beartype import beartype

@beartype
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
  return np.array(pil_img)[:, :, ::-1]

@beartype
def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
  return Image.fromarray(cv2_img[:, :, ::-1])

@beartype
def apply_clahe(
  img: Image.Image,
  clip_limit: float = 2.0,
  grid_size: Tuple[int, int] = (8, 8)
) -> Image.Image:
  
  img_np = np.asarray(img.convert("RGB"))
  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
  img_clahe = np.dstack([clahe.apply(channel) for channel in cv2.split(img_np)])
  return Image.fromarray(img_clahe)

@beartype
def process_stereo_pair(
  left_img: Image.Image,
  right_img: Image.Image,
  *,
  clahe_params: dict = {'clip_limit': 1.5, 'grid_size': (32, 16)}
) -> Tuple[Image.Image, Image.Image]:
  
  left_img = apply_clahe(left_img, **clahe_params)
  right_img = apply_clahe(right_img, **clahe_params)
  
  return left_img, right_img

@beartype
def read_png(
  png_path: str
) -> Optional[Image.Image]:

  try:
    img = Image.open(png_path)
    print(f"Read {png_path}")
    return img
  except Exception as e:
    print(f"Read error {png_path}: {e}")
    return None

@beartype
def process_images(
  img1: Image.Image,
  img2: Image.Image
) -> Tuple[Image.Image, Image.Image]:
    
  print("Processing images")
  return img1, img2

@beartype
def save_png(
  img: Image.Image,
  output_path: str,
  overwrite: bool = True
) -> bool:
    
  if not overwrite and os.path.exists(output_path):
    print(f"Exists: {output_path}")
    return False
  try:
    img.save(output_path, format='PNG', compress_level=0)
    print(f"Saved: {output_path}")
    return True
  except Exception as e:
    print(f"Save error {output_path}: {e}")
    return False

@beartype
def match_exposure(
  img_left: Image.Image,
  img_right: Image.
Image) -> Image.Image:
  
  def _histogram_match(source, template):
    matched = np.zeros_like(source)
    for c in range(source.shape[2]):
      src = source[:, :, c].ravel()
      tmpl = template[:, :, c].ravel()

      s_values, bin_idx, s_counts = np.unique(src, return_inverse=True, return_counts=True)
      t_values, t_counts = np.unique(tmpl, return_counts=True)

      s_quantiles = np.cumsum(s_counts).astype(np.float64)
      s_quantiles /= s_quantiles[-1]
      t_quantiles = np.cumsum(t_counts).astype(np.float64)
      t_quantiles /= t_quantiles[-1]

      interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
      matched[:, :, c] = interp_t_values[bin_idx].reshape(source[:, :, c].shape)

    return matched.astype(np.uint8)

  left_np = np.asarray(img_left.convert("RGB"))
  right_np = np.asarray(img_right.convert("RGB"))

  matched_right_np = _histogram_match(right_np, left_np)
  return Image.fromarray(matched_right_np)

@beartype
def process_and_save_pngs(
    png1_path: str,
    png2_path: str,
    output_folder: str,
    *,
    overwrite: bool = True
) -> None:
    
  os.makedirs(output_folder, exist_ok=True)
  
  left = read_png(png1_path)
  right = read_png(png2_path)
  
  if left is None or right is None:
    return
  
  left, right = process_stereo_pair(left, right)
  
  processed_left = left
  processed_right = match_exposure(left, right)

  output_path1 = os.path.join(output_folder, os.path.basename(png1_path))
  output_path2 = os.path.join(output_folder, os.path.basename(png2_path))
  
  save_png(processed_left, output_path1, overwrite=overwrite)
  save_png(processed_right, output_path2, overwrite=overwrite)

if __name__ == "__main__":
  process_and_save_pngs(
    png1_path="/home/canterbury/FoundationStereo/assets/left_1.png",
    png2_path="/home/canterbury/FoundationStereo/assets/right_1.png",
    output_folder="/home/canterbury/pruning_workspace/scripts/processed_stereo_images",
    overwrite=True
  )