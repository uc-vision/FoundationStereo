# Robotic Pruning Workspace

## Overview
A collection of modules for robotic pruning operations on a UCVision rover. 

**GS-Localisation**: Precise robot localisation and alignment with 3D Gaussian Splatting reconstruction. 

**localisation_ros2**: ROS2 wrapper for GS-Localisation. Publishes pose corrections to ROS2 TF tree.  



# Installing 

1. Clone repo and submodules
```
git clone git@github.com:uc-vision/pruning_workspace.git 
git submodule update --init --recursive

OR

git clone --recurse-submodules git@github.com:uc-vision/pruning_workspace.git 
  
```

2. Activate pixi 
```
pixi shell

Notes:
- you might need to set 'ulimit -n 4096' 
- if fused-ssim fails with torch dependency, run pixi shell without fused-ssim first, then again with it included.
```




