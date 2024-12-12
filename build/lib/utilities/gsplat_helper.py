from math import e
from pathlib import Path
import os
from re import T
import torch
import numpy as np
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from scipy.spatial.transform import Rotation as R
from typing import Literal, List
   
def get_nerf(map:str):
    # Generate some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    main_dir_path = os.getcwd()
    nerf_dir_path = os.path.join(workspace_path,"nerf_data")
    maps = {
        "gate_left":"sv_917_3_left_nerfstudio",
        "gate_right":"sv_917_3_right_nerfstudio",
        "gate_mid":"sv_1007_gate_mid",
        "clutter":"sv_712_nerfstudio",
        "backroom":"backroom",
        "flightroom":"sv_1018_3",
        "SRC":"srcvid"
    }

    map_folder = os.path.join(nerf_dir_path,'outputs',maps[map])
    for root, _, files in os.walk(map_folder):
        if 'config.yml' in files:
            nerf_cfg_path = os.path.join(root, 'config.yml')

    # Go into NeRF data folder and get NeRF object (because the NeRF instantation
    # requires the current working directory to be the NeRF data folder)
    os.chdir(nerf_dir_path)
    nerf = NeRF(Path(nerf_cfg_path))
    os.chdir(main_dir_path)

    return nerf

def pose2nerf_transform(pose):

    # Realsense to Drone Frame
    T_r2d = np.array([
        [ 0.99250, -0.00866,  0.12186,  0.10000],
        [ 0.00446,  0.99938,  0.03463, -0.03100],
        [-0.12209, -0.03383,  0.99194, -0.01200],
        [ 0.00000,  0.00000,  0.00000,  1.00000]
    ])
    
    # Drone to Flightroom Frame
    T_d2f = np.eye(4)
    T_d2f[0:3,:] = np.hstack((R.from_quat(pose[3:]).as_matrix(),pose[0:3].reshape(-1,1)))

    # Flightroom Frame to NeRF world frame
    T_f2n = np.array([
        [ 1.000, 0.000, 0.000, 0.000],
        [ 0.000,-1.000, 0.000, 0.000],
        [ 0.000, 0.000,-1.000, 0.000],
        [ 0.000, 0.000, 0.000, 1.000]
    ])

    # Camera convention frame to realsense frame
    T_c2r = np.array([
        [ 0.0, 0.0,-1.0, 0.0],
        [ 1.0, 0.0, 0.0, 0.0],
        [ 0.0,-1.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0]
    ])

    # Get image transform
    T_c2n = T_f2n@T_d2f@T_r2d@T_c2r

    return T_c2n