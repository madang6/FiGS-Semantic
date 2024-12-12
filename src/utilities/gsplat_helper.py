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
   
def pose2nerf_transform(pose):

    # Camera to Body Frame
    T_c2b = np.array([
        [-0.00866, -0.12186, -0.99250,  0.10000],
        [ 0.99938, -0.03463, -0.00446, -0.03100],
        [-0.03383, -0.99194,  0.12209, -0.01200],
        [ 0.00000,  0.00000,  0.00000,  1.00000]
    ])
    
    # Body to World Frame
    T_b2w = np.eye(4)
    T_b2w[0:3,:] = np.hstack((R.from_quat(pose[3:]).as_matrix(),pose[0:3].reshape(-1,1)))

    # World to GSplat Frame
    T_w2g = np.array([
        [ 1.000, 0.000, 0.000, 0.000],
        [ 0.000,-1.000, 0.000, 0.000],
        [ 0.000, 0.000,-1.000, 0.000],
        [ 0.000, 0.000, 0.000, 1.000]
    ])
    
    # Get image transform
    T_c2n = T_w2g@T_b2w@T_c2b

    return T_c2n