import numpy as np
import os
import torch
from pathlib import Path
from typing import Dict,Union
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup

class SceneRender():
    def __init__(self, gsplat_config: Dict[str,Union[str,Path]]) -> None:
        """
        SceneRender class for rendering images from GSplat pipeline.

        Args:
        - gsplat_config: Path to GSplat configuration file.

        Variables:
        - device: Device to run the pipeline on.
        - config: Configuration for the pipeline.
        - pipeline: Pipeline for the GSplat model.
        - camera_out: Output camera for the pipeline.
        - T_w2g: Transformation matrix from world to GSplat frame.

        """
        
        # Some useful constants
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        T_w2g = np.array([
            [ 1.000, 0.000, 0.000, 0.000],
            [ 0.000,-1.000, 0.000, 0.000],
            [ 0.000, 0.000,-1.000, 0.000],
            [ 0.000, 0.000, 0.000, 1.000]
        ])

        # Class variables
        self.device = device
        self.config,self.pipeline, _, _ = eval_setup(gsplat_config["path"],test_mode="inference")
        self.camera_out = self.generate_output_camera()
        self.T_w2g = T_w2g

    def generate_output_camera(self, width:int=640, height:int=360,
                 fx:float=462.956, fy:float=463.002,cx:float=323.076, cy:float=181.184) -> Cameras:
        """
        Generate an output camera for the pipeline. By default we use the realsense camera parameters
        when it is set to out 640x360 resolution.

        Args:
        - width: Width of the output image.
        - height: Height of the output image.
        - fx: Focal length in the x direction.
        - fy: Focal length in the y direction.
        - cx: Principal point in the x direction.
        - cy: Principal point in the y direction.

        Returns:
        - camera_out: Output camera for the pipeline.
            
        """

        camera_ref = self.pipeline.datamanager.eval_dataset.cameras[0]
        camera_out = Cameras(
            camera_to_worlds=1.0*camera_ref.camera_to_worlds,
            fx=fx,fy=fy,
            cx=cx,cy=cy,
            width=width,
            height=height,
            camera_type=CameraType.PERSPECTIVE,
        )

        camera_out = camera_out.to(self.device)

        return camera_out
            
    def render_rgb(self, T_c2w:np.ndarray) -> torch.Tensor:
        """
        Render an RGB image from the GSplat pipeline.

        Args:
        - T_c2w: Transformation matrix from camera to world frame.

        Returns:
        - image_rgb: Rendered RGB image.

        """

        # Extract the camera to gsplat pose
        T_c2g = self.T_w2g@T_c2w
        P_c2g = torch.tensor(T_c2g[0:3,:]).float()

        # Render rgb image from the pose
        self.camera_out.camera_to_worlds = P_c2g[None,:3, ...]
        with torch.no_grad():
            image_rgb = self.pipeline.model.get_outputs_for_camera(self.camera_out, obb_box=None)["rgb"]

        # Convert to output image
        image_rgb = image_rgb.cpu().numpy()             # Convert to numpy
        image_rgb = (255*image_rgb).astype(np.uint8)    # Convert to uint8

        return image_rgb