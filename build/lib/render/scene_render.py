import numpy as np
import os
import torch
import utilities.trajectory_helper as th
import utilities.gsplat_helper as gh
from pathlib import Path
from typing import Literal,Dict,Union
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup

class SceneRender():
    def __init__(self, gsplat_config: Dict[str,Union[str,Path]], width:int=640, height:int=360) -> None:
        gsplat_config_path = gsplat_config["path"]
        # Pytorch Config
        use_cuda = torch.cuda.is_available()                    
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        # Get config and pipeline
        self.config, self.pipeline, _, _ = eval_setup(
            gsplat_config_path, 
            test_mode="inference",
        )

        # Get reference camera
        self.camera_ref = self.pipeline.datamanager.eval_dataset.cameras[0]

        # Render parameters
        self.channels = 3
        self.camera_out,self.width,self.height = self.generate_output_camera(width,height)

    def generate_output_camera(self,width:int,height:int):
        fx,fy = 462.956,463.002
        cx,cy = 323.076,181.184
        
        camera_out = Cameras(
            camera_to_worlds=1.0*self.camera_ref.camera_to_worlds,
            fx=fx,fy=fy,
            cx=cx,cy=cy,
            width=width,
            height=height,
            camera_type=CameraType.PERSPECTIVE,
        )

        camera_out = camera_out.to(self.device)

        return camera_out,width,height
    
    def render(self, xcr:np.ndarray, xpr:np.ndarray=None,
               visual_mode:Literal["static","dynamic"]="static"):
        
        if visual_mode == "static":
            image = self.static_render(xcr)
        elif visual_mode == "dynamic":
            image = self.dynamic_render(xcr,xpr)
        else:
            raise ValueError(f"Invalid visual mode: {visual_mode}")
        
        # Convert to numpy
        image = image.cpu().numpy()

        # Convert to uint8
        image = (255*image).astype(np.uint8)

        return image
        
    def static_render(self, xcr:np.ndarray) -> torch.Tensor:
        # Extract the pose
        T_c2n = gh.pose2nerf_transform(np.hstack((xcr[0:3],xcr[6:10])))
        P_c2n = torch.tensor(T_c2n[0:3,:]).float()

        # Render from a single pose
        camera_to_world = P_c2n[None,:3, ...]
        self.camera_out.camera_to_worlds = camera_to_world

        # render outputs
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera(self.camera_out, obb_box=None)

        image = outputs["rgb"]

        return image

    def dynamic_render(self, xcr:np.ndarray,xpr:np.ndarray,frames:int=10) -> torch.Tensor:
        # Get interpolated poses
        Xs = torch.zeros(xcr.shape[0],frames)
        for i in range(xcr.shape[0]):
            Xs[i,:] = torch.linspace(xcr[i], xpr[i], frames)
        
        # Make sure quaternion interpolation is correct
        xref = xcr
        for i in range(frames):
            Xs[6:10,i] = th.obedient_quaternion(Xs[6:10,i],xref)
            xref = Xs[:,i]

        # Render images
        images = []
        for i in range(frames):
            images.append(self.static_render(Xs[:,i]))

        # Average across images
        image = torch.mean(images, axis=0)

        return image