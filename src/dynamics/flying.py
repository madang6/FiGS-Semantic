import numpy as np
import torch
import shutil
import os
from utilities.configs import ConFiGS
import utilities.trajectory_helper as th
import utilities.dynamics_helper as dh

from typing import List,Type,Union,Literal
from controller.base_controller import BaseController
from render.scene_render import SceneRender
# import dynamics.quadcopter_config as qc
# import synthesize.generate_data as gd
# import visualize.record_flight as rf
# from torchvision.io import write_video
# from torchvision.transforms import Resize
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from dynamics.quadcopter_model import export_quadcopter_ode_model

class Flying():
    def __init__(self, config:ConFiGS, name:str='flyer') -> None:
        """
        Flying class for simulating drone flights.

        Args:
        - config: FiGS configuration dictionary.

        Variables:
        
        """

        # Extract the relevant configurations
        flying_config = config.get_config("flying_parameters")
        drone_config  = config.get_config("drone_parameters")

        # Some useful intermediate variables
        drn_spec = dh.generate_specifications(drone_config)
        sim_json = 'acados_sim_nlp_'+name+'.json'

        sim = AcadosSim()
        sim.model = export_quadcopter_ode_model(drn_spec["m"],drn_spec["tn"])  
        sim.solver_options.T = 1/flying_config["simulation"]["hz_sim"]
        sim.solver_options.integrator_type = 'IRK'
        sim.code_export_directory = os.path.join(sim.code_export_directory,name)

        # Class variables
        self.nx,self.nu = sim.model.x.size()[0],sim.model.u.size()[0]
        self.hz_sim = flying_config["simulation"]["hz_sim"]
        self.t_dly = flying_config["simulation"]["delay"]
        self.mu_md = np.array(flying_config["model_noise"]["mean"])
        self.std_md = np.array(flying_config["model_noise"]["std"])
        self.mu_sn = np.array(flying_config["sensor_noise"]["mean"])
        self.std_sn = np.array(flying_config["sensor_noise"]["std"])
        self.use_fusion = flying_config["sensor_model_fusion"]["use_fusion"]
        self.Wf = np.diag(flying_config["sensor_model_fusion"]["weights"])
        self.simulator = AcadosSimSolver(sim, json_file=sim_json, verbose=False)
        
        self.code_export_path = sim.code_export_directory
        self.simulator_path = os.path.join(os.getcwd(),sim_json)

    def simulate(self,controller:Type[BaseController],gsplat:SceneRender,
                 t0:float,tf:int,x0:np.ndarray,
                 obj:Union[None,np.ndarray]=None):
        
        # Simulation Variables
        dt = np.round(tf-t0)
        Nsim = int(dt*self.hz_sim)
        Nctl = int(dt*controller.hz)
        n_sim2ctl = int(self.hz_sim/controller.hz)
        n_delay = int(self.t_dly*self.hz_sim)
        height,width,channels = int(gsplat.camera_out.height.item()),int(gsplat.camera_out.width.item()),3
        T_c2b = np.array([
            [-0.00866, -0.12186, -0.99250,  0.10000],
            [ 0.99938, -0.03463, -0.00446, -0.03100],
            [-0.03383, -0.99194,  0.12209, -0.01200],
            [ 0.00000,  0.00000,  0.00000,  1.00000]
        ])

        # Extract sensor and model parameters
        mu_md  = self.mu_md*(1/n_sim2ctl)         # Scale model mean noise to control rate
        std_md = self.std_md*(1/n_sim2ctl)        # Scale model std noise to control rate
        mu_sn = 1.0*self.mu_sn
        std_sn = 1.0*self.std_sn
        Wf_sn,Wf_md = self.Wf,1-self.Wf

        # Rollout Variables
        Tro,Xro,Uro = np.zeros(Nctl+1),np.zeros((self.nx,Nctl+1)),np.zeros((self.nu,Nctl))
        Imgs = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
        Xro[:,0] = x0

        # Diagnostics Variables
        Tsol = np.zeros((4,Nctl))
        Adv = np.zeros((self.nu,Nctl))
        
        # Transient Variables
        xcr,xpr,xsn = x0.copy(),x0.copy(),x0.copy()
        ucm = np.array([-0.5,0.0,0.0,0.0])
        udl = np.hstack((ucm.reshape(-1,1),ucm.reshape(-1,1)))
        zcr = torch.zeros(controller.nzcr) if isinstance(controller.nzcr, int) else None

        # Rollout
        for i in range(Nsim):
            # Get current time and state
            tcr = t0+i/self.hz_sim

            # Control
            if i % n_sim2ctl == 0:
                # Get current image
                Tb2w = th.xv_to_T(xcr)
                T_c2w = Tb2w@T_c2b
                icr = gsplat.render_rgb(T_c2w)

                # Add sensor noise and syncronize estimated state
                if self.use_fusion:
                    xsn += np.random.normal(loc=mu_sn,scale=std_sn)
                    xsn = Wf_sn@xsn + Wf_md@xcr
                else:
                    xsn = xcr + np.random.normal(loc=mu_sn,scale=std_sn)
                xsn[6:10] = th.obedient_quaternion(xsn[6:10],xpr[6:10])

                # Generate controller command
                ucm,zcr,adv,tsol = controller.control(tcr,xsn,ucm,obj,icr,zcr)

                # Update delay buffer
                udl[:,0] = udl[:,1]
                udl[:,1] = ucm

            # Extract delayed command
            uin = udl[:,0] if i%n_sim2ctl < n_delay else udl[:,1]

            # Simulate both estimated and actual states
            xcr = self.simulator.simulate(x=xcr,u=uin)
            if self.use_fusion:
                xsn = self.simulator.simulate(x=xsn,u=uin)

            # Add model noise
            xcr = xcr + np.random.normal(loc=mu_md,scale=std_md)
            xcr[6:10] = th.obedient_quaternion(xcr[6:10],xpr[6:10])

            # Update previous state
            xpr = xcr
            
            # Store values
            if i % n_sim2ctl == 0:
                k = i//n_sim2ctl

                Imgs[k,:,:,:] = icr
                Tro[k] = tcr
                Xro[:,k+1] = xcr
                Uro[:,k] = ucm
                Tsol[:,k] = tsol
                Adv[:,k] = adv

        # Log final time
        Tro[Nctl] = t0+Nsim/self.hz_sim

        return Tro,Xro,Uro,Imgs,Tsol,Adv

    def clear_generated_code(self):
        """
        Method to clear the generated code and files to ensure the code is recompiled correctly each time.
        """

        # Clear the generated code
        try:
            os.remove(self.simulator_path)
            shutil.rmtree(self.code_export_path)
        except:
            pass
        
        # Clear the parent directory if empty
        parent_dir_path = os.path.dirname(self.code_export_path)
        if not os.listdir(parent_dir_path) and (os.path.basename(parent_dir_path) == 'c_generated_code'):
            shutil.rmtree(parent_dir_path)