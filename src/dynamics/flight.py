import numpy as np
import torch
import shutil
import os
import utilities.trajectory_helper as th

from typing import Dict,List,Type,Union
from control.base_controller import BaseController
from render.gsplat import GSplat
from acados_template import AcadosSimSolver, AcadosSim
from dynamics.model_equations import export_quadcopter_ode_model
from dynamics.model_specifications import generate_specifications

class Flight():
    def __init__(self, rollout_config:Dict[str,Union[int,float,List[float]]],
                 frame_config:Dict[str,Union[int,float,List[float]]], name:str='flyer') -> None:
        """
        Flying class for simulating drone flights.

        Args:
            - rollout_config: Configuration dictionary for the rollout.
            - frame_config: Configuration dictionary for the frame.
            - name: Name of the flight.

        Variables:
            - nx: Number of states in the system.
            - nu: Number of controls in the system.
            - hz_sim: Simulation rate.
            - t_dly: Delay in control.
            - mu_md: Mean of the model noise.
            - std_md: Standard deviation of the model noise.
            - mu_sn: Mean of the sensor noise.
            - std_sn: Standard deviation of the sensor noise.
            - use_fusion: Use sensor model fusion.
            - Wf: Fusion weights.
            - drn_spec: Drone specifications.
            - simulator: Acados simulator.
            - code_export_path: Path to the generated code.
            - simulator_path: Path to the simulator json file.
        
        """

        # Some useful intermediate variables
        drn_spec = generate_specifications(frame_config)
        sim_json = 'acados_sim_nlp_'+name+'.json'

        sim = AcadosSim()
        sim.model = export_quadcopter_ode_model(drn_spec["m"],drn_spec["tn"])  
        sim.solver_options.T = 1/rollout_config["hz_sim"]
        sim.solver_options.integrator_type = 'IRK'
        sim.code_export_directory = os.path.join(sim.code_export_directory,name)

        # Class variables
        self.nx,self.nu = sim.model.x.size()[0],sim.model.u.size()[0]
        self.hz_sim = rollout_config["hz_sim"]
        self.t_dly = rollout_config["delay"]
        self.mu_md = np.array(rollout_config["model_noise"]["mean"])
        self.std_md = np.array(rollout_config["model_noise"]["std"])
        self.mu_sn = np.array(rollout_config["sensor_noise"]["mean"])
        self.std_sn = np.array(rollout_config["sensor_noise"]["std"])
        self.use_fusion = rollout_config["sensor_model_fusion"]["use_fusion"]
        self.Wf = np.diag(rollout_config["sensor_model_fusion"]["weights"])
        self.drn_spec = drn_spec
        self.simulator = AcadosSimSolver(sim, json_file=sim_json, verbose=False)
        
        self.code_export_path = sim.code_export_directory
        self.simulator_path = os.path.join(os.getcwd(),sim_json)

    def simulate(self,controller:Type[BaseController],gsplat:GSplat,
                 t0:float,tf:int,x0:np.ndarray,
                 obj:Union[None,np.ndarray]=None):
        """
        Method to simulate the drone flight.

        Args:
            - controller: Controller for the drone.
            - gsplat: GSplat object for rendering images.
            - t0: Initial time.
            - tf: Final time.
            - x0: Initial state.
            - obj: Object to track.

        Returns:
            - Tro: Time vector.
            - Xro: State vector.
            - Uro: Control vector.
            - Iro: Image vector.
            - Tsol: Solution time vector.
            - Adv: Advisor vector (if used).

        """
        
        # Simulation Variables
        dt = np.round(tf-t0)
        Nsim = int(dt*self.hz_sim)
        Nctl = int(dt*controller.hz)
        n_sim2ctl = int(self.hz_sim/controller.hz)
        n_delay = int(self.t_dly*self.hz_sim)
        height,width,channels = int(gsplat.camera_out.height.item()),int(gsplat.camera_out.width.item()),3
        T_c2b = self.drn_spec["T_c2b"]

        # Extract sensor and model parameters
        mu_md  = self.mu_md*(1/n_sim2ctl)         # Scale model mean noise to control rate
        std_md = self.std_md*(1/n_sim2ctl)        # Scale model std noise to control rate
        mu_sn = 1.0*self.mu_sn
        std_sn = 1.0*self.std_sn
        Wf_sn,Wf_md = self.Wf,1-self.Wf

        # Rollout Variables
        Tro,Xro,Uro = np.zeros(Nctl+1),np.zeros((self.nx,Nctl+1)),np.zeros((self.nu,Nctl))
        Iro = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
        Xro[:,0] = x0

        # Diagnostics Variables
        Tsol = np.zeros((4,Nctl))
        Adv = np.zeros((self.nu,Nctl))
        
        # Transient Variables
        xcr,xpr,xsn = x0.copy(),x0.copy(),x0.copy()
        ucm = np.array([-self.drn_spec['m']/self.drn_spec['tn'],0.0,0.0,0.0])
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

                Iro[k,:,:,:] = icr
                Tro[k] = tcr
                Xro[:,k+1] = xcr
                Uro[:,k] = ucm
                Tsol[:,k] = tsol
                Adv[:,k] = adv

        # Log final time
        Tro[Nctl] = t0+Nsim/self.hz_sim

        return Tro,Xro,Uro,Iro,Tsol,Adv
    
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