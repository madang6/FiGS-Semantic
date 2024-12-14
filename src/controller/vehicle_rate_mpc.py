import time
import shutil
import os
import numpy as np
import scipy.linalg
import tsplines.min_snap as ms
import utilities.trajectory_helper as th
import utilities.dynamics_helper as dh
from utilities.configs import ConFiGS

from controller.base_controller import BaseController
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import vertcat
from dynamics.quadcopter_model import export_quadcopter_ode_model
from typing import Union,Tuple,Dict
from copy import deepcopy
import visualize.plot_trajectories as pt

class VehicleRateMPC(BaseController):
    def __init__(self, config:ConFiGS, use_RTI:bool=False, name:str="vrmpc") -> None:
        
        """
        Constructor for the VehicleRateMPC class.
        
        Args:
            - config: FiGS configuration dictionary.
            - use_RTI: Use RTI flag.
            - name:   Name of the policy.

        Variables:
            - name:    Name of the policy.
            - Nx:      Number of states.
            - Nu:      Number of inputs.
            - tXUd:    Ideal trajectory.
            - Qk:      State cost matrix.
            - Rk:      Input cost matrix.
            - QN:      Terminal state cost matrix.
            - lbu:     Lower bound on inputs.
            - ubu:     Upper bound on inputs.
            - wts:     Search weights for xv_ds.
            - ns:      Search window size for xv_ds.
            - hz:      Frequency of the MPC rollout.
            - use_RTI: Use RTI flag.
            - model:   Acados OCP model.
            - solver:  Acados OCP Solver.

            - export_dir:       Directory where the code is exported.
            - solver_path:      Path to the solver json file.
            - simulator_path:   Path to the simulator json file.

        """

        # =====================================================================
        # Extract parameters
        # =====================================================================
        fout_wps = config.get_config("fout_waypoints")
        mpc_prms = config.get_config("mpc_parameters")
        drn_prms = config.get_config("drone_parameters")
        ctl_prms = config.get_config("control_parameters")
        
        # MPC Parameters
        Nhn = mpc_prms["horizon"]
        Qk,Rk,QN = np.diag(mpc_prms["Qk"]),np.diag(mpc_prms["Rk"]),np.diag(mpc_prms["QN"])
        Ws = np.diag(mpc_prms["Ws"])

        # Control Parameters
        hz_ctl= ctl_prms["hz"]
        lbu,ubu = np.array(ctl_prms["bounds"]["lower"]),np.array(ctl_prms["bounds"]["upper"])

        # Derived Parameters
        traj_config_pd = self.pad_trajectory(fout_wps,Nhn,hz_ctl)
        drn_spec = dh.generate_specifications(drn_prms)
        nx,nu = drn_spec["nx"], drn_spec["nu"]

        ny,ny_e = nx+nu,nx
        solver_json = 'acados_ocp_nlp_'+name+'.json'
        
        # =====================================================================
        # Compute Desired Trajectory
        # =====================================================================

        # Solve Padded Trajectory
        output = ms.solve(traj_config_pd)
        if output is not False:
            Tpi, CPi = output
        else:
            raise ValueError("Padded trajectory (for VehicleRateMPC) not feasible. Aborting.")
        
        # Convert to desired tXU
        tXUd = th.TS_to_tXU(Tpi,CPi,drn_spec,hz_ctl)
        # pt.plot_tXU_spatial(tXUd)
        # pt.plot_tXU_time(tXUd)

        # =====================================================================
        # Setup Acados Variables
        # =====================================================================

        # Initialize Acados OCP
        ocp = AcadosOcp()

        ocp.model = export_quadcopter_ode_model(drn_spec["m"],drn_spec["tn"])        
        ocp.model.cost_y_expr = vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr_e = ocp.model.x

        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        ocp.cost.W = scipy.linalg.block_diag(Qk,Rk)
        ocp.cost.W_e = QN
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        ocp.constraints.x0 = tXUd[1:11,0]
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # Initialize Acados Solver
        ocp.solver_options.N_horizon = Nhn
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.sim_method_newton_iter = 10

        if use_RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'

        ocp.solver_options.qp_solver_cond_N = Nhn
        ocp.solver_options.tf = Nhn/hz_ctl
        ocp.solver_options.qp_solver_warm_start = 1

        ocp.code_export_directory = os.path.join(ocp.code_export_directory,name)

        # =====================================================================
        # Controller Variables
        # =====================================================================

        # Base Controller Variables
        super().__init__(name,hz_ctl)

        # Controller Specific Variables
        self.Nx,self.Nu = nx,nu
        self.tXUd = 1.0*tXUd
        self.Qk,self.Rk,self.QN = Qk,Rk,QN
        self.Ws = Ws
        self.lbu,self.ubu = lbu,ubu
        self.ns = int(hz_ctl/5)
        self.use_RTI = use_RTI
        self.model = ocp.model
        self.solver = AcadosOcpSolver(ocp,json_file=solver_json,verbose=False)

        self.code_export_path = ocp.code_export_directory
        self.solver_path = os.path.join(os.getcwd(),solver_json)

        # =====================================================================
        # Warm start the solver
        # =====================================================================
        
        for _ in range(5):
            self.control(0.0,tXUd[1:11,0])
            
    def control(self,
                tcr:float,xcr:np.ndarray,
                upr:np.ndarray=None,
                obj:np.ndarray=None,
                icr:None=None,zcr:None=None) -> Tuple[
                    np.ndarray,None,None,np.ndarray]:
        
        """
        Method to compute the control input for the VehicleRateMPC controller. We use the standard input arguments
        format with the unused arguments set to None. Likewise, we use the standard output format with the unused
        outputs set to None.

        Args:
        tcr:    Current time.
        xcr:    Current state.
        upr:    Previous control input (unused).
        obj:    Objective (unused).
        icr:    Current image (unused).
        zcr:    Current image feature vector (unused).

        Returns:
        ucc:    Control input.
        None:   Output feature vector (unused).
        None:   Oracle output  (unused).
        tsol:   Time taken to solve components [setup ocp, solve ocp, unused, unused].

        """
        # Unused arguments
        _ = upr,obj,icr,zcr

        # Start timer
        t0 = time.time()

        # Get desired trajectory
        ydes = self.get_ydes(xcr,tcr)

        # Set desired trajectory
        for i in range(self.solver.acados_ocp.dims.N):
            self.solver.cost_set(i, "yref", ydes[:,i])
        self.solver.cost_set(self.solver.acados_ocp.dims.N, "yref", ydes[0:10,-1])

        # Solve OCP
        t1 = time.time()
        if self.use_RTI:
            # preparation phase
            self.solver.options_set('rti_phase', 1)
            status = self.solver.solve()

            # set initial state
            self.solver.set(0, "lbx", xcr)
            self.solver.set(0, "ubx", xcr)

            # feedback phase
            self.solver.options_set('rti_phase', 2)
            status = self.solver.solve()

            ucc = self.solver.get(0, "u")
        else:
            # Solve ocp and get next control input
            try:
                ucc = self.solver.solve_for_x0(x0_bar=xcr)
            except:
                print("Warning: VehicleRateMPC failed to solve OCP. Using previous input.")
                ucc = self.solver.get(0, "u")
        t2 = time.time()

        # Compute timer values
        tsol = np.array([t1-t0,t2-t1,0.0,0.0])

        return ucc,None,None,tsol

    def pad_trajectory(self,fout_wps:Dict[str,Union[str,int,Dict[str,Union[float,np.ndarray]]]],
                       Nhn:int,hz_ctl:float) -> Dict[str,Dict[str,Union[float,np.ndarray]]]:
        """
        Method to pad the trajectory with the final waypoint so that the MPC horizon is satisfied at the end of the trajectory.

        Args:
        fout_wps:   Dictionary containing the flat output waypoints.
        Nhn:        Prediction horizon.
        hz_ctl:     Controller frequency.

        Returns:
        fout_wps_pd: Padded flat output waypoints.

        """

        # Get final waypoint
        kff = list(fout_wps["keyframes"])[-1]
        
        # Pad trajectory
        t_pd = fout_wps["keyframes"][kff]["t"]+(Nhn/hz_ctl)
        fo_pd = np.array(fout_wps["keyframes"][kff]["fo"])[:,0:3].tolist()

        fout_wps_pd = deepcopy(fout_wps)
        fout_wps_pd["keyframes"]["fof"] = {
            "t":t_pd,
            "fo":fo_pd}

        return fout_wps_pd

    def get_ydes(self,xcr:np.ndarray,ti:float) -> np.ndarray:
        """
        Method to get the section of the desired trajectory at the current time.

        Args:
        xcr:    Current state.
        ti:     Current time.

        Returns:
        ydes:   Desired trajectory section at the current time.

        """
        # Get relevant portion of trajectory
        idx_i = int(self.hz*ti)
        Nhn_lim = self.tXUd.shape[1]-self.solver.acados_ocp.dims.N-1
        ks0 = np.clip(idx_i-self.ns,0,Nhn_lim-1)
        ksf = np.clip(idx_i+self.ns,0,Nhn_lim)
        Xi = self.tXUd[1:11,ks0:ksf]
        
        # Find index of nearest state
        dXi = Xi-xcr.reshape(-1,1)
        wl2_dXi = np.array([x.T@self.Ws@x for x in dXi.T])
        idx0 = ks0 + np.argmin(wl2_dXi)
        idxf = idx0 + self.solver.acados_ocp.dims.N+1

        # Pad if idxf is greater than the last index
        if idxf < self.tXUd.shape[1]:
            ydes = self.tXUd[1:,idx0:idxf]
        else:
            print("Warning: VehicleRateMPC.get_ydes() padding trajectory. Increase your padding horizon.")
            ydes = self.tXUd[1:,idx0:]
            ydes = np.hstack((ydes,np.tile(ydes[:,-1:],(1,idxf-self.tXUd.shape[1]))))

        return ydes
    
    def generate_simulator(self,hz):
        sim = AcadosSim()
        sim.model = self.model
        sim.solver_options.T = 1/hz
        sim.solver_options.integrator_type = 'IRK'

        sim_json = 'acados_sim_nlp.json'
        self.sim_json_file = os.path.join(os.path.dirname(self.code_export_path),sim_json)

        return AcadosSimSolver(sim,json_file=sim_json,verbose=False)
    def clear_generated_code(self):
        """
        Method to clear the generated code and files to ensure the code is recompiled correctly each time.
        """

        try:
            os.remove(self.solver_path)
            shutil.rmtree(self.code_export_path)
        except:
            pass

        # Clear the parent directory if empty
        parent_dir_path = os.path.dirname(self.code_export_path)
        if not os.listdir(parent_dir_path) and (os.path.basename(parent_dir_path) == 'c_generated_code'):
            shutil.rmtree(parent_dir_path)