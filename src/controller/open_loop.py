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
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import vertcat
from dynamics.quadcopter_model import export_quadcopter_ode_model
from typing import Union,Tuple,Dict
from copy import deepcopy
import visualize.plot_trajectories as pt

class OpenLoop(BaseController):
    def __init__(self) -> None:
        

        # Base Controller Variables
        super().__init__('haha',20)

        self.k = 0
        self.U = np.zeros((4,1000))
        self.U[0,:] = -0.7
        self.U[2,0:20] =  1.57

    def control(self,tcr,xcr,upr,obj,icr,zcr):
        _ = tcr,xcr,upr,obj,icr,zcr

        uk = self.U[:,self.k]
        self.k += 1

        return uk,None,None,None
