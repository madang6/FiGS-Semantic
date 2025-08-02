from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as ca

from figs.control.base_controller import BaseController
from figs.dynamics.model_equations import export_glider_ode_model
from figs.dynamics.model_specifications import generate_glider_specifications

def make_time_opt_waypoint_ocp(drn_prms, waypoint, N=40):
    # 1) Build model
    specs = generate_glider_specifications(drn_prms)
    model = export_glider_ode_model(specs)

    # 2) Create OCP object
    ocp = AcadosOcp()
    ocp.model = model

    # 3) Dimensions
    ocp.dims.nx = model.x.size()[0]      # = 7 (x,y,h,V,γ,ψ,τ)
    ocp.dims.nu = model.u.size()[0]      # = 3 (α,ϕ,β)
    ocp.dims.N  = N                      # number of shooting intervals

    # 4) Cost: Mayer only on τ
    #    weight only last state τ: W_e = diag([0,0,0,0,0,0, 1])
    W_e = ca.diag([0,0,0,0,0,0, 1])
    ocp.cost.cost_type            = 'ONLY_MAYER'
    ocp.cost.cost_type_mayer      = 'MAYER'
    ocp.cost.W_e                  = W_e

    # 5) Solver options
    #    - free final time: let tau drive tf
    #    - choose an explicit integrator (ERK) per docs
    ocp.solver_options.tf               = None
    ocp.solver_options.integrator_type  = 'ERK'
    ocp.solver_options.nlp_solver_type  = 'SQP'

    # 6) Path constraints on controls
    ocp.constraints.lbu   = [
        drn_prms['alpha_min'],
        drn_prms['phi_min'],
        drn_prms['beta_min']
    ]
    ocp.constraints.ubu   = [
        drn_prms['alpha_max'],
        drn_prms['phi_max'],
        drn_prms['beta_max']
    ]
    ocp.constraints.idxbu = [0,1,2]

    # 7) Initial‐state constraints
    #    Fix x[0] = current state, τ(0)=0
    ocp.constraints.x0 = [
        *drn_prms['x0'],  # unpack x0 = [x,y,h,V,γ,ψ]
         0                # τ(0)=0
    ]

    # 8) Terminal waypoint constraint on x,y,h
    ocp.constraints.idxbx_e = [0,1,2]  # only constrain x,y,h at final
    ocp.constraints.lbx_e   = [
        waypoint[0], waypoint[1], waypoint[2]
    ]
    ocp.constraints.ubx_e   = [
        waypoint[0], waypoint[1], waypoint[2]
    ]

    # 9) Build solver
    solver = AcadosOcpSolver(ocp, json_file='acados_time_opt_waypt.json')
    return solver
