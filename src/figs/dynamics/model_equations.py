from acados_template import AcadosModel
import casadi as ca

def export_quadcopter_ode_model(m:float,tn:float) -> AcadosModel:

    model_name = 'quadcopter_ode_model'

    # set up states
    px = ca.ca.SX.sym('px')
    py = ca.ca.SX.sym('py')
    pz = ca.ca.SX.sym('pz')
    p = ca.ca.vertcat(px,py,pz)

    vx = ca.ca.SX.sym('vx')
    vy = ca.SX.sym('vy')
    vz = ca.SX.sym('vz')
    v = ca.vertcat(vx,vy,vz)

    qx = ca.SX.sym('qx')
    qy = ca.SX.sym('qy')
    qz = ca.SX.sym('qz')
    qw = ca.SX.sym('qw')
    q = ca.vertcat(qx,qy,qz,qw)

    x = ca.vertcat(p,v,q)

    # set up controls
    uf = ca.SX.sym('uf')
    wx = ca.SX.sym('wx')
    wy = ca.SX.sym('wy')
    wz = ca.SX.sym('wz')
    uw = ca.vertcat(wx,wy,wz)

    u = ca.vertcat(uf,uw)

    # xdot
    px_dot = ca.SX.sym('px_dot')
    py_dot = ca.SX.sym('py_dot')
    pz_dot = ca.SX.sym('pz_dot')
    p_dot = ca.vertcat(px_dot,py_dot,pz_dot)

    vx_dot = ca.SX.sym('vx_dot')
    vy_dot = ca.SX.sym('vy_dot')
    vz_dot = ca.SX.sym('vz_dot')
    v_dot = ca.vertcat(vx_dot,vy_dot,vz_dot)

    qx_dot = ca.SX.sym('qx_dot')
    qy_dot = ca.SX.sym('qy_dot')
    qz_dot = ca.SX.sym('qz_dot')
    qw_dot = ca.SX.sym('qw_dot')
    q_dot = ca.vertcat(qx_dot,qy_dot,qz_dot,qw_dot)

    xdot = ca.vertcat(p_dot,v_dot,q_dot)

    # some intermediate variables
    V1a = ca.vertcat(0.0, 0.0, 9.81)
    V1b = (tn*uf/m)*ca.vertcat(
          2.0*(qx*qz + qy*qw),
          2.0*(qy*qz - qx*qw),
          qw*qw - qx*qx - qy*qy + qz*qz)  
    V2 = (1/2)*ca.vertcat(
         qw*wx - qz*wy + qy*wz,
         qz*wx + qw*wy - qx*wz,
        -qy*wx + qx*wy + qw*wz,
        -qx*wx - qy*wy - qz*wz)

    # dynamics
    f_expl = ca.vertcat(
                v,
                V1a + V1b,
                V2
                )
    
    f_impl = xdot-f_expl

    # Pack into acados model
    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model




def export_glider_ode_model(params):
    """
    Builds an AcadosModel for the point-mass glider with wind.
    params: dict with keys ['m','S','rho','g','C_L','C_D','wind']
    """
    model = AcadosModel()
    model.name = 'glider_time_opt'

    # States
    x     = ca.SX.sym('x')
    y     = ca.SX.sym('y')
    h     = ca.SX.sym('h')
    V     = ca.SX.sym('V')
    gamma = ca.SX.sym('gamma')
    psi   = ca.SX.sym('psi')
    tau   = ca.SX.sym('tau')  # clock state
    model.x = ca.vertcat(x, y, h, V, gamma, psi, tau)

    # Controls
    alpha = ca.SX.sym('alpha')
    phi   = ca.SX.sym('phi')
    beta  = ca.SX.sym('beta')
    model.u = ca.vertcat(alpha, phi, beta)

    # Wind
    wind = params['wind']
    Wx = wind(h)['Wx']
    Wy = wind(h)['Wy']
    Wz = wind(h)['Wz']

    # Aerodynamics
    rho = params['rho']; S = params['S']; m = params['m']; g = params['g']
    C_L = params['C_L']; C_D = params['C_D']
    L = 0.5 * rho * V**2 * S * C_L(alpha)
    D = 0.5 * rho * V**2 * S * C_D(alpha)

    # Equations of motion
    xdot = ca.vertcat(
        V*ca.cos(gamma)*ca.cos(psi) + Wx,
        V*ca.cos(gamma)*ca.sin(psi) + Wy,
        V*ca.sin(gamma)           + Wz,
        (-D)/m - g*ca.sin(gamma),
        (L*ca.cos(phi))/(m*V) - (g*ca.cos(gamma))/V,
        (L*ca.sin(phi))/(m*V*ca.cos(gamma)),
        1  # time derivative of clock state
    )
    model.f_expl_expr = xdot
    return model
