import numpy as np
import casadi as ca
import pytest

from figs.dynamics.model_equations import export_glider_ode_model
from figs.dynamics.model_specifications import generate_glider_specifications

def numerical_jacobian(f, x, u, eps=1e-6):
    """
    Compute numerical Jacobian df/dx by finite differences.
    f: function f(x, u) returning a vector (can be CasADi DM or NumPy array)
    x: state vector (nx,)
    u: control vector (nu,)
    """
    f0 = np.array(f(x, u)).flatten()
    nx = x.size
    J = np.zeros((f0.size, nx))
    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps
        f1 = np.array(f(x + dx, u)).flatten()
        J[:, i] = (f1 - f0) / eps
    return J

@pytest.fixture
def glider_model_and_func():
    # Example parameters
    drn_prms = {
        'mass': 1.0,
        'wing_area': 1.0,
        'air_density': 1.225,
        'gravity': 9.81,
        'a1': 5.7,
        'a0': 0.1,
        'b2': 0.02,
        'b0': 0.0,
        'alpha_min': -0.5,
        'alpha_max': 0.5,
        'phi_min': -np.pi/4,
        'phi_max': np.pi/4,
        'beta_min': 0.0,
        'beta_max': 0.0
    }
    specs = generate_glider_specifications(drn_prms)
    model = export_glider_ode_model(specs)
    # Create CasADi function for f_expl_expr
    f_casadi = ca.Function('f', [model.x, model.u], [model.f_expl_expr])
    # Create Jacobian function df/dx
    J_casadi = ca.Function('Jf', [model.x, model.u], [ca.jacobian(model.f_expl_expr, model.x)])
    return specs, model, f_casadi, J_casadi

def test_zero_wind_kinematics(glider_model_and_func):
    """
    With zero wind, the kinematic components should match the expected cos/sin relations.
    """
    specs, _, f, _ = glider_model_and_func
    # Override wind to zero
    specs['wind'] = lambda h: {'Wx': 0.0, 'Wy': 0.0, 'Wz': 0.0}
    model = export_glider_ode_model(specs)
    f0 = ca.Function('f0', [model.x, model.u], [model.f_expl_expr])
    # Define a test state/control
    x = np.array([10.0, 20.0, 100.0, 15.0, np.pi/6, np.pi/4, 0.0])
    u = np.array([0.1, 0.2, 0.0])
    xdot_dm = f0(x, u)
    xdot = np.array(xdot_dm).flatten()
    V     = x[3]; gamma = x[4]; psi = x[5]
    # Check kinematic parts
    assert np.isclose(xdot[0], V * np.cos(gamma) * np.cos(psi), atol=1e-6)
    assert np.isclose(xdot[1], V * np.cos(gamma) * np.sin(psi), atol=1e-6)
    assert np.isclose(xdot[2], V * np.sin(gamma),                 atol=1e-6)

def test_jacobian_matches_finite_difference(glider_model_and_func):
    """
    Compare CasADi Jacobian to numerical finite differences.
    """
    _, model, f, J_casadi_f = glider_model_and_func
    # Use original wind profile
    x = np.array([5.0, -3.0, 50.0, 12.0, 0.3, -0.1, 0.0])
    u = np.array([0.05, -0.05, 0.0])
    # Evaluate CasADi Jacobian
    J_casadi = np.array(J_casadi_f(x, u)).squeeze()
    # Evaluate numerical Jacobian (convert DM→np internally)
    J_num = numerical_jacobian(lambda xx, uu: np.array(f(xx, uu)), x, u)
    # Compare
    assert np.allclose(J_casadi, J_num, atol=1e-4, rtol=1e-4)

if __name__ == "__main__":
    '''
    This is a test script to verify the glider ODE model and its Jacobian.
    We compare CasADi's analytical Jacobian with a numerical approximation.

    Additionally, we check that the kinematic equations behave as expected
    when the wind is set to zero.
    '''
    pytest.main()

