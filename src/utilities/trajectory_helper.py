"""
Helper functions for trajectory data.
"""

import numpy as np
import math

from scipy.spatial.transform import Rotation
from typing import Dict,Tuple,Union

def fo_to_xu(fo:np.ndarray,quad:Dict[str,Union[float,np.ndarray]])  -> np.ndarray:
    """
    Converts a flat output vector to a state vector and body-rate command.

    Args:
        - fo:     Flat output array.
        - quad:   Quadcopter specifications.

    Returns:
        - xu:    State vector and control input.
    """

    # Unpack
    pt = fo[0:3,0]
    vt = fo[0:3,1]
    at = fo[0:3,2]
    jt = fo[0:3,3]

    psit  = fo[3,0]
    psidt = fo[3,1]

    m,tn = quad["m"],quad["tn"]

    # Compute Gravity
    gt = np.array([0.00,0.00,-9.81])

    # Compute Thrust
    alpha:np.ndarray = at+gt

    # Compute Intermediate Frame xy
    xct = np.array([ np.cos(psit), np.sin(psit), 0.0 ])
    yct = np.array([-np.sin(psit), np.cos(psit), 0.0 ])
    
    # Compute Orientation
    xbt = np.cross(alpha,yct)/np.linalg.norm(np.cross(alpha,yct))
    ybt = np.cross(xbt,alpha)/np.linalg.norm(np.cross(xbt,alpha))
    zbt = np.cross(xbt,ybt)
    
    Rt = np.hstack((xbt.reshape(3,1), ybt.reshape(3,1), zbt.reshape(3,1)))
    qt = Rotation.from_matrix(Rt).as_quat()

    # Compute Thrust Variables
    c = zbt.T@alpha

    # Compute Angular Velocity
    B1 = c
    D1 = xbt.T@jt
    A2 = c
    D2 = -ybt.T@jt
    B3 = -yct.T@zbt
    C3 = np.linalg.norm(np.cross(yct,zbt))
    D3 = psidt*(xct.T@xbt)

    wxt = (B1*C3*D2)/(A2*(B1*C3))
    wyt = (C3*D1)/(B1*C3)
    wzt = ((B1*D3)-(B3*D1))/(B1*C3)

    wt = np.array([wxt,wyt,wzt])

    # Compute Body-Rate Command
    ut = np.hstack((m*c/tn,wt))
    
    # Stack
    xu = np.hstack((pt,vt,qt,ut))

    return xu

def ts_to_fo(tcr:float,Tp:float,CP:np.ndarray) -> np.ndarray:
    """
    Converts a trajectory spline (defined by Tp,CP) to a flat output.

    Args:
        - tcr: Current time.
        - Tp:  Trajectory segment final time.
        - CP:  Control points.

    Returns:
        - fo:  Flat output vector.
    """
    Ncp = CP.shape[1]
    M = get_M(Ncp)

    fo = np.zeros((4,Ncp))
    for i in range(0,Ncp):
        nt = get_nt(tcr,Tp,i,Ncp)
        fo[:,i] = (CP @ M @ nt) / (Tp**i)

    return fo

def ts_to_xu(tcr:float,Tp:float,CP:np.ndarray,
             quad:Dict[str,Union[float,np.ndarray]]) -> np.ndarray:
    """
    Converts a trajectory spline (defined by tf,CP) to a state vector and control input.

    Args:
        tcr:  Current segment time.
        Tp:   Trajectory segment final time.
        CP:   Control points.
        quad: Quadcopter specifications.

    Returns:
        xu:    State vector and control input.
    """
    fo = ts_to_fo(tcr,Tp,CP)
    return fo_to_xu(fo,quad)

def TS_to_tXU(Tps:np.ndarray,CPs:np.ndarray,
              quad:Union[None,Dict[str,Union[float,np.ndarray]]],
              hz:int) -> np.ndarray:
    """
    Converts a sequence of trajectory splines (defined by Tps,CPs) to a trajectory
    rollout.

    Args:
        - Tps:  Trajectory segment times.
        - CPs:  Trajectory control points.
        - quad: Quadcopter specifications.
        - hz:   Control loop frequency.

    Returns:
        - tXU:  State vector and control input rollout.
    """
    Nt = int((Tps[-1]-Tps[0])*hz+1)

    idx = 0
    for k in range(Nt):
        tk = Tps[0]+k/hz

        if tk > Tps[idx+1] and idx < len(Tps)-2:
            idx += 1

        t0,tf = Tps[idx],Tps[idx+1]
        CPk = CPs[idx,:,:]
        xu = ts_to_xu(tk-t0,tf-t0,CPk,quad)

        if k == 0:
            ntxu = len(xu)+1
            tXU = np.zeros((ntxu,Nt))
        else:
            xu[6:10] = obedient_quaternion(xu[6:10],tXU[7:11,k-1])
                
        tXU[0,k] = tk
        tXU[1:,k] = xu

    return tXU

def get_nt(tk:float,tf:float,kd:int,Ncp:int) -> np.ndarray:  
    """
    Generates the normalized time vector based on desired derivative order.

    Args:
        - tk:     Current time on segment.
        - tf:     Segment final time.
        - kd:     Derivative order.
        - Ncp:    Number of control points.

    Returns:
        - nt:      the normalized time vector.
    """

    tn = tk/tf

    nt = np.zeros(Ncp)
    for i in range(kd,Ncp):
        c = math.factorial(i)/math.factorial(i-kd)
        nt[i] = c*tn**(i-kd)
    
    return nt

def get_M(Ncp:int) -> np.ndarray:
    """
    Generates the M matrix for polynomial interpolation.

    Args:
        - Ncp:    Number of control points.

    Returns:
        - M:      Polynomial interpolation matrix.
    """
    M = np.zeros((Ncp,Ncp))
    for i in range(Ncp):
        ci = (1/(Ncp-1))*i
        for j in range(Ncp):
            M[i,j] = ci**j
    M = np.linalg.inv(M).T

    return M

def obedient_quaternion(qcr:np.ndarray,qrf:np.ndarray) -> np.ndarray:
    """
    Ensure that the quaternion is well-behaved (unit norm and closest to reference).
    
    Args:
        - qcr:    Current quaternion.
        - qrf:    Previous quaternion.

    Returns:
        - qcr:     Closest quaternion to reference.
    """
    qcr = qcr/np.linalg.norm(qcr)

    if np.dot(qcr,qrf) < 0:
        qcr = -qcr

    return qcr

def xv_to_T(xcr:np.ndarray) -> np.ndarray:
    """
    Converts a state vector to a transfrom matrix.

    Args:
        - xcr:    State vector.

    Returns:
        - Tcr:    Pose matrix.
    """
    Tcr = np.eye(4)
    Tcr[0:3,0:3] = Rotation.from_quat(xcr[6:10]).as_matrix()
    Tcr[0:3,3] = xcr[0:3]

    return Tcr

def RO_to_tXU(RO:Tuple[np.ndarray,np.ndarray,np.ndarray]) -> np.ndarray:
    """
    Converts a tuple of rollouts to a state vector and control input rollout.

    Args:
        - RO:    Rollout tuple (Tro,Xro,Uro).

    Returns:
        - tXU:   State vector and control input rollout.
    """
    # Unpack the tuple
    Tro,Xro,Uro = RO

    # Stack the arrays
    Uro = np.hstack((Uro,Uro[:,-1].reshape(-1,1)))
    tXU = np.vstack((Tro,Xro,Uro))

    return tXU