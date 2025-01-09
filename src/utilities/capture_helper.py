"""
Helper functions for capture generation.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List,Union

def distribute_values(values,num_picks):
    values = np.array(values)
    selected_values = [values[0]]  # Start with the first element
    
    for _ in range(num_picks - 1):
        remaining = [x for x in values if x not in selected_values]
        max_dist = -1
        best_candidate = None
        
        for candidate in remaining:
            # Compute the minimum distance of this candidate to the already selected
            min_dist = min(abs(candidate - s) for s in selected_values)
            if min_dist > max_dist:
                max_dist = min_dist
                best_candidate = candidate
        
        selected_values.append(best_candidate)

    selected_values.sort()

    return selected_values


def abs_orientation(X: torch.Tensor, Y: torch.Tensor):
    """
    Determine the optimal transformation that brings points from
    X's reference frame to points in Y's.
    T(x) = c * Rx + t where x is a point 3x1, c is the scaling,
    R is a 3x3 rotation matrix, and t is a 3x1 translation.

    This is based off of "Least-Squares Estimation of Transformation
    Parameters Between Two Point Patterns" Umeyama 1991.

    Args:
        X - Tensor with dimension N x m
        Y - Tensor with dimension N x m
    returns:
        c - Scalar scaling constant
        R - Tensor 3x3 rotation matrix
        t - Tensor 3
    """

    N, m = X.shape

    mux = torch.mean(X, 0, True)
    muy = torch.mean(Y, 0, True)

    Yd = (Y - muy).unsqueeze(-1)
    Xd = (X - mux).unsqueeze(1)
    sx = torch.sum(torch.norm(Xd.squeeze(), dim=1) ** 2) / N
    Sxy = (1 / N) * torch.sum(torch.matmul(Yd, Xd), dim=0)

    if torch.linalg.matrix_rank(Sxy) < m:
        raise NameError("Absolute orientation transformation does not exist!")

    U, D, Vt = torch.linalg.svd(Sxy, full_matrices=True)
    S = torch.eye(m).to(dtype=Vt.dtype)
    if torch.linalg.det(Sxy) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    c = torch.trace(torch.diag(D) @ S) / sx
    t = muy.T - c * (R @ mux.T)

    return c, R, t.squeeze()
