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


def compute_ransac_transform(W1:np.ndarray, W2:np.ndarray,
                             n_batch:int=3,threshold:float=5e-2, max_iterations:int=1000) -> np.ndarray:
    """
    Compute the RANSAC transformation between two sets of 3D points.

    Args:
        W1:             First set of 3D points (3 x N)
        W2:             Second set of 3D points (3 x N)
        threshold:      Inlier threshold
        max_iterations: Maximum number of iterations

    Returns:
        Trsc:           Transformation matrix (4 x 4)
    """

    assert W1.shape == W2.shape, "Input arrays must have the same shape"
    assert W1.shape[0] == 3, "Input arrays must have 3 rows (3D points)"

    N = W1.shape[1]
    best_inliers = 0
    best_rotation = None
    best_translation = None
    best_scale = None

    for _ in range(max_iterations):
        # Randomly sample 3 points
        idx = np.random.choice(N, n_batch, replace=False)
        W1_sample = W1[:, idx]
        W2_sample = W2[:, idx]

        # Estimate transformation (Rigid transformation: rotation + translation + scaling)
        # Compute centroids
        centroid_W1 = np.mean(W1_sample, axis=1, keepdims=True)
        centroid_W2 = np.mean(W2_sample, axis=1, keepdims=True)

        # Center the points
        W1_centered = W1_sample - centroid_W1
        W2_centered = W2_sample - centroid_W2

        # Compute scaling factor
        scale_est = np.linalg.norm(W2_centered) / np.linalg.norm(W1_centered)

        # Apply scaling to W1
        W1_centered_scaled = W1_centered * scale_est

        # Compute cross-covariance matrix
        H = W1_centered_scaled @ W2_centered.T

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(H)
        R_est = Vt.T @ U.T

        # Ensure a proper rotation matrix (det(R) = 1)
        if np.linalg.det(R_est) < 0:
            Vt[2, :] *= -1
            R_est = Vt.T @ U.T

        t_est = centroid_W2.flatten() - scale_est * R_est @ centroid_W1.flatten()

        # Count inliers
        W1_transformed = scale_est * R_est @ W1 + t_est[:, np.newaxis]
        distances = np.linalg.norm(W2 - W1_transformed, axis=0)
        inliers = np.sum(distances < threshold)

        # Update best transformation if current one is better
        if inliers > best_inliers:
            best_inliers = inliers
            best_rotation = R_est
            best_translation = t_est
            best_scale = scale_est

    # Compute the final transformation matrix
    Trsc = np.eye(4)
    Trsc[:3, :3] = best_scale * best_rotation
    Trsc[:3, 3] = best_translation

    # Compute some statistics
    total_cost = 0.0
    for i in range(N):
        W1_transformed = best_scale * best_rotation @ W1[:, i] + best_translation
        cost = np.linalg.norm(W2[:, i] - W1_transformed)
        total_cost += cost

    print(f"RANSAC: {best_inliers} inliers out of {N} points with threshold {threshold}m")

    return Trsc
