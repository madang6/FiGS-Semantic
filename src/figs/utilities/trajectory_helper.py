"""
Helper functions for trajectory data.
"""

import numpy as np
import math

from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline
from sklearn.cluster import DBSCAN
from typing import Dict,Tuple,Union

from figs.tsampling.rrt_datagen_v10 import *

def filter_branches(paths, hover_mode=False):
    """
    Filters the branches of paths based on hover_mode and adjacency constraints.

    Parameters:
        paths (list of list of np.ndarray): The original paths for an object.
        hover_mode (bool): Whether to apply hover mode filtering. Default False.

    Returns:
        list of list of np.ndarray: The filtered branches.
    """
    skip_nodes = 2
    new_branches = []
    for idbr, positions in enumerate(paths):
        positions = np.array(positions)

        if hover_mode:
            radius = 1.5
            filtered_positions = [positions[0]]
            for pos in positions[1:]:
                if np.linalg.norm(pos - filtered_positions[0]) <= radius:
                    filtered_positions.append(pos)
            positions = np.array(filtered_positions)

            if positions.shape[0] < skip_nodes:
                print(f"Branch {idbr} has less than {skip_nodes} nodes. Skipping.")
                continue
        else:
            if positions.shape[0] < skip_nodes:
                print(f"Branch {idbr} has less than {skip_nodes} nodes. Skipping.")
                continue

        # Filter adjacent positions within 0.25
        filtered_positions = [positions[0]]
        for pos in positions[1:]:
            if np.linalg.norm(pos - filtered_positions[-1]) >= 0.25:
                filtered_positions.append(pos)

        new_branches.append(np.array(filtered_positions))

    return new_branches

def set_RRT_altitude(paths, goal_z):
    """
    Adds the altitude goal_z to each point in the given paths.
    
    Parameters:
        paths (list of list of np.ndarray): The original paths for an object.
        goal_z (float): The altitude to append to each point.
        
    Returns:
        list of list of np.ndarray: The updated paths with altitude added.
    """
    updated_paths = []
    for path in paths:
        updated_path = []
        for point in path:
            updated_path.append(np.append(point, goal_z))  # Append updated point
        updated_paths.append(updated_path)

    return updated_paths

def process_RRT_objectives(obj_targets, epcds_arr, env_bounds, radii, altitudes, hoverMode=False):
    """
    Process the object targets and return updated obj_targets and obj_centroid,
    with detailed printouts of free_pts extents and bound-rejection reasons.
    """
    new_obj_targets = obj_targets.copy()
    object_centroid = []
    
    for i, obj in enumerate(obj_targets):
        print(f"\n--- Processing object {i} ---")
        r1, r2 = radii[i]
        objctr = obj.flatten()
        print(f" Centroid: {objctr}, r1={r1:.3f}, r2={r2:.3f}")
        object_centroid.append(objctr)

        # Sample circle around centroid
        theta = np.linspace(0, 2*np.pi, 100)
        z_vals = np.full_like(theta, altitudes[i])
        circle_pts = np.vstack([
            objctr[0] + r1*np.cos(theta),
            objctr[1] + r1*np.sin(theta),
            z_vals
        ]).T

        # Find free points (no obstacle within r2)
        kdtree = cKDTree(epcds_arr.T)
        free_pts = [p for p in circle_pts
                    if not kdtree.query_ball_point(p, r2, eps=0.05, workers=-1)]
        print(f" Found {len(free_pts)} free_pts (no obstacles within r2)")

        if not free_pts:
            print("  → No free_pts found. Try reducing r2 or r1.")
            continue

        # Compute and print extents of free_pts
        arr = np.array(free_pts)
        x_min, x_max = arr[:,0].min(), arr[:,0].max()
        y_min, y_max = arr[:,1].min(), arr[:,1].max()
        z_min, z_max = arr[:,2].min(), arr[:,2].max()
        print(f"  free_pts extents:")
        print(f"    x ∈ [{x_min:.3f}, {x_max:.3f}], y ∈ [{y_min:.3f}, {y_max:.3f}], z ∈ [{z_min:.3f}, {z_max:.3f}]")

        # Print environment bounds
        minb = env_bounds["minbound"]
        maxb = env_bounds["maxbound"]
        print(f"  env_bounds:")
        print(f"    x ∈ [{minb[0]:.3f}, {maxb[0]:.3f}], "
              f"y ∈ [{minb[1]:.3f}, {maxb[1]:.3f}], "
              f"z ∈ [{minb[2]:.3f}, {maxb[2]:.3f}]")

        # Count why points are out of bounds
        x_low  = np.sum(arr[:,0] < minb[0])
        x_high = np.sum(arr[:,0] > maxb[0])
        y_low  = np.sum(arr[:,1] < minb[1])
        y_high = np.sum(arr[:,1] > maxb[1])
        z_low  = np.sum(arr[:,2] < minb[2])
        z_high = np.sum(arr[:,2] > maxb[2])
        total_out = x_low + x_high + y_low + y_high + z_low + z_high
        if total_out > 0:
            print("  Rejection summary (out-of-bounds counts):")
            if x_low:  print(f"    {x_low} pts with x < {minb[0]:.3f}")
            if x_high: print(f"    {x_high} pts with x > {maxb[0]:.3f}")
            if y_low:  print(f"    {y_low} pts with y < {minb[1]:.3f}")
            if y_high: print(f"    {y_high} pts with y > {maxb[1]:.3f}")
            if z_low:  print(f"    {z_low} pts with z < {minb[2]:.3f}")
            if z_high: print(f"    {z_high} pts with z > {maxb[2]:.3f}")
        else:
            print("  No free_pts are out-of-bounds in any axis.")

        # Filter to in-bounds
        in_bounds = [
            p for p in free_pts
            if (minb[0] <= p[0] <= maxb[0] and
                minb[1] <= p[1] <= maxb[1] and
                minb[2] <= p[2] <= maxb[2])
        ]
        print(f" {len(in_bounds)} points remain inside env bounds")

        if not in_bounds:
            print("  → All free_pts were out-of-bounds; consider:")
            print("     * increasing env_bounds or")
            print("     * decreasing r1 to sample closer to centroid\n")
            continue

        # Choose the candidate closest to the scene origin 
        new_pos = min(in_bounds, key=lambda p: np.linalg.norm(p))
        print(f"  → Selected new target: {new_pos}")

        new_obj_targets[i] = new_pos

    print("\nFinished processing all objects.")
    return new_obj_targets, object_centroid

def process_RRT_objectives_loiter(
    obj_targets,
    epcds_arr,
    env_bounds,
    radii,
    altitudes,
    sample_size=10,
    hoverMode=False
):
    """
    For each object:
      - compute its centroid
      - sample points on the circle of radius r1 around that centroid
      - reject points that are too close to obstacles (within r2) or out of env_bounds
      - return up to `sample_size` of those valid points
    
    Returns:
      sampled_obj_targets: list of arrays, each of shape (<=sample_size, 3)
      object_centroids:    list of 1×3 arrays
    """
    object_centroids = []
    sampled_obj_targets = []

    # build once
    kdtree = cKDTree(epcds_arr.T)
    minb, maxb = env_bounds["minbound"], env_bounds["maxbound"]

    for i, obj in enumerate(obj_targets):
        objctr = obj.flatten()
        object_centroids.append(objctr)

        r1, r2 = radii[i]
        z_val = altitudes[i]

        # 1) generate circle
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        circle_pts = np.vstack([
            objctr[0] + r1*np.cos(theta),
            objctr[1] + r1*np.sin(theta),
            np.full_like(theta, z_val)
        ]).T

        # 2) obstacle check
        free = []
        for p in circle_pts:
            if not kdtree.query_ball_point(p, r2, eps=0.05, workers=-1):
                free.append(p)
        free = np.array(free)
        print(f"Object {i}: {len(free)} free circle points")

        # 3) bounds check
        in_bounds = free[
            (free[:,0] >= minb[0]) & (free[:,0] <= maxb[0]) &
            (free[:,1] >= minb[1]) & (free[:,1] <= maxb[1]) &
            (free[:,2] >= minb[2]) & (free[:,2] <= maxb[2])
        ]
        print(f"Object {i}: {len(in_bounds)} points inside env bounds")

        # 4) sample up to sample_size
        if len(in_bounds) == 0:
            sampled = np.empty((0,3))
        elif len(in_bounds) <= sample_size:
            sampled = in_bounds
        else:
            idx = np.random.choice(len(in_bounds), size=sample_size, replace=False)
            sampled = in_bounds[idx]

        print(f"Object {i}: returning {sampled.shape[0]} samples\n")
        sampled_obj_targets.append(sampled)

    return sampled_obj_targets, object_centroids

def process_obstacle_clusters_and_sample(
    epcds_arr,          # 3×M array of obstacle points
    env_bounds,         # {"minbound": (x,y,z), "maxbound": (x,y,z)}
    z_range=(-2.5, -0.9), # only cluster points with z in [0.9, 2.0]
    cluster_eps=0.5,    # DBSCAN ε (meters) for clustering obstacles
    min_samples=10,     # DBSCAN min pts per cluster
    clearance=0.2,      # extra clearance (m) beyond cluster extent
    sample_size=10      # how many points on the circle
):
    """
    1) Filter obstacle points inside env_bounds
    2) Cluster them with DBSCAN → labels
    3) For each cluster:
        - compute centroid and its max‐radius
        - set R = max_radius + clearance
        - sample `sample_size` points equally around centroid at R
        - reject any that collide or leave env_bounds
    Returns:
      cluster_centroids: list of (3,) arrays
      sampled_rings:     list of (≤sample_size, 3) arrays
    """
    pts = epcds_arr.T
    minb, maxb = np.array(env_bounds["minbound"]), np.array(env_bounds["maxbound"])
    min_h, max_h = z_range

    # 1) only keep obstacles inside the bounds
    mask = np.all((pts >= minb) & (pts <= maxb), axis=1)
    in_pts = pts[mask]
    if len(in_pts) == 0:
        return [], []

    # 2) cluster
    clustering = DBSCAN(eps=cluster_eps, min_samples=min_samples).fit(in_pts)
    labels = clustering.labels_
    clusters = {lab: in_pts[labels==lab] for lab in set(labels) if lab >= 0}

    kdtree = cKDTree(pts)  # full scene for collision checks
    centroids, rings = [], []

    for lab, pts_c in clusters.items():
        # 3a) centroid & cluster‐radius
        ctr = pts_c.mean(axis=0)

        # height‐filter
        if not (min_h <= ctr[2] <= max_h):
            continue

        radii = np.linalg.norm(pts_c - ctr, axis=1)
        R_cluster = radii.max()
        R = R_cluster + clearance

        centroids.append(ctr)

        # 3b) generate equal‐angle candidates
        thetas = np.linspace(0, 2*np.pi, sample_size, endpoint=False)
        circle = np.vstack([
            ctr[0] + R*np.cos(thetas),
            ctr[1] + R*np.sin(thetas),
            np.full_like(thetas, ctr[2])
        ]).T

        # 3c) reject any too-close to obstacle (within clearance) or OOB
        good = []
        for p in circle:
            if kdtree.query_ball_point(p, clearance, eps=0.01):
                continue
            if not np.all((minb <= p) & (p <= maxb)):
                continue
            good.append(p)
        rings.append(np.array(good))

    return rings, centroids

def debug_figures_RRT(obj_loc, initial, original, smoothed, time_points):
    def extract_yaw_from_quaternion(quaternions):
        """
        Extract the yaw (heading angle) from a set of quaternions.
        """
        qx, qy, qz, qw = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        return yaw
    # Unpack trajectory components
    x_orig, y_orig, z_orig, vx_orig, vy_orig, vz_orig, qx_orig, qy_orig, qz_orig, qw_orig = original.T[1:11]
    x_smooth, y_smooth, z_smooth, vx_smooth, vy_smooth, vz_smooth, qx_smooth, qy_smooth, qz_smooth, qw_smooth = smoothed.T[1:11]

    original_rates = original[:, 13]
    smoothed_rates = smoothed[:, 13]

    x_init = initial[:, 0]
    y_init = initial[:, 1]

    # Compute yaw angles
    yaw_orig = np.arctan2(vy_orig, vx_orig)
    yaw_smooth = np.arctan2(vy_smooth, vx_smooth)

    yaw_alt_orig = extract_yaw_from_quaternion(original[:, 7:11])
    yaw_alt_smooth = extract_yaw_from_quaternion(smoothed[:, 7:11])

    # Compute orientation vectors (heading) from yaw angles
    orientation_x_orig = np.cos(yaw_alt_orig)
    orientation_y_orig = np.sin(yaw_alt_orig)
    orientation_x_smooth = np.cos(yaw_alt_smooth)
    orientation_y_smooth = np.sin(yaw_alt_smooth)

    # Create subplots
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))

    # Plot positions
    axes[0, 0].plot(x_orig, y_orig, '-o', label='Original Trajectory')
    axes[0, 0].quiver(x_orig, y_orig, vx_orig, vy_orig, angles='xy', scale_units='xy', scale=1.5, color='r', alpha=0.5, label='Heading')
    axes[0, 0].quiver(x_orig, y_orig, orientation_x_orig, orientation_y_orig, angles="xy", scale_units="xy", scale=1, color="b", alpha=0.7, label="Orientation")
    axes[0, 0].plot(x_init, y_init, '-x', label='RRT* Trajectory',color='lime')
    axes[0, 0].plot(obj_loc[0], obj_loc[1], 'o', label='Object Location', color='yellow')
    axes[0, 0].set_title('Original Trajectory in XY Plane')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].axis('equal')
    axes[0, 0].legend()
    axes[0, 0].invert_yaxis()

    axes[0, 1].plot(x_smooth, y_smooth, '--o', label='Smoothed Trajectory')
    axes[0, 1].quiver(x_smooth, y_smooth, vx_smooth, vy_smooth, # BLUE
                       angles='xy', scale_units='xy', scale=1.5, color='b', alpha=0.5, label='Heading')
    axes[0, 1].quiver(x_smooth, y_smooth, orientation_x_smooth, orientation_y_smooth, # RED
                       angles="xy", scale_units="xy", scale=1, color="r", alpha=0.7, label="Orientation")
    axes[0, 1].plot(x_init, y_init, '-x', label='RRT* Trajectory',color='lime')
    axes[0, 1].plot(obj_loc[0], obj_loc[1], 'o', label='Object Location', color='yellow')
    axes[0, 1].set_title('Smoothed Trajectory in XY Plane')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].axis('equal')
    axes[0, 1].legend()
    axes[0, 1].invert_yaxis()


    # Plot velocity components
    axes[1, 0].plot(time_points, vx_orig, label='Vx')
    axes[1, 0].plot(time_points, vy_orig, label='Vy')
    axes[1, 0].set_title('Original Velocity Components vs Time')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Velocity')
    axes[1, 0].legend()

    axes[1, 1].plot(time_points, vx_smooth, label='Vx')
    axes[1, 1].plot(time_points, vy_smooth, label='Vy')
    axes[1, 1].set_title('Smoothed Velocity Components vs Time')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Velocity')
    axes[1, 1].legend()

    # Plot yaw angles
    axes[2, 0].plot(time_points, np.degrees(yaw_orig), label='Yaw (degrees)')
    axes[2, 0].plot(time_points, np.degrees(yaw_alt_orig), label='Yaw Alt (degrees)', linestyle='--')
    axes[2, 0].set_title('Original Yaw Angle vs Time')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Yaw Angle (degrees)')
    axes[2, 0].legend()

    axes[2, 1].plot(time_points, np.degrees(yaw_smooth), label='Yaw (degrees)')
    axes[2, 1].plot(time_points, np.degrees(yaw_alt_smooth), label='Yaw Alt (degrees)', linestyle='--')
    axes[2, 1].set_title('Smoothed Yaw Angle vs Time')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Yaw Angle (degrees)')
    axes[2, 1].legend()

    # Plot quaternion components
    axes[3, 0].plot(time_points, qx_orig, label='qx')
    axes[3, 0].plot(time_points, qy_orig, label='qy')
    axes[3, 0].plot(time_points, qz_orig, label='qz')
    axes[3, 0].plot(time_points, qw_orig, label='qw')
    axes[3, 0].set_title('Original Quaternion Components vs Time')
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].set_ylabel('Quaternion Component')
    axes[3, 0].legend()

    axes[3, 1].plot(time_points, qx_smooth, label='qx')
    axes[3, 1].plot(time_points, qy_smooth, label='qy')
    axes[3, 1].plot(time_points, qz_smooth, label='qz')
    axes[3, 1].plot(time_points, qw_smooth, label='qw')
    axes[3, 1].set_title('Smoothed Quaternion Components vs Time')
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 1].set_ylabel('Quaternion Component')
    axes[3, 1].legend()

    # Plot angular rates
    axes[4, 0].plot(time_points, original_rates, label='Angular Rate')
    axes[4, 0].set_title('Original Angular Rates vs Time')
    axes[4, 0].set_xlabel('Time (s)')
    axes[4, 0].set_ylabel('Angular Rate (rad/s)')
    axes[4, 0].legend()

    axes[4, 1].plot(time_points, smoothed_rates, label='Angular Rate')
    axes[4, 1].set_title('Smoothed Angular Rates vs Time')
    axes[4, 1].set_xlabel('Time (s)')
    axes[4, 1].set_ylabel('Angular Rate (rad/s)')
    axes[4, 1].legend()

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

def process_branch(branch_id, positions, dt, constant_velocity, obj_loc, pad_t, threshold_distance, viz=False, randint=None, loiter=False):
    """
    Processes a single branch of positions to compute trajectory and smooth trajectory data.

    Parameters:
        branch_id (int): The ID of the branch being processed.
        positions (list or np.ndarray): The positions in the branch.
        dt (float): Time step for interpolation.
        constant_velocity (float): Target constant velocity for the trajectory.
        obj_loc (np.ndarray): Target object location.
        pad_t (float): Padding time in seconds.
        viz (bool): Whether to enable visualization for a specific branch.
        randint (int): Random integer for visualization selection.

    Returns:
        tuple: (trajectory, smooth_trajectory, nodes_RRT)
    """
        # Smooth the trajectory in position space using cubic spline
    def smooth_initial_trajectory(traj_x, traj_y, traj_z, traj_t, dense_time_points):
        spline_x = CubicSpline(traj_t, traj_x)
        spline_y = CubicSpline(traj_t, traj_y)
        spline_z = CubicSpline(traj_t, traj_z)
        smooth_x = spline_x(dense_time_points)
        smooth_y = spline_y(dense_time_points)
        smooth_z = spline_z(dense_time_points)
        return smooth_x, smooth_y, smooth_z
    
    # Function to weight quaternions
    def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Perform Spherical Linear Interpolation (SLERP) between two quaternions.

        Args:
            q1: First quaternion (4,).
            q2: Second quaternion (4,).
            t: Interpolation parameter (0 to 1).

        Returns:
            Interpolated quaternion (4,).
        """
        dot_product = np.dot(q1, q2)

        # If the dot product is negative, SLERP won't take the shortest path.
        # Fix by reversing one quaternion.
        if dot_product < 0.0:
            q2 = -q2
            dot_product = -dot_product

        # Clamp dot_product to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Calculate the angle between the quaternions
        theta_0 = np.arccos(dot_product)
        sin_theta_0 = np.sin(theta_0)

        if sin_theta_0 < 1e-6:
            # If the angle is very small, use linear interpolation
            return (1.0 - t) * q1 + t * q2

        theta = theta_0 * t
        sin_theta = np.sin(theta)

        s1 = np.sin(theta_0 - theta) / sin_theta_0
        s2 = sin_theta / sin_theta_0

        return s1 * q1 + s2 * q2

    def weight_quaternions(quaternions: np.ndarray, q_target: np.ndarray, progress: np.ndarray) -> np.ndarray:
        """
        Weight quaternions along the trajectory towards the target quaternion.

        Args:
            quaternions: Array of quaternions (N x 4).
            q_target: Target quaternion (4,).
            progress: Array of normalized progress values (0 to 1).

        Returns:
            Weighted quaternions (N x 4).
        """
        weighted_quaternions = []
        for i, q in enumerate(quaternions):
            weight = progress[i]
            # qt_flipped = obedient_quaternion(q_target, q)

            interpolated_q = slerp(q, q_target[i], weight)
            if i > 0:
                interpolated_q = obedient_quaternion(interpolated_q, weighted_quaternions[-1])
            weighted_quaternions.append(interpolated_q)
        return np.array(weighted_quaternions)
    
    def exp_mabr(body_rates, alpha=1.0):
        """
        Apply exponential moving average (EMA) smoothing to body rates.

        Parameters:
            body_rates (np.ndarray): Array of body rates (shape: Nx3 or similar).
            alpha (float): Smoothing factor (0 < alpha <= 1).

        Returns:
            np.ndarray: Smoothed body rates (shape: Nx3).
        """
        # Initialize the smoothed rates array
        smoothed_rates = np.zeros_like(body_rates)
        smoothed_rates[0] = body_rates[0]  # Start with the first body rate

        # Apply EMA to each subsequent body rate
        for t in range(1, len(body_rates)):
            smoothed_rates[t] = alpha * body_rates[t] + (1 - alpha) * smoothed_rates[t - 1]

        return smoothed_rates
    
    def compute_quaternions(trajectory: np.ndarray) -> list[np.ndarray]:
        """
        For each timestep i, look at (vx, vy) = (trajectory[i,4], trajectory[i,5]).
        If speed > 1e-6, compute yaw = atan2(vy, vx), then build a ZYX‐Euler quaternion
        q = from_euler('ZYX', [yaw, 0, 0]).
        If speed ≈ 0, simply reuse the previous quaternion.
        Use obedient_quaternion(…) to keep quaternion‐sign continuous.
        Returns a Python list of length N, each entry = np.array([qx, qy, qz, qw]).
        """
        quaternions: list[np.ndarray] = []
        eps = 1e-6

        for i in range(len(trajectory)):
            vx = trajectory[i, 4]
            vy = trajectory[i, 5]
            speed_xy = np.hypot(vx, vy)

            if speed_xy > eps:
                raw_yaw = np.arctan2(vy, vx)  # in (−π, +π]
                # Build a pure‐yaw quaternion using ZYX convention:
                #   R = Rz(raw_yaw) * Ry(0) * Rx(0) = Rz(raw_yaw).
                quat = Rotation.from_euler("ZYX", [raw_yaw, 0.0, 0.0], degrees=False).as_quat()
            else:
                # if speed is near zero, freeze at the previous quaternion
                if i == 0:
                    # First frame, no “previous” quaternion exists → use identity
                    quat = np.array([0.0, 0.0, 0.0, 1.0])
                else:
                    quat = quaternions[-1].copy()

            # Enforce continuity in sign (q and −q represent the same rotation)
            if i > 0:
                quat = obedient_quaternion(quat, quaternions[-1])

            quaternions.append(quat)

        return quaternions
    
    def compute_quaternions_old(trajectory):
        quaternions = []
        for i in range(len(trajectory)):
            if np.sqrt(trajectory[i,4]**2 + trajectory[i,5]**2) > 1e-6:
                traj_yaw = np.arctan2(trajectory[i, 5], trajectory[i, 4])
                quat = Rotation.from_euler('z', traj_yaw).as_quat()
            else:
                quat = quaternions[-1]
            if i > 0:
                quat = obedient_quaternion(quat, quaternions[-1])
            quaternions.append(quat)
        return quaternions

    def compute_angular_rates(trajectory, target_times):
        quats = trajectory[:, 7:11]
        quaternions = [row.tolist() for row in quats]
        angular_rates = []
        dt = np.diff(target_times)  # Time intervals between each sample
        
        for i in range(len(quaternions) - 1):
            # Current and next quaternion
            q_current = Rotation.from_quat(quaternions[i])
            q_next = Rotation.from_quat(quaternions[i + 1])
            
            # Calculate the relative rotation quaternion
            delta_q = q_current.inv() * q_next

            # Convert delta_q to axis-angle representation
            angle = delta_q.magnitude()
            axis = delta_q.as_rotvec() / angle if angle > 1e-8 else np.zeros(3)
            
            # Compute angular rate (angular velocity vector)
            omega = (axis * angle) / dt[i] if dt[i] > 0 else np.zeros(3)

            angular_rates.append(omega)
        
        # Append the last angular rate to maintain consistent array length
        angular_rates.append(angular_rates[-1])
        angular_rates = np.array(angular_rates)
        
        return angular_rates
    

    positions = np.array(positions)

    # Ensure positions have 3D coordinates
    if positions.shape[1] == 2:
        positions = np.hstack((positions, np.zeros((positions.shape[0], 1))))
    elif positions.shape[1] != 3:
        raise ValueError(f"Branch {branch_id} positions must have shape (n, 2) or (n, 3).")

    # Compute distances and handle zero-length branches
    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_distances = np.insert(np.cumsum(segment_lengths), 0, 0)
    total_distance = cumulative_distances[-1]

    if total_distance == 0:
        print(f"Branch {branch_id} has zero total length. Skipping.")
        return None, None, None
    
    # ----- Compute timepoints based on constant velocity -----
    # Time for each segment = length / speed
    # e.g. segment i is from positions[i] to positions[i+1]
    segment_times = segment_lengths / constant_velocity

    # Accumulate times
    timepoints = np.insert(np.cumsum(segment_times), 0, 0)  # shape (N,)
    # print(f"Branch {branch_id} timepoints: {timepoints}")

    # Create a dense time array at increment dt
    final_time = timepoints[-1]
    # Arange from 0 up to final_time (inclusive or slightly more)
    times = np.arange(0, final_time + dt/2, dt)
    # print(f"Branch {branch_id} times: {times}")

    # # Time points
    # timepoints = np.linspace(0, positions.shape[0] - 1, positions.shape[0])
    # print(f"Branch {branch_id} timepoints: {timepoints}")
    # times = np.arange(0, len(positions) - 1, dt)
    # times = np.append(times, times[-1] + dt)
    # print(f"Branch {branch_id} times: {times}")

    trajectory = np.zeros((len(times), 18))
    smooth_trajectory = np.zeros((len(times), 18))
    trajectory[:, 0] = times
    smooth_trajectory[:, 0] = times

    # Ensure positions have 3D coordinates
    if positions.shape[1] == 2:
        positions = np.hstack((positions, np.zeros((positions.shape[0], 1))))
    elif positions.shape[1] != 3:
        raise ValueError(f"Branch {branch_id} positions must have shape (n, 2) or (n, 3).")

    # Compute distances and handle zero-length branches
    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_distances = np.insert(np.cumsum(segment_lengths), 0, 0)
    total_distance = cumulative_distances[-1]

    if total_distance == 0:
        print(f"Branch {branch_id} has zero total length. Skipping.")
        return None, None, None

    # Interpolate positions
    x_samples = np.interp(times, timepoints, positions[:, 0])
    y_samples = np.interp(times, timepoints, positions[:, 1])
    z_samples = np.interp(times, timepoints, positions[:, 2])
    positions_samples = np.vstack((x_samples, y_samples, z_samples)).T
    trajectory[:, 1:4] = positions_samples

    # Compute velocities
    vx = np.gradient(x_samples, dt)
    vy = np.gradient(y_samples, dt)
    vz = np.gradient(z_samples, dt)
    vnorm = np.linalg.norm(np.vstack((vx, vy, vz)), axis=0)
    vx, vy, vz = (v / vnorm for v in (vx, vy, vz))
    vx, vy, vz = (v * constant_velocity for v in (vx, vy, vz))
    trajectory[:, 4:7] = np.column_stack((vx, vy, vz))

    # Calculate yaw and quaternions
    quaternions = compute_quaternions(trajectory)
    target_quaternion = traj_orient(trajectory[:, 1:4], np.array(quaternions), obj_loc)

    # Distance-based progress
    distance_to_final = np.linalg.norm(trajectory[:, 1:4] - trajectory[-1, 1:4], axis=1)
    # progress = np.zeros(len(trajectory))
    # within_threshold = distance_to_final <= threshold_distance
    # progress[within_threshold] = np.linspace(0, 1, np.sum(within_threshold))
    progress = np.linspace(0, 1, len(trajectory))  # Linear progress from 0 to 1
    # progress = (1 - np.exp(-0.5 * progress)) / (1 - np.exp(-0.5))
    # progress = (np.exp(0.5 * progress) - 1 ) / (np.exp(0.5) - 1)
    progress = np.log1p(progress * (np.e - 1)) / np.log(np.e)  # Logarithmic scaling
    # progress = np.ones(len(trajectory))
    adjusted_quaternions = weight_quaternions(np.array(quaternions), target_quaternion, progress)
    trajectory[:, 7:11] = adjusted_quaternions

    angular_rates = compute_angular_rates(trajectory, times)
    trajectory[:, 11:14] = angular_rates

    # Pad trajectory
    pad_ts = np.linspace(times[-1], times[-1] + pad_t, int(pad_t / dt))
    for i in range(len(pad_ts) - 1):
        trajectory = np.vstack((trajectory, trajectory[-1]))
        trajectory[-1, 13] = 0
        trajectory[-1, 4:7] = 0
        trajectory[-1, 0] = pad_ts[i]

    # Find the times that correspond to positions existing in both positions and in x_samples, y_samples, z_samples
    common_times = []
    common_positions = []
    for i, pos in enumerate(positions):
        for j, (x, y, z) in enumerate(zip(x_samples, y_samples, z_samples)):
            if np.allclose(pos, [x, y, z], atol=1e-1):
                common_times.append(times[j])
                common_positions.append(pos)
                break
    # Combine common_times and common_positions into a paired list
    common_time_position_pairs = list(zip(common_times, common_positions))

    # Get velocities that correspond to times in common_times
    common_velocities = []
    for t in common_times:
        idx = np.where(times == t)[0]
        if len(idx) > 0:
            common_velocities.append(trajectory[idx[0], 4:7])

    # # Set the first and last velocity to zero
    # if len(common_velocities) > 0:
    #     # common_velocities[0] = np.zeros(3)
    #     common_velocities[-1] = np.zeros(3)
    common_velocities = np.array(common_velocities)

    # Smooth trajectory
    smooth_x, smooth_y, smooth_z = smooth_initial_trajectory(
        positions[:, 0], positions[:, 1], positions[:, 2], timepoints, times
    )
    smooth_trajectory[:, 1:4] = np.column_stack((smooth_x, smooth_y, smooth_z))
    smooth_vx, smooth_vy, smooth_vz = smooth_initial_trajectory(
        common_velocities[:,0], common_velocities[:,1], common_velocities[:,2], common_times, times
    )
    smooth_trajectory[:, 4:7] = np.column_stack((smooth_vx, smooth_vy, smooth_vz))

    smoothed_quaternions = compute_quaternions(smooth_trajectory)
    smooth_trajectory[:, 7:11] = np.array(smoothed_quaternions)
###################################################################################################################
    smooth_trajectory[:, 7:11] = adjusted_quaternions

    #NOTE gives a target quaternion to track based on object location
    target_quaternion = traj_orient(smooth_trajectory[:,1:4],np.array(smoothed_quaternions),obj_loc)
    
    # Calculate distance to final position
    # Calculate progress based on distance
    progress = np.linspace(0, 1, len(smooth_trajectory))  # Linear progress from 0 to 1
    # progress = (np.exp(0.5 * progress) - 1 ) / (np.exp(0.5) - 1)
    progress = np.log1p(progress * (np.e - 1)) / np.log(np.e)  # Logarithmic scaling
    # if loiter == True:
    #     print(f"Loitering enabled for branch {branch_id}.")
    #     progress = np.ones(len(trajectory))
    smooth_adjusted_quaternions = weight_quaternions(np.array(smoothed_quaternions), target_quaternion, progress)
    smooth_trajectory[:,7:11] = smooth_adjusted_quaternions

    if loiter == True:
        print(f"Loitering enabled for branch {branch_id}.")
        smooth_trajectory[:, 7:11] = target_quaternion
###################################################################################################################
    smooth_angular_rates = compute_angular_rates(smooth_trajectory, times)

    smooth_trajectory[:,11:14] = smooth_angular_rates

    # Get smoothed orientations that correspond to times in common_times
    common_orientations = []
    for t in common_times:
        idx = np.where(times == t)[0]
        if len(idx) > 0:
            common_orientations.append(smooth_trajectory[idx[0], 7:11])

    # Pad trajectory
    pad_ts = np.linspace(times[-1], times[-1] + pad_t, int(pad_t / dt))
    for i in range(len(pad_ts) - 1):
        smooth_trajectory = np.vstack((smooth_trajectory, smooth_trajectory[-1]))
        smooth_trajectory[-1, 13] = 0
        smooth_trajectory[-1, 4:7] = 0
        smooth_trajectory[-1, 0] = pad_ts[i]
        times = np.append(times, times[-1] + dt)

    # Set the thrust magnitude
    smooth_trajectory[:,14:18] = 0.4  # u1, u2, u3, u4

    #NOTE replaces the positions in the smoothed trajectory with the original positions
    # smooth_trajectory[:,1:4] = trajectory[:,1:4]
    # smooth_trajectory[:,4:7] = trajectory[:,4:7]
    # Replace trailing elements of smooth_trajectory[:,7:11] that are very close to zero with the last nonzero element
    # nonzero_indices = np.where(np.linalg.norm(smooth_trajectory[:, 7:11], axis=1) > 1e-6)[0]
    # if len(nonzero_indices) > 0:
    #     last_nonzero_index = nonzero_indices[-1]
    #     for i in range(last_nonzero_index + 1, len(smooth_trajectory)):
    #         smooth_trajectory[i, 7:11] = smooth_trajectory[last_nonzero_index, 7:11]

    common_time_position_orientation_pairs = [
        (time, pos, orient) for (time, pos), orient in zip(common_time_position_pairs, common_orientations)
    ]

    if viz and (branch_id == randint or loiter == True):
        print(f"Visualizing branch {branch_id}.")
        if len(times) != len(trajectory):
            print(f"Length mismatch: times ({len(times)}) != trajectory ({len(trajectory)})")
            raise ValueError("The length of times and trajectory must be the same.")
        debug_figures_RRT(obj_loc, positions, trajectory, smooth_trajectory, times)
        debug_dict = {
            "obj_loc": obj_loc,
            "positions": positions,
            "trajectory": trajectory,
            "smooth_trajectory": smooth_trajectory,
            "times": times
        }
        smooth_trajectory = smooth_trajectory.T
        return smooth_trajectory, common_time_position_orientation_pairs, debug_dict
    else:
        smooth_trajectory = smooth_trajectory.T
        return smooth_trajectory, common_time_position_orientation_pairs, None
    
def parameterize_RRT_trajectories(branches, obj_loc, constant_velocity, sampling_frequency, randint=None, loiter=False):
    #NOTE True to plot a single trajectory, False for normal (generate data) use
    if randint is not None:
        viz = True
    else:
        viz = False

    dt = 1 / sampling_frequency

    new_branches = []
    nodes_RRT = []
    debug_dict = None
    branches = [branch[::-1] for branch in branches]
    for idbr, positions in enumerate(branches):
        result = process_branch(
            branch_id=idbr,
            positions=positions,
            dt=dt,
            constant_velocity=constant_velocity,
            obj_loc=obj_loc,
            pad_t=2,
            viz=viz,
            threshold_distance=1.5,
            randint=randint,
            loiter=loiter
            )
        if result[0] is None and result[1] is None and result[2] is None:
            print(f"Breaking out of the loop. Branch {idbr} returned None.")
            continue  # Continue the loop
        elif viz and (randint == idbr or loiter == True):
            new_branches.append(result[0])
            nodes_RRT.append(result[1])
            debug_dict = result[2]
        else:
            new_branches.append(result[0])
            nodes_RRT.append(result[1])
            
    if viz:
        return new_branches, nodes_RRT, debug_dict
    else:
        return new_branches, nodes_RRT

def traj_orient(
    trajectory: np.ndarray,   # shape = (N,3), camera positions [x,y,z]
    quaternions:  np.ndarray, # shape = (N,4), unused here but kept in signature
    goal_xyz:     np.ndarray   # shape = (3,), [x_goal, y_goal, z_goal]
) -> np.ndarray:
    """
    Returns an (N,4) array of quaternions (qx,qy,qz,qw) so that:
      - local +X (camera forward) points horizontally (in XY) toward goal_xyz
      - local +Z is “down” in world, so that the same ZYX‐Euler convention matches your optimizer.

    We build each frame’s yaw = atan2(dy,dx), then produce
      q = Rotation.from_euler("ZYX", [yaw, 0, 0]).as_quat()

    Special handling:
      • If camera_x,y == goal_x,y (i.e. dist_xy < eps), we keep the previous yaw (no jump).
      • We “unwrap” yaw around ±π so there is no discontinuous 360° flip.

    Returns:
      new_quaternions: np.ndarray of shape (N,4), dtype=float.
    """
    N = len(trajectory)
    new_quats = np.zeros((N, 4), dtype=float)
    eps = 1e-6

    prev_yaw_unwrapped = None

    for i in range(N):
        cam_x, cam_y, _ = trajectory[i]
        dx = goal_xyz[0] - cam_x
        dy = goal_xyz[1] - cam_y
        dist_xy = np.hypot(dx, dy)

        if dist_xy < eps:
            # camera is (almost) directly above/below the goal in XY
            if prev_yaw_unwrapped is None:
                raw_yaw = 0.0
            else:
                raw_yaw = prev_yaw_unwrapped
        else:
            raw_yaw = np.arctan2(dy, dx)  # in (−π, +π]

        # Unwrap around ±π so yaw moves smoothly
        if prev_yaw_unwrapped is None:
            unwrapped_yaw = raw_yaw
        else:
            delta = raw_yaw - prev_yaw_unwrapped
            if delta > np.pi:
                raw_yaw -= 2.0 * np.pi
            elif delta < -np.pi:
                raw_yaw += 2.0 * np.pi
            unwrapped_yaw = raw_yaw

        prev_yaw_unwrapped = unwrapped_yaw

        # Build pure‐yaw quaternion via the ZYX convention:
        #   (yaw, pitch=0, roll=0) in “ZYX” → R = Rx(0) * Ry(0) * Rz(unwrapped_yaw)
        q = Rotation.from_euler("ZYX", [unwrapped_yaw, 0.0, 0.0], degrees=False).as_quat()

        # continuity in sign for logging‐smoothness (not physically necessary)
        if i > 0 and np.dot(q, new_quats[i-1]) < 0:
            q = -q

        new_quats[i] = q

    return new_quats

def fo_to_xu(fo:np.ndarray,quad:Union[None,Dict[str,Union[float,np.ndarray]]])  -> np.ndarray:
    """
    Converts a flat output vector to a state vector and body-rate command. Returns
    just x if quad is None.

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

    # Compute Body-Rate Command if Quadcopter is defined
    if quad is not None:
        m,tn = quad["m"],quad["tn"]
        ut = np.hstack((m*c/tn,wt))
    else:
        ut = np.zeros(0)

    # Stack
    xu = np.hstack((pt,vt,qt,ut))

    return xu

def xu_to_fo(xu:np.ndarray,quad:Dict[str,Union[float,np.ndarray]]) -> np.ndarray:
    """
    Converts a state vector to approximation of flat output vector.

    Args:
        xuv:     State vector (NOTE: Uses full state).

    Returns:
        fo:     Flat output vector.
    """

    # Unpack variables
    wxk,wyk,wzk = xu[10],xu[11],xu[12]
    m,I,fMw = quad["m"],quad["I"],quad["fMw"]

    # Initialize output
    fo = np.zeros((4,5))

    # Compute position terms
    fo[0:3,0] = xu[0:3]

    # Compute velocity terms
    fo[0:3,1] = xu[3:6]

    # Compute acceleration terms
    Rk = Rotation.from_quat(xu[6:10]).as_matrix()       # Rotation matrix
    xbt,ybt,zbt = Rk[:,0],Rk[:,1],Rk[:,2]               # Body frame vectors
    gt = np.array([0.00,0.00,-9.81])                    # Acceleration due to gravity vector
    c = (fMw@xu[13:17])[0]/m                            # Acceleration due to thrust vector

    fo[0:3,2] = c*zbt-gt

    # Compute yaw term
    psi = np.arctan2(Rk[1,0], Rk[0,0])

    fo[3,0]  = psi

    # Compute yaw rate term
    xct = np.array([np.cos(psi), np.sin(psi), 0])     # Intermediate frame x vector
    yct = np.array([-np.sin(psi), np.cos(psi), 0])    # Intermediate frame y vector
    B1 = c
    B3 = -yct.T@zbt
    C3 = np.linalg.norm(np.cross(yct,zbt))
    D1 = wyk*(B1*C3)/C3
    D3 = (wzk*(B1*C3)+(B3*D1))/B1

    psid = D3/(xct.T@xbt)

    fo[3,1] = psid

    # Compute yaw acceleration term
    Iinv = np.linalg.inv(I)
    rv1:np.ndarray = xu[10:13]            # intermediate variable
    rv2:np.ndarray = I@xu[10:13]          # intermediate variable
    utau = (fMw@xu[13:17])[1:4]
    wd = Iinv@(utau - np.cross(rv1,rv2))
    E1 = wd[1]*(B1*C3)/C3
    E3 = (wd[2]*(B1*C3)+(B3*E1))/B1

    psidd = (E3 - 2*psid*wzk*xct.T@ybt + 2*psid*wyk*xct.T@zbt + wxk*wyk*yct.T@ybt + wxk*wzk*yct.T@zbt)/(xct.T@xbt)

    fo[3,2] = psidd

    return fo

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
             quad:Union[None,Dict[str,Union[float,np.ndarray]]]) -> np.ndarray:
    """
    Converts a trajectory spline (defined by tf,CP) to a state vector and control input.
    Returns just x if quad is None.

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

def TS_to_xu(tcr:float,Tps:np.ndarray,CPs:np.ndarray,
              quad:Union[None,Dict[str,Union[float,np.ndarray]]]) -> np.ndarray:
    """
    Extracts the state and input from a sequence of trajectory splines (defined by
    Tps,CPs). Returns just x if quad is None.

    Args:
        - tcr:  Current segment time.
        - Tps:  Trajectory segment times.
        - CPs:  Trajectory control points.
        - quad: Quadcopter specifications.

    Returns:
        xu:    State vector and control input.
    """
    idx = np.max(np.where(Tps < tcr)[0])
    
    if idx == len(Tps)-1:
        tcr = Tps[-1]
        t0,tf = Tps[-2],Tps[-1]
        CPk = CPs[-1,:,:]
    else:
        t0,tf = Tps[idx],Tps[idx+1]
        CPk = CPs[idx,:,:]

    xu = ts_to_xu(tcr-t0,tf-t0,CPk,quad)

    return xu

def TS_to_tXU(Tps:np.ndarray,CPs:np.ndarray,
              quad:Union[None,Dict[str,Union[float,np.ndarray]]],
              hz:int) -> np.ndarray:
    """
    Converts a sequence of trajectory splines (defined by Tps,CPs) to a trajectory
    rollout. Returns just tX if quad is None.

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