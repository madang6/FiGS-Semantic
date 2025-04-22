import numpy as np
import open3d as o3d
import torch
import yaml
import os
from typing import List

from figs.render.gsplat_semantic import GSplat

import figs.scene_editing.scene_editing_utils as scdt
import figs.utilities.trajectory_helper as th
from figs.tsampling.rrt_datagen_v10 import *

def get_objectives(nerf:GSplat, objectives, visualize):
    viz=visualize
    transform = torch.eye(4)
    dataparser_scale = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
    dataparser_transform = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform
    transform[:3,:3] = dataparser_transform[:3,:3]
    invtransform = np.asarray(torch.linalg.inv(transform))

    epcds, epcds_arr, epcds_bounds, pcd, pcd_mask, pcd_attr = scdt.rescale_point_cloud(nerf)

    # Translate to gemsplat syntax
    positives = objectives
    threshold = 0.6 #Currently may be overloaded within get_points()
    filter_radius = 0.075 #Currently may be overloaded within get_points()
    filter_radius = [filter_radius] * len(objectives)

    threshold_obj = [threshold] * len(objectives)
    # print(pcd.keys())
    # env_points = np.asarray(pcd["env_pcd"].points).T
    
    obj_targets = []

    for idx, obj in enumerate(objectives):

        print('*' * 50)
        print(f'Processing Object: {obj}')
        print('*' * 50)

        # source location
        src_centroid, src_z_bounds, scene_pcd, similarity_mask, other_attr = scdt.get_centroid(nerf=nerf,
                                                                                        env_pcd=pcd,
                                                                                        pcd_attr=pcd_attr,
                                                                                        positives=objectives[idx],
                                                                                        negatives='window,wall,floor,ceiling,object, things, stuff, texture',
                                                                                        threshold=threshold_obj[idx],
                                                                                        visualize_pcd=False,
                                                                                        enable_convex_hull=True,
                                                                                        enable_spherical_filter=True,
                                                                                        enable_clustering=False,
                                                                                        filter_radius=filter_radius[idx],
                                                                                        obj_priors={},
                                                                                        use_Mahalanobis_distance=True)

        # object
        object_pcd_points = np.asarray(scene_pcd.points)[similarity_mask]
        object_pcd_colors = np.asarray(scene_pcd.colors)[similarity_mask]
        object_pcd_sim = other_attr['raw_similarity'][similarity_mask].cpu().numpy()

        # if any(item in obj for item in ['pot', 'pan', 'lid']):
        # plane-fitting
        pcd_clus = o3d.geometry.PointCloud()
        pcd_clus.points = o3d.utility.Vector3dVector(object_pcd_points[:, :3])
        pcd_clus.colors = o3d.utility.Vector3dVector(object_pcd_colors)

        if viz:
            # vpcds = np.asarray(pcd_clus.points).T / dataparser_scale
            # vpcds = np.vstack((vpcds, np.ones((1, vpcds.shape[1]))))
            # viz_pcd = o3d.geometry.PointCloud()
            # viz_pcd.points=o3d.utility.Vector3dVector(vpcds)
            # viz_pcd.colors=o3d.utility.Vector3dVector(np.asarray(vpcds.colors))
            env_pcd_scaled = np.asarray(pcd_clus.points).T / dataparser_scale
            env_pcd_scaled = np.vstack((env_pcd_scaled, np.ones((1, env_pcd_scaled.shape[1]))))
            env_pcd_scaled = nerf.T_w2g @ env_pcd_scaled
            env_pcd_scaled = env_pcd_scaled[:3, :].T
            vpcds = o3d.geometry.PointCloud()
            vpcds.points=o3d.utility.Vector3dVector(env_pcd_scaled)
            vpcds.colors=o3d.utility.Vector3dVector(np.asarray(pcd_clus.colors))
            o3d.visualization.draw_plotly([vpcds])
        # # remove outliers
        # pcd_clus, _ = pcd_clus.remove_radius_outlier(nb_points=30, radius=0.03) # r=0.03 maybe nb_points=5

        obj_targets.append(src_centroid)

    # hemisphere_radius = [0.1, 0.1, 0.1]
    # theta_intervals = [-np.pi / 2, -np.pi, -3*np.pi/2, 0.00] #TODO (azimuth) get this dynamically from the drone?
    # phi_intervals = [80*np.pi/180] #elevation
    # exclusion_radius = 0.03 #Determines how close the camera can get to any point in the environment
    # pose_targets = gh.get_hemisphere(nerf, np.asarray(pts["env_pcd"].points).T, pts["src_centroid"], hemisphere_radius, theta_intervals, phi_intervals, exclusion_radius)
    # print(pose_targets)

    # Transform objects from scdtlat frame to Mocap frame
    for i in range(len(obj_targets)):
        obj_targets[i] = obj_targets[i].reshape(3, -1)

        obj_targets[i] = obj_targets[i] / dataparser_scale
        obj_targets[i] = np.vstack((obj_targets[i], np.ones((1, obj_targets[i].shape[1]))))

        # obj_targets[i] = np.asarray(nerf.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"]) @ invtransform @ np.asarray(obj_targets[i])
        # obj_targets[i] = np.asarray(nerf.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"]) @ np.asarray(obj_targets[i])
        obj_targets[i] = nerf.T_w2g @ obj_targets[i]

        obj_targets[i] = obj_targets[i][:3, :].T

    # print(f"epcds bounds: {epcds_bounds}")
    # print(ASDGasajklhsdgnclvkn)
    return obj_targets, epcds_bounds, epcds, epcds_arr

def generate_rrt_paths(
        config_file,
        simulator, 
        pcd, pcd_arr, 
        objectives:List[str], obj_targets, 
        semantic_centroid, env_bounds, 
        Niter_RRT, viz=True
        ):
    def get_config_option(option_name, prompt, valid_options=None, default=None):
        if option_name in config:
            value = config[option_name]
            print(f"{option_name} set to {value} from config.yml")
            if valid_options and value not in valid_options:
                print(f"Invalid value for {option_name} in config.yml. Using default or prompting.")
                value = None
        else:
            value = None

        if value is None:
            value = input(prompt).strip()
            if valid_options:
                while value not in valid_options:
                    print(f"Invalid input. Valid options are: {', '.join(valid_options)}")
                    value = input(prompt).strip()
            if default and not value:
                value = default
        return value
    def create_cylinder_between_points(p1, p2, radius=0.01, resolution=20):
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0: 
            return None
        direction /= length

        # build unit‐height cylinder along Z
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius, length, resolution, 4)
        cyl.compute_vertex_normals()

        # compute axis & angle between [0,0,1] and direction
        z_axis = np.array([0,0,1.0])
        axis = np.cross(z_axis, direction)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            # either parallel or anti‐parallel; if anti, rotate 180° around X
            if np.dot(z_axis, direction) < 0:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.pi * np.array([1,0,0]))
            else:
                R = np.eye(3)
        else:
            axis /= axis_len
            angle = np.arccos(np.dot(z_axis, direction))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

        cyl.rotate(R, center=(0,0,0))

        # move to midpoint
        midpoint = (p1 + p2) * 0.5
        cyl.translate(midpoint)

        return cyl

    viz = True
    config = {}
    # Check if the configuration file exists
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file) or {}
        print("Configuration loaded from config.yml")
    else:
        print("Configuration file not found. Proceeding with interactive inputs.")

    # Ask the user to select the algorithm
    algorithm_input = get_config_option(
        'algorithm',
        "Select algorithm (RRT/RRT*): ",
        valid_options=['RRT', 'RRT*'],
        default='RRT'
    ).upper()

    # Select dimension: 1, 2, or 3
    dimension = get_config_option(
        'dimension',
        "Enter the dimension (2 or 3): ",
        valid_options=['2', '3'],
        default='2'
    )
    dimension = int(dimension)

    # Choose whether to prevent edge overlap
    prevent_edge_overlap_input = get_config_option(
        'prevent_edge_overlap',
        "Prevent edge overlap? (y/n): ",
        valid_options=['y', 'n'],
        default='y'
    )
    prevent_edge_overlap = prevent_edge_overlap_input.lower() == 'y'

    # Choose whether to use exact edge lengths
    exact_step_input = get_config_option(
        'exact_step',
        "Exact edge lengths? (y/n): ",
        valid_options=['y', 'n'],
        default='y'
    )
    exact_step = exact_step_input.lower() == 'y'

    bounded_step = False
    use_branch_pruning = False
    if not exact_step:
        # Choose whether to bound edge lengths
        bounded_step_input = get_config_option(
            'bounded_step',
            "Bound edge lengths? (y/n): ",
            valid_options=['y', 'n'],
            default='y'
        )
        bounded_step = bounded_step_input.lower() == 'y'

    # Initialize RRT
    #NOTE Currently only does 2D, have hardcoded the envbounds
    # ebounds = (env_bounds["minbound"][:2], env_bounds["maxbound"][:2])
    # ebounds = [tuple(env_bounds["minbound"][:2]), tuple(env_bounds["maxbound"][:2])]
    ebounds = [(env_bounds["minbound"][0], env_bounds["maxbound"][0]), (env_bounds["minbound"][1], env_bounds["maxbound"][1])]
    print(f"env_bounds: {ebounds}")

    trajset = {}
    for (target, pose, centroid) in zip(objectives, obj_targets, semantic_centroid):
        pose = pose.flatten()
        pose[2] = -1.1
        # print(f"shape of pose: {pose.shape}")
        print(f"target: {target}")
        print(f"pose: {pose}")
        # print(f"shape of start: {start.shape}")
        rrt = RRT(
            env_arr=pcd_arr,
            env_pts=pcd,
            start=pose[:2],
            obj=centroid[:2],
            bounds=ebounds,
            altitude=pose[2],
            dimension=dimension,
            step_size=1.0,
            collision_check_resolution=0.1,
            max_iter=Niter_RRT,
            exact_step=exact_step,
            bounded_step=bounded_step,
            algorithm=algorithm_input,  # Pass the selected algorithm
            prevent_edge_overlap=prevent_edge_overlap
        )

        # Build RRT
        rrt.build_rrt()
        
        # Get all leaf nodes in the tree
        leaf_nodes = [node for node in rrt.nodes if not node.children]

        # Extract paths from each leaf node
        paths = []
        for leaf_node in leaf_nodes:
            path = rrt.get_path_from_leaf_to_root(leaf_node)
            paths.append(path)

        # line_sets = []
        # for idbr, positions in enumerate(paths):
        #     # Ensure positions is an (M,2) float64 array and add Z component
        #     pts = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        #     if len(pts) < 2:
        #         continue  # can’t draw a line with fewer than 2 points

        #     # Add Z component (altitude) to each point
        #     z_component = np.full((pts.shape[0], 1), pose[2])
        #     pts = np.hstack((pts, z_component))

        #     # Create edge list [(0,1), (1,2), …]
        #     edges = [[i, i+1] for i in range(len(pts)-1)]

        #     # Build the LineSet
        #     ls = o3d.geometry.LineSet(
        #     points=o3d.utility.Vector3dVector(pts),
        #     lines=o3d.utility.Vector2iVector(edges)
        #     )

        #     # Set a uniform color for all branches
        #     color = [0.0, 0.5, 1.0]  # appealing blue RGB
        #     ls.colors = o3d.utility.Vector3dVector([color for _ in edges])

        #     line_sets.append(ls)
        cylinders = []
        for branch in paths:
            pts2d = np.asarray(branch, dtype=np.float64).reshape(-1,2)
            if len(pts2d) < 2: continue
            zcol = np.full((len(pts2d),1), pose[2])
            pts3d = np.hstack([pts2d, zcol])

            for i in range(len(pts3d)-1):
                cyl = create_cylinder_between_points(pts3d[i], pts3d[i+1], radius=0.01)
                if cyl is not None:
                    cyl.paint_uniform_color([0,0.5,1.0])
                    cylinders.append(cyl)
        
        if viz is True:
            # rrt.visualize(show_sampled_points=True)
            # rrt.plot_tree()

            # 3) Draw everything together
            # flip_transform = np.linalg.inv(simulator.gsplat.T_w2g)
            # pcd.transform(flip_transform)
            # o3d.visualization.draw_geometries(
            #     [pcd, *line_sets],
            #     window_name="Point Cloud + Tree Branches",
            #     width=800, height=600
            # )
            o3d.visualization.draw_geometries(
            [pcd, *cylinders],
            window_name="Point Cloud + Fat Tree Branches",
            width=800, height=600
            )

        trajset[target] = paths
        
    return trajset

def visualize_rrt_trajectories(simulator, scene, viz=False):

    def load_config_file(base_path, subfolder, filename):
        config_path = os.path.join(base_path, subfolder)
        for root, _, files in os.walk(config_path):
            if filename in files:
                with open(os.path.join(root, filename), 'r') as file:
                    return yaml.safe_load(file), os.path.join(root, filename)
        raise FileNotFoundError(f"{filename} not found in {config_path}")

    # Load scene configuration
    scene_config,cfg_path = load_config_file(simulator.configs_path, "course", f"{scene}.yml")
    queries = scene_config.get("queries", [])
    if not queries:
        raise ValueError("No queries found in the scene.yml file.")
    radii = [(scene_config.get("r1", 0.5), scene_config.get("r2", 0.5))]
    Niter_RRT = scene_config.get("N", 1000)

    obj_targets, env_bounds, epcds, epcds_arr = get_objectives(simulator.gsplat, queries, viz)
    
    # Obtain goal poses and object centroid
    goal_poses, obj_centroid = th.process_RRT_objectives(obj_targets, epcds_arr, env_bounds, radii)
    print(f"obj_centroid: {obj_centroid}")

    # Generate RRT* Paths
    trajset = generate_rrt_paths(cfg_path, simulator,
                                epcds, epcds_arr, 
                                queries, goal_poses, 
                                obj_centroid, env_bounds, 
                                Niter_RRT)

    # new_trajset = {}
    # for k, obj in enumerate(trajset):

    #     # Set the altitude of all trajectories for this particular object
    #     goal = obj_targets[queries.index(obj)].flatten()
    #     goal_z = goal[2]
    #     goal_z = -1.1 if goal_z > 0 else np.clip(goal_z, -1.1, -1.0)
    #     # print("goal: ", goal)
    #     print(f"Position: {obj_centroid[k]}, Goal X: {goal[0]}, Goal Y: {goal[1]}, Goal Z: {goal_z}")
        
    #     # Set the altitude of all trajectories for this particular object
    #     updated_paths = th.set_RRT_altitude(trajset[obj], goal_z)

    #     # Filter branches
    #     filtered_branches = th.filter_branches(updated_paths)
        
    #     # Replace trajset[obj] with the new list of branches
    #     new_trajset[obj] = filtered_branches

        # for idbr, positions in enumerate(trajset[obj]):


    # line_sets = []
    # for idbr, positions in enumerate(branches):
    #     # Ensure positions is an (M,3) float64 array
    #     pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    #     if len(pts) < 2:
    #         continue  # can’t draw a line with fewer than 2 points

    #     # Create edge list [(0,1), (1,2), …]
    #     edges = [[i, i+1] for i in range(len(pts)-1)]

    #     # Build the LineSet
    #     ls = o3d.geometry.LineSet(
    #         points=o3d.utility.Vector3dVector(pts),
    #         lines=o3d.utility.Vector2iVector(edges)
    #     )

    #     # Optionally color each branch differently
    #     # e.g. use a simple colormap or cycle through a fixed palette
    #     color = np.random.rand(3)  # random RGB
    #     ls.colors = o3d.utility.Vector3dVector([color for _ in edges])

    #     line_sets.append(ls)

    # # 3) Draw everything together
    # o3d.visualization.draw_geometries(
    #     [pcd, *line_sets],
    #     window_name="Point Cloud + Tree Branches",
    #     width=800, height=600
    # )