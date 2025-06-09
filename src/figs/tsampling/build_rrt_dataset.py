import os
import glob
from typing import List

import numpy as np
import torch
import yaml
import imageio
import open3d as o3d
from open3d.visualization import O3DVisualizer

import plotly.io as pio
import plotly.graph_objects as go

from figs.render.gsplat_semantic import GSplat
import figs.scene_editing.scene_editing_utils as scdt
import figs.utilities.trajectory_helper as th
from figs.tsampling.rrt_datagen_v10 import *

def get_objectives(nerf:GSplat, object_names, similarities, viz=False):
    # Transformation from data parser
    world_transform = torch.eye(4)
    scale = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
    transform = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform
    world_transform[:3, :3] = transform[:3, :3]
    world_inv_transform = np.asarray(torch.linalg.inv(world_transform))

    # Get environment point cloud
    env_pcd_dict, env_pcd_array, env_bounds, env_pcd_o3d, env_pcd_mask, env_pcd_attr = scdt.rescale_point_cloud(nerf, viz=viz)

    object_world_positions = []

    for idx, object_name in enumerate(object_names):
        print('*' * 50)
        print(f'Processing Object: {object_name}')
        print('*' * 50)

        # Object detection parameters
        similarity_threshold = similarities[idx][0]
        filter_radius = similarities[idx][1]
        threshold_list = [similarity_threshold] * len(object_names)
        radius_list = [filter_radius] * len(object_names)

        centroid, z_bounds, filtered_pcd, mask, attrs = scdt.get_centroid(
            nerf=nerf,
            env_pcd=env_pcd_o3d,
            pcd_attr=env_pcd_attr,
            positives=object_name,
            negatives='window,wall,floor,ceiling',
            # negatives='q,t,g,r',
            threshold=threshold_list[idx],
            visualize_pcd=False,
            enable_convex_hull=True,
            enable_spherical_filter=True,
            enable_clustering=False,
            filter_radius=radius_list[idx],
            obj_priors={},
            use_Mahalanobis_distance=True
        )

        points = np.asarray(filtered_pcd.points)[mask]
        colors = np.asarray(filtered_pcd.colors)[mask]
        similarity = attrs['raw_similarity'][mask].cpu().numpy()

        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        object_pcd.colors = o3d.utility.Vector3dVector(colors)

        if viz:
            o3d.visualization.draw_plotly([object_pcd])

        # Outlier removal
        object_pcd, _ = object_pcd.remove_radius_outlier(nb_points=30, radius=0.03)

        object_world_positions.append(centroid)

    # Transform centroids to world coordinates
    for i in range(len(object_world_positions)):
        position = object_world_positions[i].reshape(3, -1)
        position /= scale
        position = np.vstack((position, np.ones((1, position.shape[1]))))

        if nerf.name.startswith("sv_"):
            sfm_to_mocap = np.asarray(nerf.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"])
            position = sfm_to_mocap @ world_inv_transform @ position

        position = nerf.T_w2g @ position
        object_world_positions[i] = position[:3, :].T

    return object_world_positions, env_bounds, env_pcd_dict, env_pcd_array

def get_objectives_old(nerf:GSplat, objectives, similarities, viz=False):
    # viz=visualize
    transform = torch.eye(4)
    dataparser_scale = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
    dataparser_transform = nerf.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform
    transform[:3,:3] = dataparser_transform[:3,:3]
    invtransform = np.asarray(torch.linalg.inv(transform))

    epcds, epcds_arr, epcds_bounds, pcd, pcd_mask, pcd_attr = scdt.rescale_point_cloud(nerf)

    # Translate to gemsplat syntax
    positives = objectives
    
    obj_targets = []

    for idx, obj in enumerate(objectives):
        print('*' * 50)
        print(f'Processing Object: {obj}')
        print('*' * 50)

        threshold = similarities[0] #Currently may be overloaded within get_points()
        filter_radius = similarities[1] #Currently may be overloaded within get_points()
        filter_radius = [filter_radius] * len(objectives)

        threshold_obj = [threshold] * len(objectives)

        # source location
        src_centroid, src_z_bounds, scene_pcd, similarity_mask, other_attr = scdt.get_centroid(nerf=nerf,
                                                                                        env_pcd=pcd,
                                                                                        pcd_attr=pcd_attr,
                                                                                        positives=objectives[idx],
                                                                                        negatives='window,wall,floor,ceiling',
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

    # Transform objects from Nerfstudio frame to Mocap frame
    for i in range(len(obj_targets)):
        obj_targets[i] = obj_targets[i].reshape(3, -1)

        obj_targets[i] = obj_targets[i] / dataparser_scale
        obj_targets[i] = np.vstack((obj_targets[i], np.ones((1, obj_targets[i].shape[1]))))

        if nerf.name.startswith("sv_"):
            obj_targets[i] = np.asarray(nerf.transforms_nerf["sfm_to_mocap_T"][0]["sfm_to_mocap_T"]) @ invtransform @ np.asarray(obj_targets[i])
        
        obj_targets[i] = nerf.T_w2g @ obj_targets[i]
        obj_targets[i] = obj_targets[i][:3, :].T

    return obj_targets, epcds_bounds, epcds, epcds_arr

def generate_rrt_paths(
        config_file,
        simulator, 
        pcd, pcd_arr, 
        objectives:List[str], obj_targets, 
        semantic_centroid, env_bounds,
        rings, obstacles, 
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
    all_cylinder_lists = []
    for i, (target, pose, centroid) in enumerate(zip(objectives, obj_targets, semantic_centroid)):
        r1 = config.get('radii')[i][0]
        r2 = config.get('radii')[i][1]

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
            # bounds=ebounds,
            bounds=[
            (config.get('minbound', [None, None])[0], config.get('maxbound', [None, None])[0]),
            (config.get('minbound', [None, None])[1], config.get('maxbound', [None, None])[1])
            ],
            altitude=config.get('altitudes',-1.3)[i],
            # altitude=pose[2],
            dimension=dimension,
            step_size=config.get('step_size', 1.0),
            collision_check_radius=r2,
            goal_exclusion_radius=r1,
            collision_check_resolution=config.get('collision_check_resolution', 0.1),
            max_iter=config.get('N', Niter_RRT),
            exact_step=exact_step,
            bounded_step=bounded_step,
            algorithm=algorithm_input,  # Pass the selected algorithm
            prevent_edge_overlap=prevent_edge_overlap
        )

        # Build RRT
        rrt.build_rrt()
        
        # # Get all leaf nodes in the tree
        leaf_nodes = [node for node in rrt.nodes if not node.children]

        # # Extract paths from each leaf node
        paths = []
        for leaf_node in leaf_nodes:
            path = rrt.get_path_from_leaf_to_root(leaf_node)
            paths.append(path)

        # cylinders = []
        # for branch in paths:
        #     pts2d = np.asarray(branch, dtype=np.float64).reshape(-1,2)
        #     if len(pts2d) < 2: continue
        #     # if simulator.gsplat.name.startswith("sv_"):
        #     #     zcol = np.full((len(pts2d), 1), -1 * pose[2])
        #     # else:
        #     zcol = np.full((len(pts2d), 1), pose[2])
        #     pts3d = np.hstack([pts2d, zcol])

        #     for i in range(len(pts3d)-1):
        #         cyl = create_cylinder_between_points(pts3d[i], pts3d[i+1], radius=0.01)
        #         if cyl is not None:
        #             cyl.paint_uniform_color([0,0.5,1.0])
        #             cylinders.append(cyl)
        cylinders = []
        for leaf in [n for n in rrt.nodes if not n.children]:
            path = rrt.get_path_from_leaf_to_root(leaf)
            pts2d = np.asarray(path, float).reshape(-1,2)
            if len(pts2d)<2: continue
            zcol = np.full((len(pts2d),1), config.get('altitude',-1.1))
            pts3d = np.hstack([pts2d, zcol])
            for i in range(len(pts3d)-1):
                cyl = create_cylinder_between_points(pts3d[i], pts3d[i+1], radius=0.01)
                if cyl:
                    cyl.paint_uniform_color([0,0.5,1.0])
                    cylinders.append(cyl)
        all_cylinder_lists.append(cylinders)

        trajset[target] = paths
        
        pts  = np.asarray(pcd.points)
        cols = np.clip(np.asarray(pcd.colors), 0, 1)
        rgb  = (cols * 255).astype(int)
        rgb_strs = [f"rgb({r},{g},{b})" for r,g,b in rgb]

        # 2) Build the Figure with just the points
        fig = go.Figure(layout=dict(width=1000, height=1000))
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode="markers",
            marker=dict(size=2, color=rgb_strs),
            showlegend=False
        ))

        # # Radius
        # xc, yc = pose[0], pose[1]
        # r1     = config.get("r1")
        # # the Z‐height you want the circle drawn at—e.g. the same as your camera “ground” plane
        # zc     = config.get('altitude',-1.1)    # or use 0 if you want it on the ground
        # θ = np.linspace(0, 2*np.pi, 200)
        # xs = xc + r1 * np.cos(θ)
        # ys = yc + r1 * np.sin(θ)
        # zs = np.full_like(θ, zc)
        # fig.add_trace(go.Scatter3d(
        #     x=xs, y=ys, z=zs,
        #     mode="lines",
        #     line=dict(
        #         color="black",     # or whatever color you like
        #         dash="dash",       # dashed style
        #         width=4
        #     ),
        #     showlegend=False
        # ))
        
        if not config.get('gif'):
            # 3) Now add each cylinder mesh
            for cyl in cylinders:
                verts = np.asarray(cyl.vertices)
                tris  = np.asarray(cyl.triangles)
                # pick the uniform color you used in Open3D: [0, 0.5, 1.0] → rgb(0,128,255)
                cyl_color = 'rgb(0,128,255)'
                fig.add_trace(go.Mesh3d(
                    x=verts[:,0], y=verts[:,1], z=verts[:,2],
                    i=tris[:,0], j=tris[:,1], k=tris[:,2],
                    opacity=1.0,
                    color=cyl_color,
                    showlegend=False
                ))

            # fetch radii from config (with defaults)
            # r2 = float(config.get('r2', 0.5))
            # r1 = float(config.get('r1', 0.0))   # or some default if you prefer
            z_plane = pose[2]                   # or whatever height you want the circles at

            # parameters for circle resolution
            theta = np.linspace(0, 2*np.pi, 200)

            #  centroid
            x2 = pose[0] + r2 * np.cos(theta)
            y2 = pose[1] + r2 * np.sin(theta)
            z2 = np.full_like(theta, z_plane)

            # start node
            x1 = centroid[0] + r1 * np.cos(theta)
            y1 = centroid[1] + r1 * np.sin(theta)
            z1 = np.full_like(theta, z_plane)

            fig.add_trace(go.Scatter3d(
                x=x1, y=y1, z=z1,
                mode='lines',
                line=dict(color='orange', width=4),
                name=f'Object Exclusion Radius={r1}',
                showlegend=False
            ))

            fig.add_trace(go.Scatter3d(
                x=x2, y=y2, z=z2,
                mode='lines',
                line=dict(color='red', width=4),
                name=f'Goal Exclusion Radius={r2}',
                showlegend=False
            ))

            for obstacle, ring_pts in zip(obstacles, rings):
                # flatten in case it's a column‐vector
                ctr = obstacle.flatten()

                # skip empty or bad entries
                if not isinstance(ring_pts, (list, np.ndarray)) or len(ring_pts) == 0:
                    print(f"  → no ring points for centroid {ctr[:2]}; skipping")
                    continue

                # take the very first sample
                first_pt = np.array(ring_pts)[0]

                # compute X–Y distance between that sample and the centroid
                radius = np.linalg.norm(first_pt[:2] - ctr[:2])
                theta  = np.linspace(0, 2*np.pi, 200)

                # build a perfect circle around the centroid
                x_ring = ctr[0] + radius * np.cos(theta)
                y_ring = ctr[1] + radius * np.sin(theta)
                z_ring = np.full_like(theta, ctr[2])   # use centroid's Z

                # add to your Plotly figure
                fig.add_trace(go.Scatter3d(
                    x=x_ring, y=y_ring, z=z_ring,
                    mode='lines',
                    line=dict(color='green', width=4),
                    name=f'Ring Radius={radius:.2f}',
                    showlegend=False
                ))
            
        # # once you have your point cloud `pts` (N×3 array):
        min_pt, max_pt = pts.min(axis=0), pts.max(axis=0)
        center = (min_pt + max_pt) * 0.5

        # 1) axis limits
        # Get axis limits from config file if available
        minbound = [float(x) if x not in [None, 'None'] else None for x in config.get('minbound', [None, None, None])]
        maxbound = [float(x) if x not in [None, 'None'] else None for x in config.get('maxbound', [None, None, None])]
        bounds = np.array([
            [minbound[i] if minbound[i] is not None else pts[:, i].min(),
                maxbound[i] if maxbound[i] is not None else pts[:, i].max()]
            for i in range(3)
        ])
        xmin, xmax = bounds[0]
        ymin, ymax = bounds[1]
        zmin, zmax = bounds[2]
        # zmin,zmax = -5,0
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        # 2) camera (as before)
        R    = np.linalg.norm(pts - pts.mean(axis=0), axis=1).max() * 1.5
        az, el = np.deg2rad(45), np.deg2rad(10)
        eye = dict(
        x=R*np.cos(el)*np.cos(az),
        y=R*np.cos(el)*np.sin(az),
        z=R*np.sin(el)
        )

        # 3) update layout
        if simulator.gsplat.name.startswith("sv_"):
            fig.update_layout(
            scene_camera=dict(eye=eye,
                    up=dict(x=0, y=0, z=1)),
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(x=dx, y=dy, z=dz),
                xaxis=dict(title='x',
                        range=[xmin, xmax], autorange=False,
                showticklabels=True,
                showgrid=False,
                zeroline=False,
                showbackground=False,),
                yaxis=dict(title='y',
                        range=[ymax, ymin], autorange=False,
                showticklabels=True,
                showgrid=False,
                zeroline=False,
                showbackground=False,),
                zaxis=dict(title='z',
                        range=[zmax, zmin], autorange=False,
                showticklabels=True,
                showgrid=False,
                zeroline=False,
                showbackground=False,),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            width=1000, height=1000,
            showlegend=False
            )
        else:
            fig.update_layout(
            scene_camera=dict(eye=eye,
                    up=dict(x=0, y=0, z=1)),
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(x=dx, y=dy, z=dz),
                xaxis=dict(title='',
                        range=[xmin, xmax], autorange=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showbackground=False,),
                yaxis=dict(title='',
                        range=[ymax, ymin], autorange=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showbackground=False,),
                zaxis=dict(title='',
                        range=[zmax, zmin], autorange=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showbackground=False,)
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            width=1000, height=1000,
            showlegend=False
            )

        if not config.get('gif'):
            print("Rendering the figure...")
            fig.show()

        if config.get('gif'):
            # --- 4) Render & save each frame ---
            out_dir = "/home/admin/StanfordMSL/SousVide-Semantic/notebooks/test_space"
            os.makedirs(out_dir, exist_ok=True)

            cam = fig.layout.scene.camera
            off_x = cam.eye.x - center[0]
            off_y = cam.eye.y - center[1]
            off_z = cam.eye.z - center[2]

            # 3) Compute your radius in that XY‐offset plane
            r = np.sqrt(off_x**2 + off_y**2) * 1   # distance from center to camera in XY
            # (keep the same vertical offset)
            z_offset = off_z                  # camera’s height above center

            n_frames = 60
            angles   = np.linspace(0, 360, n_frames, endpoint=False)
            png_paths = []
            # for i, ang in enumerate(angles):
            #     θ = np.deg2rad(ang)
            #     # build the *absolute* eye = center + offset
            #     new_eye = dict(
            #         x = center[0] + r * np.cos(θ),
            #         y = center[1] + r * np.sin(θ),
            #         z = center[2] + z_offset
            #     )
            #     fig.update_layout(scene_camera=dict(eye=new_eye))
            #     path = f"{out_dir}/frame_{i:03d}.png"
            #     fig.write_image(path, width=1000, height=1000)
            #     png_paths.append(path)
            num_trees = len(all_cylinder_lists)
            color_list = ["purple", "orange", "cyan"]  # len == num_trees
            # define at which frames you switch
            # e.g. switch twice means three segments:
            switch_points = [0,
                            n_frames//3,
                            2*n_frames//3,
                            n_frames]  # segments: [0..20),[20..40),[40..60)
            for i, ang in enumerate(np.linspace(0,360,n_frames,endpoint=False)):
                # decide which segment we’re in
                seg = next(j for j in range(len(switch_points)-1)
                        if switch_points[j] <= i < switch_points[j+1])
                cyl_list = all_cylinder_lists[seg]  # select the tree for this segment
                tree_col  = color_list[seg]

                # clear any old cylinder traces
                # (we know trace 0 is the scatter; everything afterward is cylinders)
                while len(fig.data) > 1:
                    fig.data = fig.data[:-1]

                # add the cylinders for *this* tree
                for cyl in cyl_list:
                    verts = np.asarray(cyl.vertices)
                    tris  = np.asarray(cyl.triangles)
                    fig.add_trace(go.Mesh3d(
                        x=verts[:,0], y=verts[:,1], z=verts[:,2],
                        i=tris[:,0], j=tris[:,1], k=tris[:,2],
                        opacity=1.0,
                        color=tree_col,
                        showlegend=False
                    ))

                # rotate camera
                θ = np.deg2rad(ang)
                new_eye = dict(
                    x = center[0] + r * np.cos(θ),
                    y = center[1] + r * np.sin(θ),
                    z = center[2] + z_offset
                )
                fig.update_layout(scene_camera=dict(eye=new_eye))

                # write out frame
                path = f"{out_dir}/frame_{i:03d}.png"
                fig.write_image(path, width=1000, height=1000)
                png_paths.append(path)

            # --- 5) Build the GIF ---
            images = [imageio.imread(p) for p in png_paths]
            # gif_path = "/home/admin/StanfordMSL/SousVide-Semantic/notebooks/test_space/.gif"
            gif_path = f"/home/admin/StanfordMSL/SousVide-Semantic/notebooks/test_space/{simulator.gsplat.name}.gif"
            imageio.mimsave(gif_path, images, duration=0.1)

            # --- 6) Clean up the individual frames ---
            for f in glob.glob(os.path.join(out_dir, "frame_*.png")):
                os.remove(f)

            print(f"Saved spinning GIF to {gif_path}")
        
    return trajset

# def visualize_rrt_trajectories(simulator, scene, viz=False):

#     def load_config_file(base_path, subfolder, filename):
#         config_path = os.path.join(base_path, subfolder)
#         for root, _, files in os.walk(config_path):
#             if filename in files:
#                 with open(os.path.join(root, filename), 'r') as file:
#                     return yaml.safe_load(file), os.path.join(root, filename)
#         raise FileNotFoundError(f"{filename} not found in {config_path}")

#     # Load scene configuration
#     scene_config,cfg_path = load_config_file(simulator.configs_path, "course", f"{scene}.yml")
#     queries = scene_config.get("queries", [])
#     if not queries:
#         raise ValueError("No queries found in the scene.yml file.")
#     radii = [(scene_config.get("r1", 0.5), scene_config.get("r2", 0.5))]
#     Niter_RRT = scene_config.get("N", 1000)

#     obj_targets, env_bounds, epcds, epcds_arr = get_objectives(simulator.gsplat, queries, viz)
    
#     # Obtain goal poses and object centroid
#     goal_poses, obj_centroid = th.process_RRT_objectives(obj_targets, epcds_arr, env_bounds, radii)
#     print(f"obj_centroid: {obj_centroid}")

#     # Generate RRT* Paths
#     trajset = generate_rrt_paths(cfg_path, simulator,
#                                 epcds, epcds_arr, 
#                                 queries, goal_poses, 
#                                 obj_centroid, env_bounds, 
#                                 Niter_RRT)

#     # new_trajset = {}
#     # for k, obj in enumerate(trajset):

#     #     # Set the altitude of all trajectories for this particular object
#     #     goal = obj_targets[queries.index(obj)].flatten()
#     #     goal_z = goal[2]
#     #     goal_z = -1.1 if goal_z > 0 else np.clip(goal_z, -1.1, -1.0)
#     #     # print("goal: ", goal)
#     #     print(f"Position: {obj_centroid[k]}, Goal X: {goal[0]}, Goal Y: {goal[1]}, Goal Z: {goal_z}")
        
#     #     # Set the altitude of all trajectories for this particular object
#     #     updated_paths = th.set_RRT_altitude(trajset[obj], goal_z)

#     #     # Filter branches
#     #     filtered_branches = th.filter_branches(updated_paths)
        
#     #     # Replace trajset[obj] with the new list of branches
#     #     new_trajset[obj] = filtered_branches

#         # for idbr, positions in enumerate(trajset[obj]):


#     # line_sets = []
#     # for idbr, positions in enumerate(branches):
#     #     # Ensure positions is an (M,3) float64 array
#     #     pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
#     #     if len(pts) < 2:
#     #         continue  # can’t draw a line with fewer than 2 points

#     #     # Create edge list [(0,1), (1,2), …]
#     #     edges = [[i, i+1] for i in range(len(pts)-1)]

#     #     # Build the LineSet
#     #     ls = o3d.geometry.LineSet(
#     #         points=o3d.utility.Vector3dVector(pts),
#     #         lines=o3d.utility.Vector2iVector(edges)
#     #     )

#     #     # Optionally color each branch differently
#     #     # e.g. use a simple colormap or cycle through a fixed palette
#     #     color = np.random.rand(3)  # random RGB
#     #     ls.colors = o3d.utility.Vector3dVector([color for _ in edges])

#     #     line_sets.append(ls)

#     # # 3) Draw everything together
#     # o3d.visualization.draw_geometries(
#     #     [pcd, *line_sets],
#     #     window_name="Point Cloud + Tree Branches",
#     #     width=800, height=600
#     # )
        
def visualize_rrt_trajectories(trajset,
                               config_file,
                               simulator, 
                               pcd, pcd_arr,
                               objectives:List[str], obj_targets,
                               semantic_centroid, env_bounds,
                               rings, obstacles
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

    config = {}
    # Check if the configuration file exists
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file) or {}
        print("Configuration loaded from config.yml")
    else:
        print("Configuration file not found. Proceeding with interactive inputs.")
    
    all_cylinder_lists = []
    for i, (target, pose, centroid, ring_pts, obstacle) in enumerate(zip(objectives, obj_targets, semantic_centroid, rings, obstacles)):
        r1 = config.get('radii')[i][0]
        r2 = config.get('radii')[i][1]
        print(f"target: {target}")
        print(f"trajset[target] size: {len(trajset[target])}")

        cylinders = []
        for path in trajset[target]:
            pts3d = np.asarray(path, float).reshape(-1, 3)
            if len(pts3d) < 2:
                continue

            for i in range(len(pts3d) - 1):
                p0 = pts3d[i]
                p1 = pts3d[i + 1]
                cyl = create_cylinder_between_points(p0, p1, radius=0.01)
                if cyl:
                    cyl.paint_uniform_color([0, 0.5, 1.0])
                    cylinders.append(cyl)

            all_cylinder_lists.append(cylinders)
        
        print(f"Number of cylinder lists: {len(all_cylinder_lists)}")
        pts  = np.asarray(pcd.points)
        cols = np.clip(np.asarray(pcd.colors), 0, 1)
        rgb  = (cols * 255).astype(int)
        rgb_strs = [f"rgb({r},{g},{b})" for r,g,b in rgb]

        # 2) Build the Figure with just the points
        fig = go.Figure(layout=dict(width=1000, height=1000))
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode="markers",
            marker=dict(size=2, color=rgb_strs),
            showlegend=False
        ))

        # # Radius
        # xc, yc = pose[0], pose[1]
        # r1     = config.get("r1")
        # # the Z‐height you want the circle drawn at—e.g. the same as your camera “ground” plane
        # zc     = config.get('altitude',-1.1)    # or use 0 if you want it on the ground
        # θ = np.linspace(0, 2*np.pi, 200)
        # xs = xc + r1 * np.cos(θ)
        # ys = yc + r1 * np.sin(θ)
        # zs = np.full_like(θ, zc)
        # fig.add_trace(go.Scatter3d(
        #     x=xs, y=ys, z=zs,
        #     mode="lines",
        #     line=dict(
        #         color="black",     # or whatever color you like
        #         dash="dash",       # dashed style
        #         width=4
        #     ),
        #     showlegend=False
        # ))
        
        if not config.get('gif'):
            # 3) Now add each cylinder mesh
            for cyl in cylinders:
                verts = np.asarray(cyl.vertices)
                tris  = np.asarray(cyl.triangles)
                # pick the uniform color you used in Open3D: [0, 0.5, 1.0] → rgb(0,128,255)
                cyl_color = 'rgb(0,128,255)'
                fig.add_trace(go.Mesh3d(
                    x=verts[:,0], y=verts[:,1], z=verts[:,2],
                    i=tris[:,0], j=tris[:,1], k=tris[:,2],
                    opacity=1.0,
                    color=cyl_color,
                    showlegend=False
                ))

            # fetch radii from config (with defaults)
            # r2 = float(config.get('r2', 0.5))
            # r1 = float(config.get('r1', 0.0))   # or some default if you prefer
            z_plane = pose[2]                   # or whatever height you want the circles at

            # parameters for circle resolution
            theta = np.linspace(0, 2*np.pi, 200)

            #  centroid
            x2 = pose[0] + r2 * np.cos(theta)
            y2 = pose[1] + r2 * np.sin(theta)
            z2 = np.full_like(theta, z_plane)

            # start node
            x1 = centroid[0] + r1 * np.cos(theta)
            y1 = centroid[1] + r1 * np.sin(theta)
            z1 = np.full_like(theta, z_plane)

            fig.add_trace(go.Scatter3d(
                x=x1, y=y1, z=z1,
                mode='lines',
                line=dict(color='orange', width=4),
                name=f'Object Exclusion Radius={r1}',
                showlegend=False
            ))

            fig.add_trace(go.Scatter3d(
                x=x2, y=y2, z=z2,
                mode='lines',
                line=dict(color='red', width=4),
                name=f'Goal Exclusion Radius={r2}',
                showlegend=False
            ))

            for obstacle, ring_pts in zip(obstacles, rings):
                # flatten in case it's a column‐vector
                ctr = obstacle.flatten()

                # skip empty or bad entries
                if not isinstance(ring_pts, (list, np.ndarray)) or len(ring_pts) == 0:
                    print(f"  → no ring points for centroid {ctr[:2]}; skipping")
                    continue

                # take the very first sample
                first_pt = np.array(ring_pts)[0]

                # compute X–Y distance between that sample and the centroid
                radius = np.linalg.norm(first_pt[:2] - ctr[:2])
                theta  = np.linspace(0, 2*np.pi, 200)

                # build a perfect circle around the centroid
                x_ring = ctr[0] + radius * np.cos(theta)
                y_ring = ctr[1] + radius * np.sin(theta)
                z_ring = np.full_like(theta, ctr[2])   # use centroid's Z

                # add to your Plotly figure
                fig.add_trace(go.Scatter3d(
                    x=x_ring, y=y_ring, z=z_ring,
                    mode='lines',
                    line=dict(color='green', width=4),
                    name=f'Ring Radius={radius:.2f}',
                    showlegend=False
                ))
            
        # # once you have your point cloud `pts` (N×3 array):
        min_pt, max_pt = pts.min(axis=0), pts.max(axis=0)
        center = (min_pt + max_pt) * 0.5

        # 1) axis limits
        # Get axis limits from config file if available
        minbound = [float(x) if x not in [None, 'None'] else None for x in config.get('minbound', [None, None, None])]
        maxbound = [float(x) if x not in [None, 'None'] else None for x in config.get('maxbound', [None, None, None])]
        bounds = np.array([
            [minbound[i] if minbound[i] is not None else pts[:, i].min(),
                maxbound[i] if maxbound[i] is not None else pts[:, i].max()]
            for i in range(3)
        ])
        xmin, xmax = bounds[0]
        ymin, ymax = bounds[1]
        zmin, zmax = bounds[2]
        # zmin,zmax = -5,0
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        # 2) camera (as before)
        R    = np.linalg.norm(pts - pts.mean(axis=0), axis=1).max() * 1.5
        az, el = np.deg2rad(45), np.deg2rad(10)
        eye = dict(
        x=R*np.cos(el)*np.cos(az),
        y=R*np.cos(el)*np.sin(az),
        z=R*np.sin(el)
        )

        # 3) update layout
        if simulator.gsplat.name.startswith("sv_"):
            fig.update_layout(
            scene_camera=dict(eye=eye,
                    up=dict(x=0, y=0, z=1)),
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(x=dx, y=dy, z=dz),
                xaxis=dict(title='x',
                        range=[xmin, xmax], autorange=False,
                showticklabels=True,
                showgrid=False,
                zeroline=False,
                showbackground=False,),
                yaxis=dict(title='y',
                        range=[ymax, ymin], autorange=False,
                showticklabels=True,
                showgrid=False,
                zeroline=False,
                showbackground=False,),
                zaxis=dict(title='z',
                        range=[zmax, zmin], autorange=False,
                showticklabels=True,
                showgrid=False,
                zeroline=False,
                showbackground=False,),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            width=1000, height=1000,
            showlegend=False
            )
        else:
            fig.update_layout(
            scene_camera=dict(eye=eye,
                    up=dict(x=0, y=0, z=1)),
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(x=dx, y=dy, z=dz),
                xaxis=dict(title='',
                        range=[xmin, xmax], autorange=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showbackground=False,),
                yaxis=dict(title='',
                        range=[ymax, ymin], autorange=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showbackground=False,),
                zaxis=dict(title='',
                        range=[zmax, zmin], autorange=False,
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showbackground=False,)
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            width=1000, height=1000,
            showlegend=False
            )

        if not config.get('gif'):
            print("Rendering the figure...")
            fig.show()

        if config.get('gif'):
            # --- 4) Render & save each frame ---
            out_dir = "/home/admin/StanfordMSL/SousVide-Semantic/notebooks/test_space"
            os.makedirs(out_dir, exist_ok=True)

            cam = fig.layout.scene.camera
            off_x = cam.eye.x - center[0]
            off_y = cam.eye.y - center[1]
            off_z = cam.eye.z - center[2]

            # 3) Compute your radius in that XY‐offset plane
            r = np.sqrt(off_x**2 + off_y**2) * 1   # distance from center to camera in XY
            # (keep the same vertical offset)
            z_offset = off_z                  # camera’s height above center

            n_frames = 60
            angles   = np.linspace(0, 360, n_frames, endpoint=False)
            png_paths = []
            # for i, ang in enumerate(angles):
            #     θ = np.deg2rad(ang)
            #     # build the *absolute* eye = center + offset
            #     new_eye = dict(
            #         x = center[0] + r * np.cos(θ),
            #         y = center[1] + r * np.sin(θ),
            #         z = center[2] + z_offset
            #     )
            #     fig.update_layout(scene_camera=dict(eye=new_eye))
            #     path = f"{out_dir}/frame_{i:03d}.png"
            #     fig.write_image(path, width=1000, height=1000)
            #     png_paths.append(path)
            num_trees = len(all_cylinder_lists)
            color_list = ["purple", "orange", "cyan"]  # len == num_trees
            # define at which frames you switch
            # e.g. switch twice means three segments:
            switch_points = [0,
                            n_frames//3,
                            2*n_frames//3,
                            n_frames]  # segments: [0..20),[20..40),[40..60)
            for i, ang in enumerate(np.linspace(0,360,n_frames,endpoint=False)):
                # decide which segment we’re in
                seg = next(j for j in range(len(switch_points)-1)
                        if switch_points[j] <= i < switch_points[j+1])
                cyl_list = all_cylinder_lists[seg]  # select the tree for this segment
                tree_col  = color_list[seg]

                # clear any old cylinder traces
                # (we know trace 0 is the scatter; everything afterward is cylinders)
                while len(fig.data) > 1:
                    fig.data = fig.data[:-1]

                # add the cylinders for *this* tree
                for cyl in cyl_list:
                    verts = np.asarray(cyl.vertices)
                    tris  = np.asarray(cyl.triangles)
                    fig.add_trace(go.Mesh3d(
                        x=verts[:,0], y=verts[:,1], z=verts[:,2],
                        i=tris[:,0], j=tris[:,1], k=tris[:,2],
                        opacity=1.0,
                        color=tree_col,
                        showlegend=False
                    ))

                # rotate camera
                θ = np.deg2rad(ang)
                new_eye = dict(
                    x = center[0] + r * np.cos(θ),
                    y = center[1] + r * np.sin(θ),
                    z = center[2] + z_offset
                )
                fig.update_layout(scene_camera=dict(eye=new_eye))

                # write out frame
                path = f"{out_dir}/frame_{i:03d}.png"
                fig.write_image(path, width=1000, height=1000)
                png_paths.append(path)

            # --- 5) Build the GIF ---
            images = [imageio.imread(p) for p in png_paths]
            # gif_path = "/home/admin/StanfordMSL/SousVide-Semantic/notebooks/test_space/.gif"
            gif_path = f"/home/admin/StanfordMSL/SousVide-Semantic/notebooks/test_space/{simulator.gsplat.name}.gif"
            imageio.mimsave(gif_path, images, duration=0.1)

            # --- 6) Clean up the individual frames ---
            for f in glob.glob(os.path.join(out_dir, "frame_*.png")):
                os.remove(f)

            print(f"Saved spinning GIF to {gif_path}")
        
    return trajset