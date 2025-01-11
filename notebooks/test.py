import open3d as o3d

# Load the point cloud
point_cloud = o3d.io.read_point_cloud("../gsplats/workspace/scene003/sparse_pc.ply")
# point_cloud = o3d.io.read_point_cloud("../gsplats/workspace/scene004/sparse_pc.ply")

vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the point cloud to the visualizer
vis.add_geometry(point_cloud)

# Get the view control object
view_ctl = vis.get_view_control()

# Set the view control parameters
view_ctl.set_zoom(0.1)
view_ctl.set_front([ 0,-1, 0])
view_ctl.set_up([ 0, 0, 1])
view_ctl.set_lookat([0,5,2])

vis.run()