import os
import yaml
import argparse
import numpy as np
import random
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

class Node:
    def __init__(self, position, parent=None):
        self.position = position  # NumPy array
        self.parent = parent      # Parent node
        self.children = []        # List of child nodes
        self.cost = 0.0           # Cost from the start node

class RRT:
    def __init__(self, env_arr, env_pts, start, obj, bounds, altitude, algorithm='RRT', dimension=2, step_size=1.0, max_iter=10000,
                  collision_check_radius=0.5, goal_exclusion_radius=0.5, collision_check_resolution=0.1, exact_step=False, bounded_step = False, prevent_edge_overlap=False):
        self.env_pts = env_pts
        self.env_arr = env_arr
        # print(f"Environment Points: {self.env_arr.shape}")
        if self.env_pts is not None:
            self.obstacle_kdtree = cKDTree(self.env_arr.T)
        else:
            self.obstacle_kdtree = None
        self.algorithm = algorithm
        self.dimension = dimension
        self.exact_step = exact_step
        self.bounded_step = bounded_step
        self.prevent_edge_overlap = prevent_edge_overlap
        self.goal_node = Node(np.array(start))
        self.bounds = bounds          # Bounds of the environment: List of (min, max) tuples for each dimension
        self.obj_exclusion_radius = goal_exclusion_radius  # Set the exclusion radius for the object node
        self.obj_exclusion = (np.array(obj), self.obj_exclusion_radius)
        print(f"Object Exclusion: {self.obj_exclusion}")
        # self.obj_exclusion_radius = 1.4  # Set the exclusion radius for the object node
        # self.obj_exclusion = (np.array(obj), self.obj_exclusion_radius)
        # print(f"Object Exclusion: {self.obj_exclusion}")
        self.goal_exclusion_radius = collision_check_radius  # Set the exclusion radius for the goal node
        self.goal_exclusion = (self.goal_node.position, self.goal_exclusion_radius)
        self.step_size = step_size
        self.collision_check_resolution = collision_check_resolution
        self.max_iter = max_iter
        self.nodes = [self.goal_node]     # List to store all nodes
        self.sampled_points = []  # List to store sampled points

        self.altitude = altitude
        self.exclusion_radius = 0.45  # Determines how close the camera can get to any point in the environment
        self.min_edge_separation = 0.1  # Adjust this value as needed
    
    def sample_free(self):
        # Sample a random point within the bounds
        point = np.array([
            random.uniform(self.bounds[d][0], self.bounds[d][1])
            for d in range(self.dimension)
        ])
        return point
    
    def is_within_bounds(self, point):
        # Check if the point is within the environment bounds
        return all([
            self.bounds[d][0] <= point[d] <= self.bounds[d][1]
            for d in range(self.dimension)
        ])

    def is_within_goal_exclusion(self, point):
        # Check if the point is within the goal exclusion zone
        center, radius = self.goal_exclusion
        return np.linalg.norm(point - center) <= radius
    
    def is_within_obj_exclusion(self, point):
        # Check if the point is within the object exclusion zone
        center, radius = self.obj_exclusion
        return np.linalg.norm(point - center) <= radius

    def at_edge(self, point):
        # Check if the point is at the edge of the environment
        return any([
            np.isclose(point[d], self.bounds[d][0]) or np.isclose(point[d], self.bounds[d][1])
            for d in range(self.dimension)
        ])
    
    def obstacle_collision(self, point):
        pose_xyz = np.hstack((point, self.altitude))  # Shape: (3,)
        
        if self.exclusion_radius is not None:
            # Use KD-tree to find indices of obstacle points within exclusion_radius of pose_xyz
            idx = self.obstacle_kdtree.query_ball_point(pose_xyz, r=self.exclusion_radius, eps=0.05, workers=-1)
            
            # Check if any obstacle points are within the exclusion radius
            if len(idx) == 0:
                return False  # No Collision
            else:
                return True   # Collision
        else:
            return False
        
    def is_collision_free(self, from_point, to_point):
        # Check for collisions along the path from from_point to to_point
        distance = np.linalg.norm(to_point - from_point)
        num_samples = int(np.ceil(distance / self.collision_check_resolution))
        t_values = np.linspace(0, 1, num_samples)
        for t in t_values:
            point = from_point + t * (to_point - from_point)
            if self.obstacle_collision(point):
                return False
        return True

    def nearest(self, point):
        # Find the nearest node in the tree to the sampled point
        node_positions = np.array([node.position for node in self.nodes])
        tree = cKDTree(node_positions)
        _, idx = tree.query(point)
        return self.nodes[idx]
    
    def steer(self, from_node, to_point):
        # Move from from_node towards to_point by step_size
        direction = to_point - from_node.position
        length = np.linalg.norm(direction)
        if length == 0:
            return from_node.position
        
        if self.exact_step:
            # Make the new position exactly step_size away
            direction = (direction / length) * self.step_size
        elif self.bounded_step:
            # Make the new position fall within a range of values min=0.75, max = step_size
            step_size = max(0.75, min(self.step_size, length))
            direction = (direction / length) * step_size
        else:
            # Make the new position at most step_size away
            direction = (direction / length) * min(self.step_size, length)
        
        new_position = from_node.position + direction
        return new_position
    
    def update_descendant_costs(self, node):
        # Recursively update the cost of the node and its descendants based on euclidean distance
        for child in node.children:
            # Update the cost of the child
            child.cost = node.cost + np.linalg.norm(child.position - node.position)
            # Continue updating costs for the descendants of the child
            self.update_descendant_costs(child)

    def build_rrt(self):
        node_positions = np.array([node.position for node in self.nodes])
        tree = cKDTree(node_positions)

        # Define the search radius for RRT*
        if self.algorithm == 'RRT*':
            # Radius can be tuned based on the space size and number of nodes
            gamma_rrt_star = 16 * (1 + 1 / self.dimension)**(1 / self.dimension)
            radius = min(gamma_rrt_star * (np.log(len(self.nodes)) / len(self.nodes))**(1 / self.dimension), self.step_size)
        else:
            radius = 0  # Not used in RRT

        rewire_count = 0  # Initialize rewiring counter
        total_distance_saved = 0.0  # Counter for total distance saved

        for i in range(self.max_iter):
            # Rebuild KD-tree periodically for efficiency
            if i % 1 == 0:
                node_positions = np.array([node.position for node in self.nodes])
                tree = cKDTree(node_positions)
                if self.algorithm == 'RRT*':
                    radius = gamma_rrt_star * (np.log(len(self.nodes)) / len(self.nodes))**(1 / self.dimension)

            rnd_point = self.sample_free()
            self.sampled_points.append(rnd_point)  # Store the sampled point

            nearest_node = self.nearest(rnd_point)
            new_position = self.steer(nearest_node, rnd_point)

            if not self.is_within_bounds(new_position):
                print(f"Sampled point {rnd_point} is out of bounds.")
                continue  # Skip if out of bounds

            if self.is_within_goal_exclusion(new_position):
                print(f"Sampled point {new_position} is within goal exclusion zone.")
                continue # Skip if within goal exclusion zone

            if self.is_within_obj_exclusion(new_position):
                print(f"Sampled point {new_position} is within object exclusion zone.")
                continue # Skip if within object exclusion zone

            if self.obstacle_collision(new_position):
                print(f"Sampled point {new_position} is in collision with an obstacle.")
                continue  # Skip if collision detected

            # Initialize the new node's cost
            new_node_cost = nearest_node.cost + np.linalg.norm(new_position - nearest_node.position)

            # RRT* specific: Find the best parent from nearby nodes
            if self.algorithm == 'RRT*':
                # Query nodes within the search radius
                indices = tree.query_ball_point(new_position, r=radius)
                min_cost = new_node_cost
                best_parent = nearest_node

                for idx in indices:
                    candidate_node = self.nodes[idx]
                    tentative_cost = candidate_node.cost + np.linalg.norm(new_position - candidate_node.position)
                    if self.is_collision_free(candidate_node.position, new_position) and tentative_cost < min_cost:
                        min_cost = tentative_cost
                        best_parent = candidate_node

                # Update the new node's cost and parent
                new_node_cost = min_cost
                new_node = Node(new_position, parent=best_parent)
                new_node.cost = new_node_cost
                best_parent.children.append(new_node)
                self.nodes.append(new_node)
            else:
                # RRT: Use nearest node as parent
                new_node = Node(new_position, parent=nearest_node)
                nearest_node.children.append(new_node)
                self.nodes.append(new_node)

            # RRT* specific: Rewire the tree
            if self.algorithm == 'RRT*':
                # Query nodes within the search radius to potentially rewire
                indices = tree.query_ball_point(new_position, r=radius)
                for idx in indices:
                    candidate_node = self.nodes[idx]
                    if candidate_node is best_parent:
                        continue  # Skip the parent
                    tentative_cost = new_node.cost + np.linalg.norm(candidate_node.position - new_position)
                    if tentative_cost < candidate_node.cost and self.is_collision_free(new_position, candidate_node.position):
                        # Rewire: Change parent to new_node
                        old_parent = candidate_node.parent
                        if old_parent:
                            try:
                                old_parent.children.remove(candidate_node)
                            except ValueError:
                                pass  # Node already removed from parent's children
                        candidate_node.parent = new_node
                        candidate_node.cost = tentative_cost
                        new_node.children.append(candidate_node)

                        # **Update costs of descendants**
                        self.update_descendant_costs(candidate_node)

                        # **Calculate distance saved**
                        distance_saved = (old_parent.cost + np.linalg.norm(candidate_node.position - old_parent.position)) - candidate_node.cost
                        total_distance_saved += distance_saved
                        
                        rewire_count += 1

            # Stop expanding if the new node is at the edge
            if self.at_edge(new_position):
                continue  # You can choose to stop expanding in this direction

        print("RRT/RRT* construction completed.")
        print(f"Total Nodes Added: {len(self.nodes)}")
        print(f"Total Rewiring Actions: {rewire_count}")
        print(f"Total Distance Saved: {total_distance_saved:.2f} units")

    def get_path_from_leaf_to_root(self, leaf_node):
        path = []
        current_node = leaf_node
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        # Reverse the path to go from root to leaf
        path.reverse()
        return path

    def plot_tree(self):
        plt.figure(figsize=(16, 16))
        
        # Plot each edge in the tree (i.e., between a node and its parent)
        for node in self.nodes:
            if node.parent is not None:
                x_vals = [node.parent.position[0], node.position[0]]
                y_vals = [node.parent.position[1], node.position[1]]
                plt.plot(x_vals, y_vals, 'blue', zorder=1)

        # Plot start node
        start_node = self.nodes[0]
        plt.scatter(start_node.position[0], start_node.position[1], c='red', s=50, label='Start Node', zorder=2)

        # Plot object node
        obj_node = self.obj_exclusion[0]
        print(f"Object Node: {obj_node}")
        plt.scatter(obj_node[0], obj_node[1], c='yellow', s=50, label='Object Node', zorder=2)
                
        # Plot leaf nodes
        leaves = [node for node in self.nodes if not node.children]
        x_leaves = [node.position[0] for node in leaves]
        y_leaves = [node.position[1] for node in leaves]
        plt.scatter(x_leaves, y_leaves, c='orange', s=20, label='Leaf Nodes', zorder=2)

        # Plot intermediate nodes (exclude leaf nodes)
        intermediate_nodes = [node for node in self.nodes if node.children and node.parent]
        x_intermediate = [node.position[0] for node in intermediate_nodes]
        y_intermediate = [node.position[1] for node in intermediate_nodes]
        plt.scatter(x_intermediate, y_intermediate, c='blue', s=10, label='Intermediate Nodes', zorder=2)

        # Plot exclusion radius for the object node
        obj_circle = plt.Circle(obj_node, self.obj_exclusion_radius, color='orange', fill=False, linestyle='--', label='Object Exclusion Radius', zorder=1)
        plt.gca().add_patch(obj_circle)

        # Plot exclusion radius for the start node
        start_circle = plt.Circle(start_node.position, self.goal_exclusion_radius, color='red', fill=False, linestyle='--', label='Start Exclusion Radius', zorder=1)
        plt.gca().add_patch(start_circle)

        plt.title("RRT/RRT* Tree")
        plt.xlabel("X")
        plt.ylabel("Y")
        # plt.axis("equal")
        plt.grid(True)
        plt.xlim(min(self.bounds[0][0], obj_node[0]), max(self.bounds[0][1], obj_node[0]))
        plt.ylim(max(self.bounds[1][1], obj_node[1]), min(self.bounds[1][0], obj_node[1]))
        plt.gca().set_aspect(1, adjustable='box')

        # plt.ylim(self.bounds[1])
        plt.legend()
        plt.show()