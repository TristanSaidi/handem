import os
import time

import numpy as np
import torch
from isaacgym import gymtorch
from torch.utils.tensorboard import SummaryWriter
import pickle
from .ihm_base import IHMBase
from termcolor import cprint

from isaacgym.torch_utils import tensor_clamp, torch_rand_float, quat_conjugate, quat_mul
from handem.utils.torch_jit_utils import quat_to_angle_axis, my_quat_rotate
from handem.utils.misc import sample_random_quat, compute_quat_angle, get_euler_disps

class RRT(IHMBase):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        # setup reorientation axis
        self.configure_rotation_axis()
        # setup saving
        self.configure_output()
        # setup search space constraints
        self.configure_constraints()
        # setup RRT parameters
        self.configure_rrt_parameters()
        # setup parallelization
        self.setup_parallel(self.cfg["RRT"]["samples_per_iter"])
        # configure RRT data structures
        self.init_rrt_data_structures()

        self.saved_num_nodes = 0
        self.node_distances_to_root = [0.0]
        self.n_iter = 0
        self.time_start = time.time()
    
    def configure_rotation_axis(self):
        """
        set up rotation axis for object orientation
        0: x-axis
        1: y-axis
        2: z-axis
        3: all axes (arbitrary reorientation)
        """
        self.tree_type_map = [
            "x",
            "y",
            "z",
            "all"
        ]

        self.axis = self.cfg["RRT"]["axis"]
        assert self.axis in [2, 3], "Invalid rotation axis" # x, y axis not supported yet
        # keep track of type of tree for saving and loading
        self.type = self.tree_type_map[self.axis]

        if self.type == "all":
            # abitrary reorientation
            self.rotation_axis = None
            return
        self.rotation_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self.rotation_axis[:, self.axis] = 1

    def configure_output(self):
        self.save_file = self.cfg["RRT"]["save_file"]

    def configure_constraints(self):
        # sample-space constraints. This implementation allows for 
        # smoother sampling without restricting object orientation

        self.sample_space_max = torch.cat(
            [self.hand_dof_upper_limits, self.obj_xyz_upper_lim],
        )
        self.sample_space_min = torch.cat(
            [self.hand_dof_lower_limits, self.obj_xyz_lower_lim],
        )
        self.sample_space_dimension = self.sample_space_max.size(0)
        self.obj_tilt_lim_cos = self.cfg["env"]["reset"]["obj_tilt_lim_cos"]

    def configure_rrt_parameters(self):
        self.max_nodes = self.cfg["RRT"]["max_nodes"]
        self.action_scale = self.cfg["RRT"]["action_scale"]
        self.stability_check_duration = self.cfg["RRT"]["stability_check_duration"]
        self.feedback_state = self.cfg["env"]["feedbackState"]

    def init_rrt_data_structures(self):
        """
        There are three main datastrutures that store information about the RRT tree:

        1. self.state_space_nodes: tensor storing the state (as defined in RRT.yaml) for each node of the tree
        2. self.sample_space_nodes: tensor storing the coordinates of each tree node in the search space (hand joint pos, object pos, object orientation)
        3. self.sim_states: tensor storing the simulator state for each node of the tree (hand joint pos, object pos, object orientation, velocities)

        """
        # dimensions for each possible field of sim state
        dims = {
            "hand_joint_pos": 15,
            "hand_joint_vel": 15,
            "hand_joint_target": 15,
            "object_pos": 3,
            "object_orientation": 4,
            "object_lin_vel": 3,
            "object_ang_vel": 3,
            "ftip_contact_force": 15,
            "hand_joint_torque": 15,
            "ftip_contact_bool": 5,
            "ftip_contact_pos": 15,
        }

        # create sim state map for parsing states
        s = 0
        self.state_space_map = {}
        for key in self.feedback_state:
            self.state_space_map[key] = (s, s + dims[key])
            s += dims[key]
        self.ftip_contact_pos, contact_vectors = self.compute_ftip_contact_pos_vec()
        self.contact_bool_tensor = self.compute_contact_bool()

        feedback = {
            "hand_joint_pos": self.default_hand_joint_pos,
            "hand_joint_vel": torch.tensor([0] * 15, device=self.device),
            "hand_joint_target": self.default_hand_joint_pos,
            "object_pos": self.default_object_pos,
            "object_orientation": self.default_object_quat,
            "object_lin_vel": torch.tensor([0] * 3, device=self.device),
            "object_ang_vel": torch.tensor([0] * 3, device=self.device),
            "ftip_contact_force": torch.tensor([0]*15, device=self.device),
            "hand_joint_torque": self.hand_torques[0],
            "ftip_contact_bool": self.contact_bool_tensor[0],
            "ftip_contact_pos": torch.cat(self.ftip_contact_pos, dim=1)[0],
        }

        state = {key: feedback[key].unsqueeze(0) for key in self.feedback_state}
        
        ########## State space ##########
        self.state_space_nodes = torch.cat(list(state.values()), dim=-1)
        ########## State space ##########

        print("RRT state space buffer shape:", self.state_space_nodes.size())

        state_space_dimension = self.state_space_nodes.size(1)
        print("Dimensionality of RRT state space: ", state_space_dimension)
        
        self.sample_space_map = {
            "hand_joint_pos": (0, 15),
            "object_pos": (15, 18),
            "object_orientation": (18, 22),
        }
        ########## Sample space ##########
        self.sample_space_nodes = torch.cat(
            [
                self.default_hand_joint_pos,
                self.default_object_pos,
                self.default_object_quat
            ]
        ).unsqueeze(0)
        ########## Sample space ##########

        self.sample_dimension = self.sample_space_nodes.size(1)
        print("Dimensionality of RRT sample-space: ", self.sample_dimension)

        ########## Simulator state ##########
        self.sim_states = torch.cat(
            [
                self.default_hand_joint_pos,
                self.default_hand_joint_pos,
                self.default_object_pos,
                self.default_object_quat,
                torch.tensor([0] * 6, device=self.device),
            ]
        ).unsqueeze(0)
        ########## Simulator state ##########

        # parents and children are stored as indices
        self.rrt_parent_indices = [None]
        self.rrt_children_indices = [[]]
        # actions taken to reach this node from parent
        self.rrt_actions = torch.zeros((1, 15), device=self.device)

    def setup_parallel(self, samples_per_iter):
        self.samples_per_iter = samples_per_iter
        assert self.num_envs % self.samples_per_iter == 0
        self.envs_per_sample = self.num_envs // self.samples_per_iter
        self.nearest_idx = [0] * self.samples_per_iter

    def get_states(self, feedback=None):
        """ Constructs state tensor based on fields requested in configs """
        self.ftip_contact_pos, contact_vectors = self.compute_ftip_contact_pos_vec()
        self.contact_bool_tensor = self.compute_contact_bool()

        feedback = {
            "hand_joint_pos": self.hand_dof_pos,
            "hand_joint_vel": self.hand_dof_vel,
            "hand_joint_target": self.target_hand_joint_pos,
            "object_pos": self.object_pos,
            "object_orientation": self.object_orientation,
            "object_lin_vel": self.object_lin_vel,
            "object_ang_vel": self.object_ang_vel,
            "ftip_contact_force": torch.cat(self.ftip_contact_force, dim=1),
            "hand_joint_torque": self.hand_torques,
            "ftip_contact_bool": self.contact_bool_tensor,
            "ftip_contact_pos": torch.cat(self.ftip_contact_pos, dim=1),
        }

        state = {key: feedback[key] for key in self.feedback_state}
        state_tensor = torch.cat(list(state.values()), dim=-1)
        return state_tensor

    def pre_physics_step(self, actions: torch.Tensor):
        return super().pre_physics_step(actions)
    
    def post_physics_step(self):
        self._refresh_tensors()
        self.progress_buf += 1
        if self.progress_buf[0] >= self.stability_check_duration:
            self.reset()
            self.progress_buf[:] = 0

    def check_constraints(self):
        """check sample space constraints"""
        s_space_current = torch.cat(
            [self.hand_dof_pos, self.object_pose[:, 0:3]], 
            dim=1
        )
        invalid = torch.zeros_like(self.reset_buf)

        # check sample_space constraints
        for current_dof in range(s_space_current.size(1)):
            # check upper limit for this dof
            invalid = torch.where(
                s_space_current[:,current_dof] > self.sample_space_max[current_dof],
                torch.ones_like(invalid),
                invalid
            )
            # check lower limit for this dof
            invalid = torch.where(
                s_space_current[:,current_dof] < self.sample_space_min[current_dof],
                torch.ones_like(invalid),
                invalid
            )

        if self.type == "all":
            return invalid
        current_orientations = self.object_pose[:,3:7]
        object_axis = my_quat_rotate(current_orientations, self.rotation_axis)
        invalid = torch.where(
            torch.linalg.norm(object_axis * self.rotation_axis, dim=1) < self.obj_tilt_lim_cos,
            torch.ones_like(invalid),
            invalid
        )
        return invalid
        
    def collect_states_update_tree(self):
        invalid = self.check_constraints()
        self.n_iter += 1
        # store new state for all envs
        new_states = self.get_states()
        # create simulator state tensor
        simulator_states = torch.cat(
            [
                self.hand_dof_pos,
                self.target_hand_joint_pos, 
                self.object_state
            ],
            dim=1
        )
        if self.n_iter > 1:
            for i in range(self.samples_per_iter):
                batch = range(
                    self.envs_per_sample * i,
                    self.envs_per_sample * (i + 1)
                )
                # fetch batch of states
                new_states_batch = new_states[batch]
                simulator_states_batch = simulator_states[batch]
                # fetch batch of sampled actions
                sampled_actions_batch = self.sampled_actions[batch]

                distances = self.compute_distance(
                    node=self.samples[i],
                    node_type="sample_space_node",
                    node_set=new_states_batch,
                    node_set_type="state_space_node",
                )
                invalid_before = invalid[batch].nonzero()
                if self.type == "all": # for arbitrary reorientation, check for large displacement jumps
                    s,e = self.state_space_map["object_orientation"]
                    new_quats = new_states_batch[:, s:e].view(-1, 4)
                    nearest_node_quat = self.state_space_nodes[self.nearest_state_space_indices[i], s:e].unsqueeze(0).repeat(self.envs_per_sample, 1)
                    root_node_quat = self.state_space_nodes[0, s:e].unsqueeze(0).repeat(self.envs_per_sample, 1)

                    disp_current_sim = compute_quat_angle(new_quats, root_node_quat)
                    disp_nearest_node = compute_quat_angle(nearest_node_quat, root_node_quat)
                    disp = (disp_current_sim - disp_nearest_node).squeeze(-1)
                    # if displacement is too large, mark the new state as invalid
                    invalid[batch] = torch.where(
                        torch.abs(disp) > 1.0,
                        torch.ones_like(invalid[batch]),
                        invalid[batch]
                    )                    
                # for axis centric tree make sure sampled actions do not go in invalid direction
                else:
                    # compute displacement of sampled action about the tree axis
                    s,e = self.state_space_map["object_orientation"]
                    new_quats = new_states_batch[:, s:e].view(-1, 4)
                    nearest_node_quat = self.state_space_nodes[self.nearest_state_space_indices[i], s:e].unsqueeze(0).repeat(self.envs_per_sample, 1)
                    root_node_quat = self.state_space_nodes[0, s:e].unsqueeze(0).repeat(self.envs_per_sample, 1)
                    # contains roll, pitch, yaw tensors. Each has dimension (envs_per_sample,) in range [0, 2*pi]
                    rpy_current_sim = get_euler_disps(new_quats, root_node_quat)
                    rpy_nearest_node = get_euler_disps(nearest_node_quat, root_node_quat)
                    # fetch displacement of sampled action about the tree axis
                    axis_centric_disp_current_state = rpy_current_sim[self.axis]
                    axis_centric_disp_nearest_node = rpy_nearest_node[self.axis]
                    axis_centric_disp = axis_centric_disp_current_state - axis_centric_disp_nearest_node
                    # if displacement is too large, mark the new state as invalid (due to angular wraparound)
                    invalid[batch] = torch.where(
                        torch.abs(axis_centric_disp) > 1.0,
                        torch.ones_like(invalid[batch]),
                        invalid[batch]
                    )

                invalid_indices_in_batch = invalid[batch].nonzero()

                # if there are valid states in the batch
                if len(invalid_indices_in_batch) < self.envs_per_sample:
                    # remove invalid indices from contention
                    distances[invalid_indices_in_batch] = torch.inf
                    best_new_state_idx = torch.argmin(distances)

                    # configure new RRT tree node
                    best_new_state = new_states_batch[best_new_state_idx]
                    best_action = sampled_actions_batch[best_new_state_idx]
                    parent_index = self.nearest_state_space_indices[i]
                                        
                    # parse search space point from state space
                    s_hand_joint_pos, e_hand_joint_pos = self.state_space_map["hand_joint_pos"]
                    s_object_pos, e_object_pos = self.state_space_map["object_pos"]
                    s_object_quat, e_object_quat = self.state_space_map["object_orientation"]


                    # best node hand joint pos
                    best_node_hand_joint_pos = best_new_state[s_hand_joint_pos:e_hand_joint_pos]
                    # best node object pos
                    best_node_object_pos = best_new_state[s_object_pos:e_object_pos]
                    # best node object quat
                    best_node_object_quat = best_new_state[s_object_quat:e_object_quat]

                    best_node_ss = torch.cat(
                        [
                            best_node_hand_joint_pos,
                            best_node_object_pos,
                            best_node_object_quat
                        ],
                        dim=0
                    )

                    # add new node to RRT tree
                    self.state_space_nodes = torch.cat([ # state space
                        self.state_space_nodes,
                        best_new_state.unsqueeze(0)
                    ])
                    self.sample_space_nodes = torch.cat([ # sample space
                        self.sample_space_nodes,
                        best_node_ss.unsqueeze(0)
                    ])
                    self.sim_states = torch.cat([ # simulator state
                        self.sim_states,
                        simulator_states_batch[best_new_state_idx].unsqueeze(0)
                    ])

                    self.rrt_parent_indices.append(parent_index)
                    self.rrt_children_indices.append([])
                    # set new node as child of parent
                    self.rrt_children_indices[parent_index].append(len(self.state_space_nodes) - 1)
                    self.rrt_actions = torch.cat([
                        self.rrt_actions,
                        best_action.unsqueeze(0)
                    ])

                    # compute distance to root for new node and store for logging
                    root_quat = self.state_space_nodes[0, s_object_quat:e_object_quat].unsqueeze(0)
                    if self.type == "all": # just compute angle between oracle quat and current
                        distance = compute_quat_angle(best_node_object_quat.unsqueeze(0), root_quat)
                    else: # return roll, pitch or yaw (depending on tree axis)
                        rpy = get_euler_disps(best_node_object_quat.unsqueeze(0), root_quat)
                        distance = rpy[self.axis]
                    self.node_distances_to_root.append(distance.item())

        self.best_distance = max(self.node_distances_to_root)
        self._logger()

    def sample_new_configuration(self):
        """ Samples a batch of new configurations
        Returns:
            tensor: (samples_per_iter, sample_dimension=)
        """
        alpha = torch_rand_float(
            0.0,
            1.0,
            (self.samples_per_iter, self.sample_space_dimension),
            device=self.device,
        )
        sample = alpha * self.sample_space_max + (1 - alpha) * self.sample_space_min
        # randomly sample batch of quaternion rotations
        quat = sample_random_quat(self.samples_per_iter, self.device)
        sample = torch.cat((sample, quat), dim=1)
        return sample
    
    def compute_distance(self, node, node_type, node_set, node_set_type):
        """ computes distance from node to each element in node set """
        """ node type (either state space node or search space node) needs to be specified for each arg"""
        node_set_len = node_set.size(0)
        distance = torch.zeros((node_set_len, 1)).to(self.device)

        eligible_types = ["state_space_node", "sample_space_node"]
        assert node_type in eligible_types, "node type not recognized"
        # fetch map for parsing node
        if node_type == "state_space_node":
            node_map = self.state_space_map
        else:
            node_map = self.sample_space_map
        # fetch map for parsing node set
        if node_set_type == "state_space_node":
            node_set_map = self.state_space_map
        else:
            node_set_map = self.sample_space_map

        # hand position distance
        s_node, e_node = node_map["hand_joint_pos"]
        s_node_set, e_node_set = node_set_map["hand_joint_pos"]
        node_hand_pos = node[s_node:e_node]
        node_set_hand_pos = node_set[:, s_node_set:e_node_set]
        distance += torch.norm(node_hand_pos - node_set_hand_pos, dim=1).unsqueeze(-1)

        # object position distance
        s_node, e_node = node_map["object_pos"]
        s_node_set, e_node_set = node_set_map["object_pos"]
        node_object_pos = node[s_node:e_node]
        node_set_object_pos = node_set[:, s_node_set:e_node_set]
        distance += torch.norm(node_object_pos - node_set_object_pos, dim=1).unsqueeze(-1)

        # object orientation distance
        s_node, e_node = node_map["object_orientation"]
        s_node_set, e_node_set = node_set_map["object_orientation"]
        node_object_quat = node[s_node:e_node].unsqueeze(0)
        node_set_object_quat = node_set[:, s_node_set:e_node_set].view(node_set_len, 4)
        distance += compute_quat_angle(node_object_quat, node_set_object_quat)

        return distance

    def get_nearest_tree_nodes(self):
        # stores the nearest simulator states
        nearest_tree_nodes_sim_state_list = []
        # stores the nearest simulator state indices
        nearest_tree_nodes_index_list = []
        for i in range(self.samples_per_iter):
            # compute distance from each sample to all nodes in the tree
            distance = self.compute_distance(
                node=self.samples[i],
                node_type="sample_space_node",
                node_set=self.sample_space_nodes,
                node_set_type="sample_space_node"
            )
            nearest_index = int(torch.argmin(distance))
            nearest_tree_nodes_index_list.append(nearest_index)
            nearest_tree_nodes_sim_state_list.append(self.sim_states[nearest_index])
        return nearest_tree_nodes_index_list, nearest_tree_nodes_sim_state_list

    def set_sim_state(self, sim_states):
        for i in range(self.samples_per_iter):
            batch = range(self.envs_per_sample * i, self.envs_per_sample * (i + 1))
            sim_state = sim_states[i]
            # set hand pose
            # Note: these indices never change --> can be hardcoded
            self.hand_dof_pos[batch] = sim_state[0:15]
            # set joint setpoints
            self.target_hand_joint_pos[batch] = sim_state[15:30]
            self.target_hand_joint_pos[batch] = tensor_clamp(
                self.target_hand_joint_pos[batch],
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
            )
            # set object pose
            object_pos = sim_state[30:33]
            object_quat = sim_state[33:37]
            self.root_state_tensor[self.object_indices[batch], :3] = object_pos
            self.root_state_tensor[self.object_indices[batch], 3:7] = object_quat
            # set object velocity to zero
            self.root_state_tensor[self.object_indices[batch], 7:13] = 0.0
        self.hand_dof_vel[:] = 0.0
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))

    def compute_random_target_update(self):
        """Random actions to compute target"""
        self.sampled_actions = self.random_actions()
        return self.sampled_actions * self.action_scale
    
    def update_sim_target(self, target_update):
        """Update joint setpoints"""
        for i in range(self.samples_per_iter):
            batch = range(self.envs_per_sample * i, self.envs_per_sample * (i + 1))
            self.target_hand_joint_pos[batch] = tensor_clamp(
                self.target_hand_joint_pos[batch] + target_update[batch],
                self.hand_dof_lower_limits,
                self.hand_dof_upper_limits,
            )
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

    def step_with_sampled_actions(self):
        """ Implements the sampling portion of the RRT algorithm"""
        # sample a new point in search space
        self.samples = self.sample_new_configuration() # sample space nodes
        # find nearest nodes in existing tree
        self.nearest_state_space_indices, nearest_tree_nodes_sim_state_list = self.get_nearest_tree_nodes()
        # set simulator state to nearest nodes
        self.set_sim_state(nearest_tree_nodes_sim_state_list)
        # sample random actions
        target_update = self.compute_random_target_update()
        # update simulator state with random actions
        self.update_sim_target(target_update)

    def _logger(self):
        print("------------------------")
        print("Tree size: ", len(self.state_space_nodes))
        print("Farthest node distance: ", self.best_distance)
        print(f"Time elapsed: {round(time.time() - self.time_start,1)}s")
        print(f"nodes per second: {round(len(self.state_space_nodes) / (time.time() - self.time_start),1)}")
        print("------------------------")

    def load(self, load_path):
        with open(load_path, 'rb') as f:
            tree = pickle.load(f)
        assert self.type == tree["type"], "Tree type mismatch"
        self.sample_space_min = torch.from_numpy(tree["sample_space_min"]).to(self.device)
        self.sample_space_max = torch.from_numpy(tree["sample_space_max"]).to(self.device)
        self.sample_space_nodes = torch.from_numpy(tree["sample_space_nodes"]).to(self.device)
        self.state_space_nodes = torch.from_numpy(tree["state_space_nodes"]).to(self.device)
        self.sim_states = torch.from_numpy(tree["simulator_states"]).to(self.device)
        self.rrt_parent_indices = tree["parents"]
        self.rrt_children_indices = tree["children"]
        self.rrt_actions = torch.from_numpy(tree["actions"]).to(self.device)

    def save(self, save_file):
        tree = {
            "sample_space_min" : self.sample_space_min.cpu().numpy(),
            "sample_space_max" : self.sample_space_max.cpu().numpy(),
            "sample_space_nodes" : self.sample_space_nodes.cpu().numpy(),
            "state_space_nodes" : self.state_space_nodes.cpu().numpy(),
            "simulator_states" : self.sim_states.cpu().numpy(),
            "parents" : self.rrt_parent_indices,
            "children" : self.rrt_children_indices,
            "actions" : self.rrt_actions.cpu().numpy(),
            "type" : self.type,
            "state_space_map": self.state_space_map.copy(), # for PPO class
        }
        with open(save_file, "wb") as f:
            print("Saving tree to ", save_file)
            pickle.dump(tree, f)

    def save_tree_exit_if_done(self):
        if len(self.sample_space_nodes) > self.saved_num_nodes + 10000:
            self.save("outputs/trees/" + self.save_file + f"/{self.save_file}_tree_nodes_{len(self.sample_space_nodes)}.pkl")
            self.saved_num_nodes = len(self.sample_space_nodes)
        if len(self.sample_space_nodes) > self.max_nodes:
            self.save("outputs/trees/" + self.save_file + f"/{self.save_file}_tree_nodes_{len(self.sample_space_nodes)}.pkl")
            self.saved_num_nodes = len(self.sample_space_nodes)
            exit()

    def reset(self):
        self.collect_states_update_tree()
        self.save_tree_exit_if_done()
        self.step_with_sampled_actions()