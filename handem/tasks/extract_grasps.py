import pickle
import time

import torch
import numpy as np
from isaacgym import gymtorch
from termcolor import cprint

from handem.tasks.rrt import RRT
from handem.utils.misc import compute_quat_angle, get_euler_disps


class ExtractGrasps(RRT):
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
        # Use a very small dt to "disable" physics.
        # We only want to show the states
        cfg["sim"]["dt"] = 0.001
        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        self.paths = None
        self.curr_path_idx = -1
        self.path_prog = 0
        self.max_nodes = self.cfg["extract_grasps"]["max_nodes"]
        self.save_file = self.cfg["extract_grasps"]["save_file"]

    def post_physics_step(self):
        time.sleep(0.1)
        return super().post_physics_step()

    def compute_angular_distances_to_root(self, node_quats=None):
        node_quats = self.sim_states[:, 33:37].view(-1, 4) if node_quats is None else node_quats
        root_quat = node_quats[0].unsqueeze(0).repeat(len(node_quats), 1)
        if self.type == "all": # just compute angle between oracle quat and current
            distances = compute_quat_angle(root_quat, node_quats)
        else: # return roll, pitch or yaw (depending on tree axis)
            rpy = get_euler_disps(node_quats, root_quat)
            distances = rpy[self.axis]
        return distances

    def reset(self):
        """Computes best paths on first call"""
        if self.paths is None:
            self.extract()

    def extract(self):
        self.load(self.cfg["extract_grasps"]["tree_path"])
        distances = self.compute_angular_distances_to_root()
        cprint(f"Max distance to root: {torch.max(distances)}", color="green", attrs=["bold"])
        paths = [] # for rendering
        grasps = [] # for saving
        while len(grasps) < self.max_nodes:
            idx = int(torch.argmax(distances))
            path = [] # current path
            while idx is not None:
                if distances[idx] != -torch.inf: # only add to grasps if not already added
                    grasps.append(torch.reshape(self.sim_states[idx], (-1, 1)))
                path.append(idx)
                distances[idx] = -torch.inf  # this node will be ignored in finding the next path
                idx = self.rrt_parent_indices[idx]      
            paths.append(list(reversed(path)))
        self.paths = paths
        grasps_tensor = torch.cat(grasps, dim=0)
        self.save_grasps(grasps_tensor)

    def save_grasps(self, grasps):
        np_grasps = grasps.cpu().numpy()
        path = f"outputs/grasps/{self.save_file}_grasps"
        np.save(path, np_grasps)

    def pre_physics_step(self, actions: torch.Tensor):
        state_idx = self.paths[self.curr_path_idx][self.path_prog]
        node_distances_to_root = self.compute_angular_distances_to_root()
        print(f"Distance to root: {node_distances_to_root[state_idx]}")
        states = self.sim_states[state_idx].repeat(self.num_envs, 1)
        self.hand_dof_pos[:] = states[:, :15]
        self.hand_dof_vel[:] = 0.0
        self.ur5e_dof_pos[:] = self.default_ur5e_joint_pos
        self.root_state_tensor[self.object_indices, :7] = states[:, 30:37]
        self.root_state_tensor[self.object_indices, 7:] = 0.0
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))
        self.path_prog += 1
        path = self.paths[self.curr_path_idx]
        if self.path_prog > len(path) - 1:
            self.curr_path_idx = (self.curr_path_idx + 1) % len(self.paths)
            self.path_prog = 0
            cprint(f"Switching to path {self.curr_path_idx}", color="green", attrs=["bold"])
            time.sleep(2.0)

        return super().pre_physics_step(actions)