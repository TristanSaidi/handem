import os
import numpy as np
import torch
import gc

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_mul
from termcolor import cprint
from handem.tasks.ihm_base import IHMBase
from handem.utils.torch_jit_utils import quat_to_angle_axis, my_quat_rotate
from time import sleep
from pytorch3d.loss import chamfer_distance
from matplotlib import pyplot as plt

class HANDEM_Reconstruct(IHMBase):
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

        self.saved_grasp_states = None
        if self.cfg["env"]["use_saved_grasps"]:
            self.load_grasps()
        self._setup_rotation_axis(cfg["env"]["rotationAxis"])
        self._setup_reward_config()
        self._setup_reset_config()
        
        self.vertex_pred = torch.zeros((self.num_envs, self.num_vertices, 2), device=self.device)
        self.visualize_enabled = (self.headless==False) and self.num_envs == 1
        self.lbda = self.cfg["env"]["edge_loss_lambda"]
        if self.visualize_enabled:
            self.label_offset = np.array([0.3, 0.0, 0.2])
            self.pred_offset = np.array([0.4, 0.0, 0.2])

    def _setup_rotation_axis(self, axis_idx=2):
        self.rotation_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self.rotation_axis[:, axis_idx] = 1

    def _setup_reward_config(self):
        # Reward
        self.reg_loss_reward = self.cfg["env"]["reward"]["reg_loss_reward"]
        self.ftip_obj_dist_rew = self.cfg["env"]["reward"]["ftip_obj_dist_rew"]
        self.object_disp_rew = self.cfg["env"]["reward"]["object_disp_rew"]
        self.contact_loc_pen = self.cfg["env"]["reward"]["contact_loc_pen"]
        self.hand_pose_pen = self.cfg["env"]["reward"]["hand_pose_pen"]

    def update_regressor_output(self, output):
        output = output.reshape(self.num_envs, self.num_vertices, 2)
        self.vertex_pred = self.vertex_pred + output.clone().detach().to(self.device)

    def _setup_reset_config(self):
        self.loss_threshold = self.cfg["env"]["reset"]["loss_threshold"]
        self.obj_xyz_lower_lim = self.cfg["env"]["reset"]["obj_xyz_lower_lim"]
        self.obj_xyz_upper_lim = self.cfg["env"]["reset"]["obj_xyz_upper_lim"]

    def get_reg_correct(self):
        "return whether last regressor prediction was correct (within threshold)"
        return self.correct.clone().detach()

    def visualize_vertices(self, vertices, offset, color):
        # labels
        vertices = np.concatenate((vertices, np.zeros((self.num_envs, self.num_vertices, 1))), axis=-1)
        # predictions
        for i in range(self.num_vertices):
            start_idx = i
            end_idx = i+1 if i < self.num_vertices-1 else 0
            vertex = [vertices[0, start_idx] + offset, vertices[0, end_idx] + offset]
            self.gym.add_lines(self.viewer, self.envs[0], 1, vertex, color)

    @torch.no_grad()
    def compute_regressor_loss(self):
        # broadcasting magic
        vertex_pred = self.vertex_pred.clone().detach() # (B, N, 2)
        # compute chamfer distance loss
        chamfer_loss, _ = chamfer_distance(self.transformed_vertex_labels, vertex_pred, batch_reduction=None)
        # edge loss
        vertex_offset = torch.cat((vertex_pred[:, 1:, :], vertex_pred[:, 0:1, :]), dim=1)
        edge_loss = torch.linalg.norm(vertex_offset - vertex_pred, dim=2).mean(dim=1)
        # total loss
        loss = chamfer_loss + edge_loss

        correct = torch.where(
            loss < self.loss_threshold,
            torch.ones_like(loss),
            torch.zeros_like(loss)
        ).unsqueeze(1)
        return loss, correct

    def compute_reward(self):
        if self.visualize_enabled:
            self.gym.clear_lines(self.viewer)
            self.visualize_vertices(self.transformed_vertex_labels.clone().cpu().numpy(), self.label_offset, color=[0, 1, 0])
            self.visualize_vertices(self.vertex_pred.clone().cpu().numpy(), self.pred_offset, color=[1, 0, 0])
        # correct predictions
        self.loss, self.correct = self.compute_regressor_loss()
        reg_loss_reward = self.reg_loss_reward * torch.exp(-1 * self.loss).unsqueeze(1)
        # ftip-object distance reward
        total_ftip_obj_disp = self.compute_ftip_obj_disp()
        ftip_obj_dist_rew = -1 * self.ftip_obj_dist_rew * total_ftip_obj_disp.unsqueeze(1)
        # object displacement from default
        obj_disp = torch.linalg.norm(self.object_pos.clone() - self.default_object_pos.clone(), dim=1)
        obj_disp_rew = -1 * self.object_disp_rew * obj_disp.unsqueeze(1)
        # contact location penalty
        contact_loc_pen = -1 * self.contact_loc_pen * self.contact_location_constraint().float().unsqueeze(1)
        # hand pose penalty
        close_hand = torch.tensor([0.0, 0.25, 0.45]*5).to(self.device)
        hand_pose_diff = torch.linalg.norm(self.hand_dof_pos.clone() - close_hand, dim=1)
        hand_pose_pen = -1 * self.hand_pose_pen * hand_pose_diff.unsqueeze(1)
        # total reward
        reward = reg_loss_reward + ftip_obj_dist_rew + obj_disp_rew + contact_loc_pen + hand_pose_pen
        reward = reward.squeeze(1)
        self.rew_buf[:] = reward

    def check_reset(self):
        super().check_reset() # check if the object is out of bounds
        # task specific reset conditions
        reset = self.reset_buf[:]
        # if confident
        reset = torch.where(self.correct.squeeze(1) == 1, torch.ones_like(self.reset_buf), reset)
        # if end of episode
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), reset)
        self.reset_buf[:] = reset
