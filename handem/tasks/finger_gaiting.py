import os


# from utils.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles, quaternion_to_axis_angle

import numpy as np
import torch

from isaacgym import gymtorch

# from isaacgym.torch_utils import tensor_clamp, to_torch
from isaacgym.torch_utils import quat_conjugate, quat_mul
from termcolor import cprint

from handem.tasks.ihm_base import IHMBase
from handem.utils.torch_jit_utils import quat_to_angle_axis, my_quat_rotate


class FingerGaiting(IHMBase):
    """Samples random stable grasps for initial distribution"""

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
        assert "target_orientation" not in self.cfg["env"]["feedbackObs"] \
            and "target_orientation" not in self.cfg["env"]["feedbackState"], "target_orientation is not supported for fingerGaiting"
        self.saved_grasp_states = None
        if self.cfg["env"]["use_saved_grasps"]:
            self.load_grasps()
        self._setup_rotation_axis(cfg["env"]["rotationAxis"])
        self._setup_reward_config()
        self._setup_reset_config()

    def _setup_rotation_axis(self, axis_idx=2):
        self.rotation_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self.rotation_axis[:, axis_idx] = 1

    def _setup_reward_config(self):
        # Reward
        self.ang_vel_clip_max = self.cfg["env"]["reward"]["angVelClipMax"]
        self.ang_vel_clip_min = self.cfg["env"]["reward"]["angVelClipMin"]
        self.rotation_reward_scale = self.cfg["env"]["reward"]["rotationRewardScale"]
        self.object_lin_vel_penalty_scale = self.cfg["env"]["reward"]["objectLinVelPenaltyScale"]
        self.pose_diff_penalty_scale = self.cfg["env"]["reward"]["poseDiffPenaltyScale"]
        self.torque_penalty_scale = self.cfg["env"]["reward"]["torquePenaltyScale"]
        self.work_penalty_scale = self.cfg["env"]["reward"]["workPenaltyScale"]
        self.ftip_obj_disp_pen = self.cfg["env"]["reward"]["ftip_obj_disp_pen"] # penalty for displacement btwn ftip and object COM
        self.out_of_bounds_pen = self.cfg["env"]["reward"]["out_of_bounds_pen"]

    def _setup_reset_config(self):
        self.obj_xyz_lower_lim = self.cfg["env"]["reset"]["obj_xyz_lower_lim"]
        self.obj_xyz_upper_lim = self.cfg["env"]["reset"]["obj_xyz_upper_lim"]
        self.object_tilt_lim = self.cfg["fingerGaiting"]["object_tilt_lim"]

    def compute_reward(self):

        obj_pos = self.object_pos
        obj_quat = self.object_state[:, 3:7]
        obj_vel = self.object_lin_vel
        prev_obj_quat = self.object_state_prev[:, 3:7]
        torques = self.hand_torques
        hand_dof_vel = self.hand_dof_vel
        hand_dof_pos = self.hand_dof_pos

        # compute reward
        quat_diff = quat_mul(obj_quat, quat_conjugate(prev_obj_quat))
        magnitude, axis = quat_to_angle_axis(quat_diff)
        magnitude = magnitude - 2 * np.pi * (magnitude > np.pi)
        axis_angle = torch.mul(axis, torch.reshape(magnitude, (-1, 1)))
        avg_angular_vel = axis_angle / (self.sim_params.dt * self.control_freq_inv)

        rotation_axis_reshaped = self.rotation_axis[0:avg_angular_vel.size(0), :]
        vec_dot = (avg_angular_vel * rotation_axis_reshaped).sum(-1)
        rotation_reward = torch.clip(vec_dot, max=self.ang_vel_clip_max)
        rotation_reward = self.rotation_reward_scale * rotation_reward

        # linear velocity penalty
        object_lin_vel_penalty = torch.norm(obj_vel, p=1, dim=-1)
        # torque penalty
        torque_penalty = (torques**2).sum(-1)
        work_penalty = ((torques * hand_dof_vel).sum(-1)) ** 2
        # pose difference penalty
        pose_diff_penalty = ((self.default_hand_joint_pos - hand_dof_pos) ** 2).sum(-1)
        object_pose_diff_penalty = ((self.default_object_pos - obj_pos) ** 2).sum(-1)


        reward = rotation_reward
        reward = reward + object_lin_vel_penalty * self.object_lin_vel_penalty_scale
        reward = reward + pose_diff_penalty * self.pose_diff_penalty_scale
        reward = reward + object_pose_diff_penalty * self.pose_diff_penalty_scale
        reward = reward + torque_penalty * self.torque_penalty_scale
        reward = reward + work_penalty * self.work_penalty_scale
        reward = reward + -1 * self.out_of_bounds.clone() * self.out_of_bounds_pen        
        self.rew_buf[:] = reward

    def check_reset(self):
        super().check_reset() # check if the object is out of bounds
        # task specific reset conditions
        reset = self.reset_buf[:]
        obj_quat = self.object_state[:, 3:7]
        obj_axis = my_quat_rotate(obj_quat, self.rotation_axis)

        reset = torch.where(
            torch.linalg.norm(obj_axis * self.rotation_axis, dim=1) < self.object_tilt_lim,
            torch.ones_like(self.reset_buf),
            reset,
        )
        # if end of episode
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), reset)
        self.reset_buf[:] = reset
