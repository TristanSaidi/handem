import os
import numpy as np
import torch

from isaacgym import gymtorch

from isaacgym.torch_utils import quat_conjugate, quat_mul
from termcolor import cprint

from handem.tasks.ihm_base import IHMBase
from handem.utils.torch_jit_utils import quat_to_angle_axis, my_quat_rotate

class HANDEM(IHMBase):
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

    def _setup_rotation_axis(self, axis_idx=2):
        self.rotation_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self.rotation_axis[:, axis_idx] = 1

    def _setup_reward_config(self):
        # Reward
        pass

    def _setup_reset_config(self):
        self.obj_xyz_lower_lim = self.cfg["env"]["reset"]["obj_xyz_lower_lim"]
        self.obj_xyz_upper_lim = self.cfg["env"]["reset"]["obj_xyz_upper_lim"]

    def compute_reward(self):
        # Reward
        pass

    def check_reset(self):
        super().check_reset() # check if the object is out of bounds
        # task specific reset conditions
        reset = self.reset_buf[:]
        # if end of episode
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), reset)
        self.reset_buf[:] = reset
