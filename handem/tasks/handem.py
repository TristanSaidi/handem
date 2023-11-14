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
        # current discriminator output
        self.discriminator_log_softmax = torch.ones((self.num_envs, self.num_objects), device=self.device)
        self.discriminator_log_softmax = torch.log(self.discriminator_log_softmax / self.num_objects)
        self.confidence = torch.zeros((self.num_envs, 1), device=self.device)

    def _setup_rotation_axis(self, axis_idx=2):
        self.rotation_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self.rotation_axis[:, axis_idx] = 1

    def _setup_reward_config(self):
        # Reward
        self.disc_pred_reward = self.cfg["env"]["reward"]["disc_pred_reward"]
        self.disc_loss_reward = self.cfg["env"]["reward"]["disc_loss_reward"]

    def update_discriminator_output(self, output):
        self.discriminator_log_softmax = output.clone().detach()

    def get_disc_correct(self):
        "return whether last discriminator prediction was correct"
        return self.correct.clone().detach()

    def _setup_reset_config(self):
        self.confidence_threshold = self.cfg["env"]["reset"]["confidence_threshold"] * torch.ones((self.num_envs, 1), device=self.device)
        self.obj_xyz_lower_lim = self.cfg["env"]["reset"]["obj_xyz_lower_lim"]
        self.obj_xyz_upper_lim = self.cfg["env"]["reset"]["obj_xyz_upper_lim"]

    def discriminator_predict(self):
        # convert discriminator log softmax to predictions
        disc_softmax = torch.exp(self.discriminator_log_softmax)
        disc_max_softmax, disc_pred = disc_softmax.max(dim=1)
        correct = torch.where(
            disc_pred == self.object_labels, 
            torch.ones_like(disc_pred), 
            torch.zeros_like(disc_pred)
        ).unsqueeze(1)

        ########### print information for inference-time debugging ###########
        if self.headless == False:
            color = 'green' if correct[0] else 'red'
            bold = ["bold"] if disc_max_softmax[0] > self.confidence_threshold[0] else None
            cprint(f'Discriminator prediction: {disc_pred[0]}, confidence: {disc_max_softmax[0]:.2f}', color, end='\r', attrs=bold)
            if color=='green' and bold is not None:
                sleep(3)
        ########### print information for inference-time debugging ###########
        
        disc_max_softmax = disc_max_softmax.unsqueeze(1)
        # mask out correct predictions with low confidence
        correct = torch.where(
            disc_max_softmax > self.confidence_threshold, 
            correct, 
            torch.zeros_like(correct)
        )
        return correct, disc_max_softmax

    def compute_disc_loss(self):
        # compute discriminator loss
        loss = torch.nn.functional.nll_loss(
            self.discriminator_log_softmax, 
            self.object_labels,
            reduction='none'
        )
        return loss

    def compute_reward(self):
        # correct predictions
        self.correct, self.confidence = self.discriminator_predict()
        disc_pred_reward = self.disc_pred_reward * self.correct
        # discriminator loss reward
        loss = self.compute_disc_loss()
        disc_loss_reward = -1 * self.disc_loss_reward * loss.unsqueeze(1)
        # total reward
        reward = disc_pred_reward + disc_loss_reward
        reward = reward.squeeze(1)
        self.rew_buf[:] = reward

    def check_reset(self):
        super().check_reset() # check if the object is out of bounds
        # task specific reset conditions
        reset = self.reset_buf[:]
        # if confident
        confident = (self.confidence > self.confidence_threshold).int().squeeze(1)
        reset = torch.where(confident == 1, torch.ones_like(self.reset_buf), reset)
        # if end of episode
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), reset)
        self.reset_buf[:] = reset
