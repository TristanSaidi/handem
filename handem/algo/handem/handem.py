# --------------------------------------------------------
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

import os
import time
import torch


from tensorboardX import SummaryWriter

from handem.algo.handem.experience import ExperienceBuffer
from handem.algo.models.models import ActorCritic
from handem.algo.models.models import MLPDiscriminator, MLPRegressor
from handem.algo.models.running_mean_std import RunningMeanStd
from handem.utils.misc import AverageScalarMeter
from handem.utils.torch_jit_utils import quat_to_angle_axis, my_quat_rotate
from pytorch3d.loss import chamfer_distance

class HANDEM(object):
    def __init__(self, env, output_dir, full_config):
        self.device = full_config['rl_device']
        self.ppo_net_config = full_config.train.handem.ppo_network
        self.train_config = full_config.train.handem
        # ---- build environment ----
        self.env = env
        self.num_actors = self.train_config['num_actors']
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        # ---- Explorer ----
        self.asymmetric = self.train_config['asymmetric']
        self.state_dim = self.env.cfg['env']['numStates']
        ppo_net_config = {
            'actor_units': self.ppo_net_config.mlp.actor_units,
            'critic_units': self.ppo_net_config.mlp.critic_units,
            'actions_num': self.actions_num,
            'actor_input_shape': self.obs_shape[0],
            'critic_input_shape': self.state_dim if self.asymmetric else self.obs_shape[0],
            'asymmetric': self.asymmetric,
        }
        self.explorer = ActorCritic(ppo_net_config)
        self.explorer.to(self.device)
        # ---- Discriminator/Regressor ----
        self.proprio_hist_len = full_config.task["env"]["propHistoryLen"]
        self.proprio_dim = self.env.num_obs // self.env.obs_hist_len
        self.discriminator = None
        self.regressor = None
        self.n_vertices = None
        if full_config["task"]['name'] == 'HANDEM_Reconstruct':
            self.reconstruction_net_config = full_config.train.handem.reconstruction_network
            self.n_vertices = self.reconstruction_net_config.n_vertices
            self.vertex_dim = self.reconstruction_net_config.vertex_dim
            self.autoregressive = self.reconstruction_net_config.autoregressive
            units = self.reconstruction_net_config.mlp.units
            regressor_net_config = {
                'proprio_dim': self.proprio_dim,
                'proprio_hist_len': self.proprio_hist_len,
                'units': units,
                'vertex_dim': self.vertex_dim,
                'n_vertices': self.n_vertices,
                'autoregressive': self.autoregressive,
            }
            self.regressor = MLPRegressor(regressor_net_config)
            num_params = self.regressor.get_num_params()
            print(f'Number of regressor parameters: {num_params}')
            self.regressor_epochs = self.train_config["reconstruction_network"]["regressor_epochs"]
            self.regressor.to(self.device)
        else:
            self.disc_net_config = full_config.train.handem.discriminator_network
            self.num_classes = self.env.num_objects
            self.disc_arch = self.disc_net_config.arch
            if self.disc_arch == 'mlp':
                disc_net_config = {
                    'proprio_dim': self.proprio_dim,
                    'proprio_hist_len': self.proprio_hist_len,
                    'units': self.disc_net_config.mlp.units,
                    'num_classes': self.num_classes,
                }
                self.discriminator = MLPDiscriminator(disc_net_config)
                num_params = self.discriminator.get_num_params()
                print(f'Number of discriminator parameters: {num_params}')
            else:
                obs_dim = self.obs_shape[0]
                hidden_size = self.disc_net_config.transformer.hidden_size
                num_classes = self.num_classes
                proprio_hist_len = self.proprio_hist_len
                n_layer = self.disc_net_config.transformer.n_layer
                n_head = self.disc_net_config.transformer.n_head
                self.discriminator = GPT2Discriminator(
                    obs_dim,
                    hidden_size,
                    num_classes,
                    proprio_hist_len,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_positions=self.proprio_hist_len,
                    n_ctx=self.proprio_hist_len
                )
                num_params = self.discriminator.get_num_params()
                print(f'Number of discriminator parameters: {num_params}')
            self.discriminator.to(self.device)
            self.discriminator_epochs = self.train_config["discriminator_network"]['discriminator_epochs']
        # ---- Normalization ----
        self.obs_mean_std = RunningMeanStd(self.obs_shape).to(self.device) # observation running mean
        self.state_mean_std = RunningMeanStd((self.state_dim,)).to(self.device) # state running mean for asymmetric critic
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        self.hist_mean_std = RunningMeanStd((self.proprio_hist_len, self.proprio_dim)).to(self.device)
        # ---- Labels ----
        self.labels = self.env.object_labels.clone().unsqueeze(-1)
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'nn')
        self.tb_dif = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.last_lr = float(self.train_config['learning_rate'])
        self.weight_decay = self.train_config.get('weight_decay', 0.0)
        self.actor_optimizer = torch.optim.Adam(self.explorer.parameters(), self.last_lr, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.explorer.parameters(), self.last_lr, weight_decay=self.weight_decay)
        if full_config["task"]['name'] == 'HANDEM_Reconstruct':
            self.regressor_optimizer = torch.optim.Adam(self.regressor.parameters(), self.last_lr, weight_decay=self.weight_decay)
        else:
            self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.last_lr, weight_decay=self.weight_decay)
        # ---- PPO Train Param ----
        self.e_clip = self.train_config['e_clip']
        self.clip_value = self.train_config['clip_value']
        self.entropy_coef = self.train_config['entropy_coef']
        self.bounds_loss_coef = self.train_config['bounds_loss_coef']
        self.gamma = self.train_config['gamma']
        self.tau = self.train_config['tau']
        self.truncate_grads = self.train_config['truncate_grads']
        self.grad_norm = self.train_config['grad_norm']
        self.value_bootstrap = self.train_config['value_bootstrap']
        self.normalize_advantage = self.train_config['normalize_advantage']
        self.normalize_input = self.train_config['normalize_input']
        self.normalize_value = self.train_config['normalize_value']
        # ---- PPO Collect Param ----
        self.horizon_length = self.train_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.train_config['minibatch_size']
        self.actor_mini_epochs = self.train_config['actor_mini_epochs']
        self.critic_mini_epochs = self.train_config['critic_mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.kl_threshold = self.train_config['kl_threshold']
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
        # ---- Snapshot
        self.save_freq = self.train_config['save_frequency']
        self.save_best_after = self.train_config['save_best_after']
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        self.episode_rewards = AverageScalarMeter(int(1.6e10))
        self.episode_lengths = AverageScalarMeter(int(1.6e10))
        self.num_success = AverageScalarMeter(int(1.6e10))

        self.obs = None
        self.epoch_num = 0
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size, self.proprio_dim, self.env.obs_hist_len,
            self.state_dim, self.actions_num, self.proprio_hist_len, self.n_vertices, self.device,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.train_config['max_agent_steps']
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

    def write_stats(self, a_losses, c_losses, b_losses, d_losses, r_losses, entropies, kls):
        self.writer.add_scalar('performance/RLTrainFPS', self.agent_steps / self.rl_train_time, self.agent_steps)
        self.writer.add_scalar('performance/EnvStepFPS', self.agent_steps / self.data_collect_time, self.agent_steps)

        self.writer.add_scalar('losses/actor_loss', torch.mean(torch.stack(a_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/bounds_loss', torch.mean(torch.stack(b_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/critic_loss', torch.mean(torch.stack(c_losses)).item(), self.agent_steps)
        if d_losses is not None:
            self.writer.add_scalar('losses/disc_loss', torch.mean(torch.stack(d_losses)).item(), self.agent_steps)
        if r_losses is not None:
            self.writer.add_scalar('losses/reg_loss', torch.mean(torch.stack(r_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/entropy', torch.mean(torch.stack(entropies)).item(), self.agent_steps)

        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        self.writer.add_scalar('info/kl', torch.mean(torch.stack(kls)).item(), self.agent_steps)

        for k, v in self.extra_info.items():
            self.writer.add_scalar(f'{k}', v, self.agent_steps)

    def set_eval(self):
        self.explorer.eval()
        if self.discriminator is not None:
            self.discriminator.eval()
        if self.regressor is not None:
            self.regressor.eval()
        if self.normalize_input:
            self.obs_mean_std.eval()
            self.state_mean_std.eval()
            self.hist_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.explorer.train()
        if self.discriminator is not None:
            self.discriminator.train()
        if self.regressor is not None:
            self.regressor.train()
        if self.normalize_input:
            self.obs_mean_std.train()
            self.state_mean_std.train()
            self.hist_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):
        """ Produces action from explorer and prediction from discriminator/regressor """
        processed_obs = self.obs_mean_std(obs_dict['obs'])
        processed_state = self.state_mean_std(obs_dict['state'])
        processed_hist = self.hist_mean_std(obs_dict['proprio_hist'])
        input_dict = {
            'obs': processed_obs,
            'state': processed_state,
        }
        # forward pass through explorer
        res_dict = self.explorer.act(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        # forward pass through discriminator/regressor
        if self.discriminator is not None:
            res_dict['disc_preds'] = self.discriminator(processed_hist)
        if self.regressor is not None:
            previous_pred = self.env.vertex_pred.clone()
            res_dict['reg_preds'] = self.regressor(processed_hist, previous_pred)
        return res_dict

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        self.agent_steps = self.batch_size
        d_loss_min = torch.inf
        r_loss_min = torch.inf

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, d_losses, r_losses, entropies, kls = self.train_epoch()
            
            # save checkpoint if discriminator loss is at a minimum
            if d_losses is not None:
                d_losses_avg = sum(d_losses)/len(d_losses)
                if d_losses_avg < d_loss_min:
                    print(f'save current best disc loss: {d_losses_avg:.2f}')
                    d_loss_min = d_losses_avg
                    self.save(os.path.join(self.nn_dir, 'best_disc_loss'))
            # save checkpoint if regressor loss is at a minimum
            if r_losses is not None:
                r_losses_avg = sum(r_losses)/len(r_losses)
                if r_losses_avg < r_loss_min:
                    print(f'save current best reg loss: {r_losses_avg:.5f}')
                    r_loss_min = r_losses_avg
                    self.save(os.path.join(self.nn_dir, 'best_reg_loss'))
            
            self.storage.data_dict = None

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                          f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                          f'Current Best: {self.best_rewards:.2f}'
            print(info_string)

            self.write_stats(a_losses, c_losses, b_losses, d_losses, r_losses, entropies, kls)

            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M_reward_{mean_rewards:.2f}'

            # update success rate if environment has returned such data
            running_mean_success = self.num_success.get_mean()
            self.writer.add_scalar('success/step', running_mean_success, self.agent_steps)

            if self.save_freq > 0:
                if self.epoch_num % self.save_freq == 0:
                    self.save(os.path.join(self.nn_dir, 'last'))

            if mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
                print(f'save current best reward: {mean_rewards:.2f}')
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, 'best_reward'))

        print('max steps achieved')

    def save(self, name):
        weights = {
            'explorer': self.explorer.state_dict(),
        }
        if self.discriminator is not None:
            weights['discriminator'] = self.discriminator.state_dict()
        if self.regressor is not None:
            weights['regressor'] = self.regressor.state_dict()
        if self.obs_mean_std:
            weights['obs_mean_std'] = self.obs_mean_std.state_dict()
        if self.state_mean_std:
            weights['state_mean_std'] = self.state_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        if self.hist_mean_std:
            weights['hist_mean_std'] = self.hist_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.explorer.load_state_dict(checkpoint['explorer'])
        if 'discriminator' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
        if 'regressor' in checkpoint:
            self.regressor.load_state_dict(checkpoint['regressor'])
        self.obs_mean_std.load_state_dict(checkpoint['obs_mean_std'])
        self.state_mean_std.load_state_dict(checkpoint['state_mean_std'])
        self.value_mean_std.load_state_dict(checkpoint['value_mean_std'])
        self.hist_mean_std.load_state_dict(checkpoint['hist_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.explorer.load_state_dict(checkpoint['explorer'])
        if 'discriminator' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
        if 'regressor' in checkpoint:
            self.regressor.load_state_dict(checkpoint['regressor'])
        if self.normalize_input:
            self.obs_mean_std.load_state_dict(checkpoint['obs_mean_std'])
            self.state_mean_std.load_state_dict(checkpoint['state_mean_std'])
            self.hist_mean_std.load_state_dict(checkpoint['hist_mean_std'])

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            # forward pass through explorer
            input_dict = {
                'obs': self.obs_mean_std(obs_dict['obs']),
                'state': self.state_mean_std(obs_dict['state'])
            }
            mu, _ = self.explorer.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            # update environment with discriminator prediction
            hist = self.hist_mean_std(obs_dict['proprio_hist'])
            if self.discriminator is not None:
                discriminator_output = self.discriminator(hist)
                self.env.update_discriminator_output(discriminator_output)
            if self.regressor is not None:
                regressor_output = self.regressor(hist, self.env.vertex_pred)
                self.env.update_regressor_output(regressor_output, self.autoregressive)
            # do env step
            obs_dict, r, done, info = self.env.step(mu)

    def train_critic(self):
        "Train critic network on data from rollout"
        c_losses = []
        for _ in range(0, self.critic_mini_epochs):
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs, states, _, _, _, _ = self.storage[i]
                obs = self.obs_mean_std(obs)
                states = self.state_mean_std(states)
                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                    'state': states,
                }
                # forward pass
                res_dict = self.explorer(batch_dict)
                values = res_dict['values']
                # compute critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped).mean()
                # update critic
                self.critic_optimizer.zero_grad()
                c_loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.explorer.parameters(), self.grad_norm)
                self.critic_optimizer.step()
                c_losses.append(c_loss)
        return c_losses

    def train_actor(self):
        "Train actor network on data from rollout"
        a_losses, b_losses = [], []
        entropies, kls = [], []
        for _ in range(0, self.actor_mini_epochs):
            ep_kls = []
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs, states, _, _, _, _ = self.storage[i]
                obs = self.obs_mean_std(obs)
                states = self.state_mean_std(states)
                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                    'state': states,
                }
                res_dict = self.explorer(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = 0
                a_loss, entropy, b_loss = [torch.mean(loss) for loss in [a_loss, entropy, b_loss]]

                loss = a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

                self.actor_optimizer.zero_grad()
                loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.explorer.parameters(), self.grad_norm)
                self.actor_optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.last_lr
            kls.append(av_kls)
        return a_losses, b_losses, entropies, kls
        
    def train_discriminator(self):
        d_losses = []
        for _ in range(self.discriminator_epochs):
            for i in range(len(self.storage)):
                _, _, _, _, _, _, _, _, _, proprio_hist, labels, _, _ = self.storage[i]
                proprio_hist = self.hist_mean_std(proprio_hist)
                # forward pass
                disc_preds = self.discriminator(proprio_hist).to(self.device)
                # compute discriminator loss
                labels = labels.squeeze(-1).type(torch.LongTensor).to(self.device)
                d_loss = torch.nn.functional.nll_loss(disc_preds, labels)
                # update discriminator
                self.disc_optimizer.zero_grad()
                d_loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_norm)
                self.disc_optimizer.step()
                d_losses.append(d_loss)
        return d_losses
    
    def train_regressor(self):
        r_losses = []
        for _ in range(self.regressor_epochs):
            for i in range(len(self.storage)):
                _, _, _, _, _, _, _, _, _, proprio_hist, _, vertex_labels, vertex_preds = self.storage[i]
                proprio_hist = self.hist_mean_std(proprio_hist)
                vertex_preds = vertex_preds.to(self.device) # (B, N, 2)
                # forward pass
                reg_preds = self.regressor(proprio_hist, vertex_preds)
                if self.autoregressive:
                    updated_vertex_preds = vertex_preds + reg_preds.reshape(-1, self.n_vertices, 2) # (B, N, 2)
                else:
                    updated_vertex_preds = reg_preds.reshape(-1, self.n_vertices, 2) # (B, N, 2)
                # compute chamfer loss
                chamfer_loss, _ = chamfer_distance(vertex_labels, updated_vertex_preds)
                # edge loss
                vertex_offset = torch.cat((updated_vertex_preds[:, 1:, :], updated_vertex_preds[:, 0:1, :]), dim=1)
                edge_loss = torch.linalg.norm(vertex_offset - updated_vertex_preds, dim=2).mean()
                # total loss
                r_loss = chamfer_loss + self.env.lbda * edge_loss
                # update regressor
                self.regressor_optimizer.zero_grad()
                r_loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), self.grad_norm)
                self.regressor_optimizer.step()
                r_losses.append(r_loss)
        return r_losses

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        
        # train the critic
        c_losses = self.train_critic()
        # train the actor
        a_losses, b_losses, entropies, kls = self.train_actor()
        # train the discriminator/regressor
        d_losses = None
        r_losses = None
        if self.discriminator is not None:
            d_losses = self.train_discriminator()
        if self.regressor is not None:
            r_losses = self.train_regressor()
        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, d_losses, r_losses, entropies, kls

    def play_steps(self):
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # update environment with prediction
            if self.discriminator is not None:
                discriminator_output = res_dict['disc_preds'].detach()
                self.env.update_discriminator_output(discriminator_output)
            if self.regressor is not None:
                regressor_output = res_dict['reg_preds'].detach()
                vertex_pred = self.env.vertex_pred.clone()
                self.storage.update_data('vertex_preds', n, vertex_pred)
                self.env.update_regressor_output(regressor_output, self.autoregressive)
                self.storage.update_data('vertex_labels', n, self.obs['vertex_labels'])
            # collect o_t
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('states', n, self.obs['state'])
            self.storage.update_data('proprio_hist', n, self.obs['proprio_hist'])
            self.storage.update_data('object_labels', n, self.labels) # storing this allows us to vary batchsize without trouble
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # do env step
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = 0.01 * rewards.clone()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards)
            self.episode_lengths.update(self.current_lengths)
            # update prediction success rate
            if self.discriminator is not None:
                success = self.env.get_disc_correct()
            if self.regressor is not None:
                success = self.env.get_reg_correct()
            self.num_success.update(success)
            assert isinstance(infos, dict), 'Info Should be a Dict'
            self.extra_info = {}
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        self.agent_steps += self.batch_size
        self.storage.computer_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
