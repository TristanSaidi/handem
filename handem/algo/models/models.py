import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from handem.algo.models.architectures.mlp import MLP

class MLPDiscriminator(nn.Module):
    def __init__(self, kwargs):
        super(MLPDiscriminator, self).__init__()
        proprio_dim = kwargs.pop("proprio_dim")
        proprio_hist_len = kwargs.pop("proprio_hist_len")
        units = kwargs.pop("units")
        num_classes = kwargs.pop("num_classes")
        units.append(num_classes)
        self.mlp = MLP(units, proprio_dim * proprio_hist_len)

    def forward(self, x):
        # x: tensor of size (B x proprio_hist_len x proprio_dim)
        x = x.flatten(1)
        x = self.mlp(x)
        x = F.log_softmax(x, dim=-1)
        return x

class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop("actions_num")
        actor_input_shape = kwargs.pop("actor_input_shape")
        critic_input_shape = kwargs.pop("critic_input_shape")
        self.actor_units = kwargs.pop("actor_units")
        self.critic_units = kwargs.pop("critic_units")
        self.asymmetric = kwargs.pop("asymmetric")

        # actor network
        self.actor_mlp = MLP(units=self.actor_units, input_size=actor_input_shape)
        self.mu = torch.nn.Linear(self.actor_units[-1], actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        # critic network
        self.critic_mlp = MLP(units=self.critic_units, input_size=critic_input_shape)
        self.value = torch.nn.Linear(self.critic_units[-1], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            "neglogpacs": -distr.log_prob(selected_action).sum(1),  # self.neglogp(selected_action, mu, sigma, logstd),
            "values": value,
            "actions": selected_action,
            "mus": mu,
            "sigmas": sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, extrin, _ = self._actor_critic(obs_dict)
        return mu, extrin # want to view extrin preds at inference

    def _actor_critic(self, obs_dict):
        obs = obs_dict["obs"]
        state = obs_dict["state"]
        extrin, extrin_gt = None, None
        # actor forward pass
        x_actor = self.actor_mlp(obs)
        mu = self.mu(x_actor)
        sigma = self.sigma
        # critic forward pass
        if self.asymmetric:
            critic_input = state
        else:
            critic_input = obs
        x_critic = self.critic_mlp(critic_input)
        value = self.value(x_critic)

        return mu, mu * 0 + sigma, value, extrin, extrin_gt

    def forward(self, input_dict):
        prev_actions = input_dict.get("prev_actions", None)
        rst = self._actor_critic(input_dict)
        mu, logstd, value, extrin, extrin_gt = rst
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            "prev_neglogp": torch.squeeze(prev_neglogp),
            "values": value,
            "entropy": entropy,
            "mus": mu,
            "sigmas": sigma,
            "extrin": extrin,
            "extrin_gt": extrin_gt,
        }
        return result