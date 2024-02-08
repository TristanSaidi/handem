import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from handem.algo.models.architectures.mlp import MLP
from handem.algo.models.architectures.transformer import Block

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
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

class MLPRegressor(nn.Module):
    def __init__(self, kwargs):
        super(MLPRegressor, self).__init__()
        self.autoregressive = kwargs.pop("autoregressive")
        # observation hist
        proprio_dim = kwargs.pop("proprio_dim")
        proprio_hist_len = kwargs.pop("proprio_hist_len")
        # vertex prediction
        vertex_dim = kwargs.pop("vertex_dim")
        n_vertices = kwargs.pop("n_vertices")
        # input size
        if self.autoregressive:
            input_size = proprio_dim * proprio_hist_len + vertex_dim * n_vertices # proprio_hist + previous vertex prediction
        else:
            input_size = proprio_dim * proprio_hist_len
        units = kwargs.pop("units")
        units.append(n_vertices * vertex_dim)
        self.mlp = MLP(units, input_size=input_size)

    def forward(self, proprio_hist, vertex_pred):
        # x: tensor of size (B x proprio_hist_len x proprio_dim)
        if self.autoregressive:
            x = torch.cat([proprio_hist.flatten(1), vertex_pred.flatten(1)], dim=1)
        else:
            x = proprio_hist.flatten(1)
        x = self.mlp(x)
        return x

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

class TransformerDiscriminator(nn.Module):

    def __init__(self, n_layer, n_head, n_embd, proprio_hist_len, proprio_dim, num_classes, dropout, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.history_embedding = nn.Linear(proprio_dim, n_embd)
        self.position_embedding_table = nn.Embedding(proprio_hist_len, n_embd)
        self.blocks = nn.Sequential(*[Block(n_head, n_embd, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.predict_class = nn.Linear(n_embd, num_classes)
        self.device = device
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, proprio_hist, targets=None):
        # x: tensor of size (batch_size x proprio_hist_len x proprio_dim)
        # targets: tensor of size (batch_size x proprio_hist_len)
        _, proprio_hist_len, _ = proprio_hist.shape

        hist_emb = self.history_embedding(proprio_hist) # (batch_size x proprio_hist_len x proprio_dim) --> (batch_size x proprio_hist_len x n_embd)
        pos_emb = self.position_embedding_table(torch.arange(proprio_hist_len, device=self.device)) # (proprio_hist_len x proprio_hist_len)
        x = hist_emb + pos_emb # (batch_size x proprio_hist_len x n_embd)
        x = self.blocks(x) # (batch_size x proprio_hist_len x n_embd)
        x = self.ln_f(x) # (batch_size x proprio_hist_len x n_embd)
        logits = self.predict_class(x) # (batch_size x proprio_hist_len x num_classes)

        if targets is None:
            return logits
        else:
            batch_size, proprio_hist_len, num_classes = logits.shape
            logits = logits.view(batch_size*proprio_hist_len, num_classes)
            targets = targets.reshape(batch_size*proprio_hist_len)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


class TransformerRegressor(nn.Module):

    def __init__(self, n_layer, n_head, n_embd, proprio_hist_len, proprio_dim, n_vertices, vertex_dim, autoregressive, dropout, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.history_embedding = nn.Linear(proprio_dim, n_embd)
        self.position_embedding_table = nn.Embedding(proprio_hist_len, n_embd)
        self.blocks = nn.Sequential(*[Block(n_head, n_embd, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.predict = nn.Linear(n_embd, n_vertices * vertex_dim)
        self.device = device
        self.autoregressive = autoregressive
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, proprio_hist, vertex_preds):
        # x: tensor of size (batch_size x proprio_hist_len x proprio_dim)
        # targets: tensor of size (batch_size x proprio_hist_len)
        if self.autoregressive:
            raise NotImplementedError("Autoregressive not implemented yet for Transformer Regressor")
        _, proprio_hist_len, _ = proprio_hist.shape

        hist_emb = self.history_embedding(proprio_hist) # (batch_size x proprio_hist_len x proprio_dim) --> (batch_size x proprio_hist_len x n_embd)
        pos_emb = self.position_embedding_table(torch.arange(proprio_hist_len, device=self.device)) # (proprio_hist_len x proprio_hist_len)
        x = hist_emb + pos_emb # (batch_size x proprio_hist_len x n_embd)
        x = self.blocks(x) # (batch_size x proprio_hist_len x n_embd)
        x = self.ln_f(x) # (batch_size x proprio_hist_len x n_embd)
        predictions = self.predict(x) # (batch_size x proprio_hist_len x n_vertices * vertex_dim)
        # take the last prediction
        predictions = predictions[:, -1, :]
        return predictions

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

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