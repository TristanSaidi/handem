import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from handem.algo.models.architectures.mlp import MLP
from handem.algo.models.architectures.transformer import GPT2Model

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
        # observation hist
        proprio_dim = kwargs.pop("proprio_dim")
        proprio_hist_len = kwargs.pop("proprio_hist_len")
        # vertex prediction
        vertex_dim = kwargs.pop("vertex_dim")
        n_vertices = kwargs.pop("n_vertices")
        # input size
        input_size = proprio_dim * proprio_hist_len + vertex_dim * n_vertices # proprio_hist + previous vertex prediction
        units = kwargs.pop("units")
        units.append(n_vertices * vertex_dim)
        self.mlp = MLP(units, input_size=input_size)

    def forward(self, proprio_hist, vertex_pred):
        # x: tensor of size (B x proprio_hist_len x proprio_dim)
        x = torch.cat([proprio_hist.flatten(1), vertex_pred.flatten(1)], dim=1)
        x = self.mlp(x)
        return x

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

class GPT2Discriminator(nn.Module):

    def __init__(
                self,
                obs_dim,
                hidden_size,
                num_classes,
                proprio_hist_len,
                **kwargs
        ):
            super(GPT2Discriminator, self).__init__()


            self.obs_dim = obs_dim

            self.hidden_size = hidden_size
            config = transformers.GPT2Config(
                vocab_size=1,  # doesn't matter -- we don't use the vocab
                n_embd=hidden_size,
                **kwargs
            )

            # note: the only difference between this GPT2Model and the default Huggingface version
            # is that the positional embeddings are removed (since we'll add those ourselves)
            self.transformer = GPT2Model(config)

            self.embed_position = nn.Embedding(proprio_hist_len, hidden_size)
            self.embed_proprio_hist = torch.nn.Linear(self.obs_dim, hidden_size)

            self.embed_ln = nn.LayerNorm(hidden_size)

            # note: we don't predict states or returns for the paper
            self.predict_class = nn.Linear(hidden_size, num_classes)

    def forward(self, proprio_hist, attention_mask=None):

        batch_size, seq_length = proprio_hist.shape[0], proprio_hist.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(proprio_hist.device)
    
        positions = torch.arange(0, seq_length, dtype=torch.long, device=proprio_hist.device)

        # embed each modality with a different head
        proprio_hist_embeddings = self.embed_proprio_hist(proprio_hist)
        position_embeddings = self.embed_position(positions)

        # time embeddings are treated similar to positional embeddings
        proprio_hist_embeddings = proprio_hist_embeddings + position_embeddings

        proprio_hist_embeddings = self.embed_ln(proprio_hist_embeddings)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=proprio_hist_embeddings,
            attention_mask=attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        # we only care about the most recent prediction
        x = x[:, -1, :]
        # predict the class
        class_logits = self.predict_class(x)
        log_softmax = F.log_softmax(class_logits, dim=-1)
        return log_softmax

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