from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class CableActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )

        self.actor_mean = nn.Sequential(nn.Linear(256, action_dim), nn.Tanh())
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state: torch.Tensor):
        features = self.features(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std).clamp(min=1e-6, max=1.0)
        value = self.critic(features)
        return action_mean, action_std, value


class ImprovedPPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01,
    ):
        self.actor_critic = CableActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.actor_critic.eval()

    def get_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            if torch.isnan(state_tensor).any():
                state_tensor = torch.nan_to_num(state_tensor, nan=0.0)

            action_mean, action_std, value = self.actor_critic(state_tensor)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            action = torch.clamp(action, min=-0.1, max=0.1)
            return action.numpy(), log_prob.item(), value.item()
