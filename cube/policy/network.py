"""Actor network of the checkpoints (the critic is stored but unused here)."""

import time

import torch
from torch import nn


def mlp(sizes, last_gain=1.0):
    layers = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.orthogonal_(linear.weight, gain=last_gain if i == len(sizes) - 2 else 2**0.5)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    def forward(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-8, 8)


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, logstd=-0.7):
        super().__init__()
        self.inference_seconds = 0.0
        self.inference_calls = 0
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.norm = Normalizer(obs_dim)
        self.actor = mlp([obs_dim, 512, 256, 128, action_dim], 0.01)
        self.critic = mlp([obs_dim, 512, 256, 128, 1])
        self.logstd = nn.Parameter(torch.full((action_dim,), float(logstd)))

    def mean_action(self, obs):
        return self.actor(self.norm(obs))

    def predict_action(self, obs):
        """Run the actor on its device and return bounded CPU actions for MuJoCo.

        Timing includes observation/action transfers; .cpu() waits for CUDA work.
        """
        started = time.perf_counter()
        device = self.norm.mean.device
        with torch.no_grad():
            action = self.mean_action(obs.to(device)).clamp(-1, 1).cpu()
        self.inference_seconds += time.perf_counter() - started
        self.inference_calls += 1
        return action


def load_policy(path, device="cpu"):
    cp = torch.load(path, map_location="cpu", weights_only=False)
    policy = Policy(cp["obs_dim"], cp["action_dim"])
    policy.load_state_dict(cp["policy"])
    policy.to(device)
    policy.eval()
    return cp, policy
