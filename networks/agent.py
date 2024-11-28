# https://github.com/ellenjxu/tinygym/model.py
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from networks.utils import Normalizer

def mlp(hidden_sizes: list[int], activation: nn.Module = nn.Tanh, output_activation: nn.Module = nn.Identity):
  layers = []
  for j in range(len(hidden_sizes)-1):
    act = activation if j < len(hidden_sizes)-2 else output_activation
    layers += [nn.Linear(hidden_sizes[j], hidden_sizes[j+1]), act()]
  return nn.Sequential(*layers)

class MLPGaussian(nn.Module):
  def __init__(self, obs_dim: int, hidden_sizes: list[int], act_dim: int, activation: nn.Module = nn.Tanh, log_std: float = 3.) -> None: # TODO: 0
    super(MLPGaussian, self).__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    self.log_std = torch.nn.Parameter(torch.full((act_dim,), log_std, dtype=torch.float32))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.obs_normalizer:
      x = self.obs_normalizer.norm(x)
    return self.mlp(x)
  
  def get_policy(self, obs: torch.Tensor) -> torch.Tensor:
    mean = self.forward(obs)
    mean = torch.tanh(mean)
    std = self.log_std.exp()
    return mean, std

  def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
    mean, std = self.get_policy(obs)
    action = mean if deterministic else torch.normal(mean, std)
    return action.detach().cpu().numpy().squeeze(0)

  def get_logprob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    mean, std = self.get_policy(obs)
    logprob = -0.5 * (((act - mean)**2) / std**2 + 2 * self.log_std + torch.log(torch.tensor(2*torch.pi)))
    entropy = (torch.log(std) + 0.5 * (1 + torch.log(torch.tensor(2*torch.pi))))
    assert logprob.shape == act.shape
    return logprob.sum(dim=-1), entropy.sum(dim=-1)

class MLPBeta(nn.Module):
  '''Beta distribution for bounded continuous control, output between 0 and 1'''
  def __init__(self, obs_dim, hidden_sizes, act_dim, activation=nn.Tanh, bias=True, act_bound: tuple[float, float] = (0, 1)):
    super(MLPBeta, self).__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim*2], activation)
    self.act_dim = act_dim
    self.act_bound = act_bound
    
  def forward(self, x: torch.Tensor):
    if self.obs_normalizer:
      x = self.obs_normalizer.norm(x)
    return self.mlp(x)
  
  def get_policy(self, obs: torch.Tensor):
    alpha_beta = self.forward(obs)
    alpha, beta = torch.split(alpha_beta, self.act_dim, dim=-1)
    alpha = F.softplus(alpha) + 1
    beta = F.softplus(beta) + 1
    return alpha, beta

  def get_action(self, obs: torch.Tensor, deterministic=False):
    alpha, beta = self.get_policy(obs)
    action = alpha / (alpha + beta) if deterministic else torch.distributions.Beta(alpha, beta).sample()
    action = action.detach()
    scaled_action = action * (self.act_bound[1] - self.act_bound[0]) + self.act_bound[0]
    return scaled_action.detach().cpu().numpy().squeeze(-1)

  def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
    unscaled_act = (act - self.act_bound[0]) / (self.act_bound[1] - self.act_bound[0])  
    alpha, beta = self.get_policy(obs)
    dist = torch.distributions.Beta(alpha, beta)
    logprob = dist.log_prob(unscaled_act)
    assert logprob.shape == act.shape
    entropy = dist.entropy()
    return logprob.sum(dim=-1), entropy.sum(dim=-1)

class MLPCritic(nn.Module):
  def __init__(self, obs_dim: int, hidden_sizes: list[int], activation: nn.Module = nn.Tanh) -> None:
    super().__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    self.return_normalizer = Normalizer(1) # value is scalar

  def forward(self, x: torch.Tensor, out_norm: bool = False) -> torch.Tensor:
    if self.obs_normalizer:
      x = self.obs_normalizer.norm(x)
    value = self.mlp(x)
    return self.return_normalizer.denorm(value) if not out_norm else value # default is unnormalized

class ActorCritic(nn.Module):
  def __init__(self, obs_dim: int, hidden_sizes: dict[str, list[int]], act_dim: int, 
              discrete: bool = False, shared_layers: bool = True, act_bound: tuple[float, float] = None) -> None:
    super().__init__()
    self.obs_normalizer = Normalizer(obs_dim)
    model_class = MLPCategorical if discrete else (MLPGaussian if not act_bound else MLPBeta)
    print('using model', model_class, 'with action bound', act_bound)
    
    if act_bound:
      self.actor = model_class(obs_dim, hidden_sizes["pi"], act_dim, act_bound=act_bound)
      act_dim *= 2
    else:
      self.actor = model_class(obs_dim, hidden_sizes["pi"], act_dim)
    self.critic = MLPCritic(obs_dim, hidden_sizes["vf"])

    # share normalizer with actor and critic
    self.actor.obs_normalizer = self.obs_normalizer
    self.critic.obs_normalizer = self.obs_normalizer

    if shared_layers and len(hidden_sizes["pi"]) > 1:
      self.shared = mlp([obs_dim] + hidden_sizes["pi"][:-1], nn.Tanh)
      self.actor.mlp = nn.Sequential(
        self.shared,
        mlp([hidden_sizes["pi"][-2], hidden_sizes["pi"][-1], act_dim], nn.Tanh)
      )
      self.critic.mlp = nn.Sequential(
        self.shared,
        mlp([hidden_sizes["vf"][-2], hidden_sizes["vf"][-1], 1], nn.Tanh)
      )

  def update_normalizers(self, obs, returns=None):
    """Update running statistics for normalization"""
    self.obs_normalizer.update(obs)
    if returns is not None:
      self.critic.return_normalizer.update(returns.reshape(-1, 1))

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    actor_out = self.actor(x) # normalization handled in forward pass
    critic_out = self.critic(x)
    return actor_out, critic_out