"""Train A0C model on cartlataccel, multiprocessed MCTS rollouts"""

import os
import time
import torch
import argparse
import numpy as np
import pickle as pkl
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import gym_cartlataccel
import multiprocessing as mp
import matplotlib.pyplot as plt
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict

from networks.mcts import MCTS, A0CModel, Node
from networks.agent import ActorCritic
from utils.helpers import sample_rollout, plot_losses, RunningStats

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def mcts_worker(env_params, model_params, seed, max_steps=1000):
  """
  Worker function for an instance of parallel MCTS run.
  
  Args:
  - env_params: Dict of parameters for Gymnasium env
  - model_params: State dict of model parameters
  - max_steps: Max steps per episode
  
  Returns:
  - Episode data (states, returns, mcts_states, mcts_actions, mcts_counts)
  """
  np.random.seed(seed)
  torch.manual_seed(seed)

  env = gym.make("CartLatAccel-v1", **env_params)
  model = ActorCritic(env.observation_space.shape[-1], {"pi": [32], "vf": [32]}, env.action_space.shape[-1])
  if model_params:
    model.load_state_dict(model_params)
  mcts = A0CModel(model, exploration_weight=1e-1, gamma=0.99, k=1, alpha=0.5)
  
  states, returns, mcts_states, mcts_actions, mcts_counts = [], [], [], [], []
  state, _ = env.reset()

  for _ in range(max_steps):
    s = Node(state)
    # get action from mcts simulation
    action = mcts.get_action(env, s, d=10, n=100, deterministic=True) # TODO: cleanup into A0C params
    next_state, reward, terminated, truncated, info = env.step(np.array([action]))
    states.append(state)

    # get policy statistics
    actions, norm_counts, max_q = mcts.get_policy(s)
    actions = np.array([a.value for a in actions])  # convert back to float
    mcts_counts.append(norm_counts)
    mcts_actions.append(actions)
    mcts_states.append([state]*len(actions))
    returns.append(max_q)
    
    state = next_state
    if terminated or truncated:
      state, _ = env.reset()
      mcts.reset()
  
  return states, returns, mcts_states, mcts_actions, mcts_counts

class A0C:
  def __init__(self, env, model, tau=1, lr=1e-1, epochs=10, bs=512, ent_coeff=0.01, env_bs=1, device='cpu', debug=False, n_trees=None, noise_mode=None):
    self.env = env
    self.env_bs = env_bs
    self.noise_mode = noise_mode
    self.model = model.to(device)
    self.epochs = epochs
    self.tau = tau
    self.ent_coeff = ent_coeff
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.bs = bs

    self.n_trees = n_trees if n_trees is not None else mp.cpu_count()
    
    self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=100000, device=device))
    self.start = time.time()
    self.device = device
    self.debug = debug
    self.running_stats = RunningStats()
    self.hist = {'iter': [], 'reward': [], 'value_loss': [], 'policy_loss': [], 'total_loss': []}

  def _normalize_return(self, returns):
    for r in returns:
      self.running_stats.push(r)

    self.mu = self.running_stats.mean()
    self.sigma = self.running_stats.standard_deviation()
    self.model.critic.mean = torch.tensor(self.mu, device=self.device)
    self.model.critic.std = torch.tensor(self.sigma, device=self.device)

    normalized_returns = (returns - self.mu) / (self.sigma + 1e-8)
    return normalized_returns

  def value_loss(self, states, returns):
    V = self.model.critic(states, normalize=True).squeeze()
    normalized_returns = self._normalize_return(returns)
    value_loss = nn.L1Loss()(V, normalized_returns)
    if self.debug:
      print(f"value {V[0]} return {normalized_returns[0]}")
      print(f"value range: {V.min():.3f} {V.max():.3f}, returns range: {normalized_returns.min():.3f} {normalized_returns.max():.3f}")
    return value_loss

  def policy_loss(self, mcts_states, mcts_actions, mcts_counts):
    logcounts = torch.log(torch.tensor(mcts_counts, device=self.device))
    logprobs, entropy = self.model.actor.get_logprob(mcts_states, mcts_actions.unsqueeze(-1))
    with torch.no_grad():
      error = logprobs - self.tau * logcounts
    policy_loss = (error * logprobs).mean() - self.ent_coeff * entropy.mean()
    if self.debug:
      print(f"mean absolute error: {torch.mean(torch.abs(error))}")
      # print(f"std: {self.model.actor.log_std.item()}")
    return policy_loss

  def l2_loss(self):
    l2_loss = sum(torch.norm(param) for param in self.model.parameters())
    return l2_loss

  def mcts_rollout(self, n_steps=30):
    """Rolls out n_trees parallel MCTS simulations each for n_steps"""
    env_params = {'noise_mode': self.noise_mode, 'env_bs': self.env_bs}
    model_params = self.model.state_dict()
    
    with mp.Pool(processes=self.n_trees) as pool:
      seeds = np.random.randint(0, 2**32 - 1, size=self.n_trees)
      results = pool.starmap(
        mcts_worker, 
        [(env_params, model_params, seed, n_steps) for seed in seeds]
      )
    
    all_states, all_returns = [], []
    all_mcts_states, all_mcts_actions, all_mcts_counts = [], [], []
    
    for states, returns, mcts_states, mcts_actions, mcts_counts in results:
      all_states.extend(states)
      all_returns.extend(returns)
      all_mcts_states.extend(mcts_states)
      all_mcts_actions.extend(mcts_actions)
      all_mcts_counts.extend(mcts_counts)

    # print(torch.FloatTensor(all_states).shape)
    # print(torch.FloatTensor(all_mcts_actions).shape)
    # print(torch.FloatTensor(all_mcts_states).shape)

    episode_dict = TensorDict({
      "states": torch.FloatTensor(all_states).to(self.device),
      "returns": torch.FloatTensor(all_returns).to(self.device),
      "mcts_states": torch.FloatTensor(all_mcts_states).to(self.device),
      "mcts_actions": torch.FloatTensor(all_mcts_actions).to(self.device),
      "mcts_counts": torch.FloatTensor(all_mcts_counts).to(self.device),
    }, batch_size=len(all_states))
    
    return episode_dict

  def train(self, max_iters=1000, n_steps=30):
    for i in range(max_iters):
      # collect data
      mcts_start = time.time()
      episode_dict = self.mcts_rollout(n_steps)
      mcts_time = time.time() - mcts_start
      self.replay_buffer.extend(episode_dict)

      # update model
      train_start = time.time()
      for _ in range(self.epochs):
        batch = self.replay_buffer.sample(batch_size=self.bs)
        value_loss = self.value_loss(batch['states'], batch['returns'])
        policy_loss = self.policy_loss(batch['mcts_states'], batch['mcts_actions'], batch['mcts_counts'])
        l2_loss = self.l2_loss()
        loss = policy_loss + 0.5 * value_loss + 1e-4 * l2_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      train_time = time.time() - train_start

      # evaluate on fixed set of episodes
      eval_start = time.time()
      avg_reward = np.mean(sample_rollout(self.model.actor, self.env, n_episodes=5, n_steps=200))
      eval_time = time.time() - eval_start

      self.hist['iter'].append(i)
      self.hist['reward'].append(avg_reward)
      self.hist['value_loss'].append(value_loss.item())
      self.hist['policy_loss'].append(abs(policy_loss.item()))
      self.hist['total_loss'].append(policy_loss.item() + 0.5 * value_loss.item() + 1e-4 * l2_loss.item())

      print(f"iter {i}, reward {avg_reward:.3f}, total time {time.time()-self.start:.2f}s")
      if self.debug:
        print(f"actor loss {abs(policy_loss.item()):.3f} value loss {value_loss.item():.3f} l2 loss {l2_loss.item():.3f}")
        print(f"mean action {np.abs(batch['mcts_actions']).mean()}")
        print(f"runtimes - mcts: {mcts_time:.2f}s, train: {train_time:.2f}s, eval: {eval_time:.2f}s\n")

    print(f"Total time: {time.time() - self.start}")
    return self.model, self.hist

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_iters", type=int, default=10)
  parser.add_argument("--n_steps", type=int, default=30)
  parser.add_argument("--env_bs", type=int, default=1)
  parser.add_argument("--n_trees", type=int, default=16)
  parser.add_argument("--save", default=True)
  parser.add_argument("--noise_mode", default=None)
  parser.add_argument("--debug", default=False)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  print(f"Training A0C with max_iters {args.max_iters} and {args.noise_mode} noise")
  
  env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=args.env_bs)
  model = ActorCritic(env.observation_space.shape[-1], {"pi": [32], "vf": [32]}, env.action_space.shape[-1])
  
  a0c = A0C(env, model, env_bs=args.env_bs, debug=args.debug, n_trees=args.n_trees, noise_mode=args.noise_mode)
  best_model, hist = a0c.train(args.max_iters, args.n_steps)

  print("Rollout out best actor")
  env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=1, render_mode="human")
  rewards = sample_rollout(best_model.actor, env, n_episodes=5, n_steps=200)
  env.close()
  print(f"reward {np.mean(rewards):.3f}, std {np.std(rewards):.3f}")
  
  if args.save:
    os.makedirs('out', exist_ok=True)
    torch.save(best_model, 'out/best.pt')
    with open('out/history.pkl', 'wb') as f:
      pkl.dump(hist, f)
    plot_losses(hist, save_path="out/loss_curve.png")
  else:
    plot_losses(hist)