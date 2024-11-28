"""Implementation of AlphaZero Continuous (A0C)"""

import torch
import math
from mcts import MCTS, State
from mcts_cart import CartState

import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import gym_cartlataccel
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from model import ActorCritic
from utils import RunningStats

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AlphaZero(MCTS):
  """AlphaZero Continuous. Uses NN to guide MCTS search."""
  def __init__(self, model, exploration_weight=1e-3, gamma=0.99, k=1, alpha=0.5, device='cpu'):
    super().__init__(exploration_weight, gamma, k, alpha)
    self.model = model # f_theta(s) -> pi(s), V(s)
    self.device = device

  def _value(self, state): # override methods to use NN
    with torch.no_grad():
      _, value = self.model(state.to_tensor().to(self.device))
    return value

  def _puct(self, state: State, action: np.ndarray):
    state_tensor = state.to_tensor().to(self.device)
    action_tensor = torch.tensor([action], device=self.device)
    with torch.no_grad():
      logprob, entropy = self.model.actor.get_logprob(state_tensor, action_tensor)
    return self.Q[(state,action)] + self.exploration_weight * math.exp(logprob) * math.sqrt(self.Ns[state])/(self.N[(state,action)]+1)

class A0C:
  def __init__(self, env, model, lr=1e-1, gamma=0.99, tau=1,clip_range=0.2, epochs=10, ent_coeff=0.01, env_bs=1, device='cpu', debug=False):
    self.env = env
    self.env_bs = env_bs
    self.model = model.to(device)
    self.gamma = gamma
    self.tau = tau
    self.clip_range = clip_range
    self.epochs = epochs
    self.ent_coeff = ent_coeff
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=100000, device=device))
    self.hist = []
    self.start = time.time()
    self.device = device
    self.debug = debug
    self.mcts = AlphaZero(model, device=device)
    # self.mcts = MCTS()
    self.running_stats = RunningStats()

  def _compute_return(self, states):
    states = np.array(states)
    returns = []
    for state in states:
      s = CartState.from_array(state)
      q_values = [self.mcts.Q[(s,a)] for a in self.mcts.children[s]] # a0c uses max_a Q vs environment rewards
      v_target = max(q_values) if q_values else 0
      returns.append(v_target)
    return np.array(returns)

  def _normalize_return(self, returns):
    for r in returns:
      self.running_stats.push(r)

    self.mu = self.running_stats.mean()
    self.sigma = self.running_stats.standard_deviation()
    self.model.critic.mean = torch.tensor(self.mu, device=self.device) # update value net
    self.model.critic.std = torch.tensor(self.sigma, device=self.device)

    normalized_returns = (returns - self.mu) / (self.sigma + 1e-8)
    return normalized_returns
  
  def _evaluate_cost(self, states, returns, mcts_states, mcts_actions, mcts_counts):
    # when evaluating cost use normalized; unnorm only used in mcts search
    V = self.model.critic(states, normalize=True).squeeze()
    normalized_returns = self._normalize_return(returns)
    value_loss = nn.MSELoss()(V, normalized_returns)
    # if self.debug:
      # print(f"value {V[0]} return {normalized_returns[0]}")
      # print(f"value range: {V.min():.3f} {V.max():.3f}, returns range: {normalized_returns.min():.3f} {normalized_returns.max():.3f}")
    
    logcounts = torch.log(torch.tensor(mcts_counts, device=self.device))
    logprobs, entropy = self.model.actor.get_logprob(mcts_states, mcts_actions.unsqueeze(-1))
    with torch.no_grad():
      error = logprobs - self.tau * logcounts # pos/neg tells you whether to push action up or down
    policy_loss = (error * logprobs).mean()
    # if self.debug:
      # print(torch.exp(logcounts).sum(dim=-1)) # shape is the same as first dim of logcounts. make sure logcounts sum to 1
      # print(f"logprobs: {logprobs[0]}, logcounts: {logcounts[0]}, error: {error[0]}")
      # print(f"mean absolute error: {torch.mean(torch.abs(error))}")
      # print(f"std: {self.model.actor.log_std}")
    
    policy_loss -= self.ent_coeff * entropy.mean()
    
    l2_loss = sum(torch.norm(param) for param in self.model.parameters())
    return {"policy": policy_loss, "value": value_loss, "l2": l2_loss}
  
  @staticmethod
  def rollout(env, model, max_steps=1000, deterministic=False, device='cpu'):
    states, rewards, mcts_states, mcts_actions, mcts_counts = [], [], [], [], []
    state, _ = env.reset()

    for _ in range(max_steps):
      s = CartState.from_array(state)
      best_action, _ = model.get_action(s, deterministic=True) # get single action and prob
      next_state, reward, terminated, truncated, info = env.step(np.array([[best_action]]))
      states.append(state)
      rewards.append(reward)
      done = terminated or truncated

      actions, counts = model.get_policy(s)
      mcts_counts.append(counts)
      mcts_actions.append(actions)
      mcts_states.append([state]*len(actions))

      state = next_state
      if done:
        state, _ = env.reset()
        model.reset() # reset mcts tree
    return states, rewards, mcts_states, mcts_actions, mcts_counts

  def train(self, max_iters=1000, n_episodes=10, n_steps=30):
    for i in range(max_iters):
      episode_rewards = []
      for _ in range(n_episodes):
        # collect data
        states, rewards, mcts_states, mcts_actions, mcts_counts = self.rollout(self.env, self.mcts, n_steps, device=self.device)
        returns = self._compute_return(states)
        episode_rewards.append(sum(rewards)) # for logging
        
        # add to replay buffer
        episode_dict = TensorDict(
          {
            "states": states,
            "returns": returns,
            "mcts_states": np.array(mcts_states),
            "mcts_actions": np.array(mcts_actions),
            "mcts_counts": np.array(mcts_counts),
          },
          batch_size=len(states)
        )
        self.replay_buffer.extend(episode_dict)

      # update
      for _ in range(self.epochs):
        batch = self.replay_buffer.sample(batch_size=n_steps*n_episodes) # TODO: taking entire batch of 300
        costs = self._evaluate_cost(batch['states'], batch['returns'], batch['mcts_states'], batch['mcts_actions'], batch['mcts_counts'])
        loss = costs["policy"] + 0.5 * costs["value"] + 1e-4 * costs["l2"]
        self.optimizer.zero_grad()
        loss.backward()

        # # check grad norms
        # total_grad_norm = 0
        # for name, param in self.model.actor.named_parameters():
        #   if param.grad is not None:
        #     grad_norm = param.grad.norm().item()
        #     total_grad_norm += grad_norm
        #     if self.debug:
        #       print(f"Gradient norm for {name}: {grad_norm:.5f}")
        self.optimizer.step()

      avg_reward = np.mean(episode_rewards)
      if self.debug:
        print(f"actor loss {costs['policy'].item():.3f} value loss {costs['value'].item():.3f} l2 loss {costs['l2'].item():.3f}")
      print(f"iter {i}, reward {avg_reward:.3f}, t {time.time()-self.start:.2f}")
      self.hist.append((i, avg_reward))

    print(f"Total time: {time.time() - self.start}")
    return self.model.actor, self.hist

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_iters", type=int, default=10)
  parser.add_argument("--n_eps", type=int, default=10)
  parser.add_argument("--n_steps", type=int, default=30)
  parser.add_argument("--env_bs", type=int, default=1) # TODO: batch
  parser.add_argument("--save_model", default=False)
  parser.add_argument("--noise_mode", default=None)
  parser.add_argument("--debug", default=True)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  print(f"training a0c with max_iters {args.max_iters}") 
  env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=args.env_bs)
  model = ActorCritic(env.observation_space.shape[-1], {"pi": [32], "vf": [32]}, env.action_space.shape[-1]) #, act_bound=(-1, 1))
  a0c = A0C(env, model, env_bs=args.env_bs, debug=args.debug)
  best_model, hist = a0c.train(args.max_iters, args.n_eps, args.n_steps)

  # run value net online planner
  # from mcts_cart import run_mcts
  # env = gym.make("CartLatAccel-v1", render_mode="human", noise_mode=args.noise_mode)
  # run_mcts(a0c.mcts, env, max_steps=200, search_depth=10, n_sims=100, seed=args.seed)

  # run actor net model
  env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=1, render_mode="human")
  state, _ = env.reset()
  max_steps = 200
  rewards = []
  for _ in range(max_steps):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = best_model.get_action(state_tensor, deterministic=True)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state
    rewards.append(reward)
    if done:
      env.close()
      break
  print(f"reward {sum(rewards)}")

  if args.save_model:
    os.makedirs('out', exist_ok=True)
    torch.save(best_model, 'out/best.pt')
