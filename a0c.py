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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AlphaZero(MCTS):
  """AlphaZero Continuous. Uses NN to guide MCTS search."""
  def __init__(self, model, exploration_weight=1, gamma=0.99, k=1, alpha=0.5, device='cpu'):
    super().__init__(exploration_weight, gamma, k, alpha)
    self.model = model # f_theta(s) -> pi(s), V(s)
    self.device = device

  def _value(self, state): # override methods to use NN
    action, value = self.model(state.to_tensor().to(self.device))
    return value

  # def _puct(self, state: State, action: np.ndarray):
  #   logprob, entropy = self.model.actor.get_logprob(
  #       state.to_tensor().to(self.device), 
  #       torch.tensor(action, device=self.device)
  #   )
  #   # print(self.Q[(state,action)], math.sqrt(self.Ns[state])/(self.N[(state,action)]+1))
  #   # TODO: check math.exp(logprob). Q values should be negative since returns are negative
  #   return self.Q[(state,action)] + self.exploration_weight * math.exp(logprob) * math.sqrt(self.Ns[state])/(self.N[(state,action)]+1)

class A0C:
  def __init__(self, env, model, lr=1e-3, gamma=0.9, tau=1,clip_range=0.2, epochs=1, ent_coeff=0.01, env_bs=1, device='cpu', debug=False):
    self.env = env
    self.env_bs = env_bs
    self.model = model.to(device)
    self.gamma = gamma
    self.tau = tau
    # self.lam = lam
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
    # self.mcts = MCTS() # TODO: use MCTS without NN for now

  def compute_return(self, rewards, dones, next_value):
    returns = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
    # if done, then return reward. Otherwise estimate future reward
      returns[t] = rewards[t] + self.gamma*(1-dones[t]) * next_value
      next_value = returns[t]
      # print(returns.flatten())
    # print('rewards', rewards.flatten(), 'returns', returns.flatten())
    return returns
  
  def evaluate_cost(self, states, returns, logprobs, mcts_probs):
    V = self.model.critic(states).squeeze()
    print(V[0], returns[0])
    value_loss = nn.MSELoss()(V, returns)
    # print(logprobs, mcts_probs)
    # print(mcts_probs.shape, logprobs.shape)

    # A0C policy loss uses REINFORCE trick to move distribution closer to normalized visit counts
    # print(mcts_probs.shape, logprobs.shape)
    # policy_loss = -torch.sum(mcts_probs * logprobs, dim=1).mean() # both mcts_probs and logprobs are (s,m) where m is the number of actions
    # TODO: set to 0 for now
    policy_loss = torch.tensor(0)
    
    l2_loss = sum(torch.norm(param) for param in self.model.parameters())
    return {"policy": policy_loss, "value": value_loss, "l2": l2_loss}
    
  @staticmethod
  def rollout(env, model, max_steps=1000, deterministic=False, device='cpu'):
    states, actions, rewards, dones, probs = [], [], [], [], []
    state, _ = env.reset()

    for _ in range(max_steps):
      # select best action based on MCTS
      s = CartState.from_array(state)
      best_action, _ = model.get_action(s, deterministic=True) # get single action and prob
      action, prob_dist = model.get_policy(s) # MCTS policy over all children actions
      next_state, reward, terminated, truncated, info = env.step(np.array([[best_action]]))
      # print('state', state)
      # print('action', action, 'best_action', best_action, 'prob_dist', prob_dist)
      # print('reward', reward)
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      probs.append(prob_dist)
      done = terminated or truncated
      dones.append(done)

      state = next_state
      if done:
        state, _ = env.reset()
        model.reset() # reset mcts tree
    return states, actions, rewards, dones, next_state, probs

  def train(self, max_iters=1000, n_episodes=10, n_steps=30):
    for i in range(max_iters):

      # collect data
      episode_rewards = []
      for _ in range(n_episodes):
        states, actions, rewards, dones, next_state, probs = self.rollout(self.env, self.mcts, n_steps, device=self.device)
        episode_rewards.append(np.sum(rewards))
        
        with torch.no_grad():
          state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
          next_state_tensor = torch.FloatTensor(next_state).to(self.device)
          next_value = self.model.critic(next_state_tensor).cpu().numpy().squeeze()
          action_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
          logprobs_tensor, _ = self.model.actor.get_logprob(state_tensor, action_tensor)
        
        returns = self.compute_return(np.array(rewards), np.array(dones), next_value)
        
        episode_dict = TensorDict(
          {
            "states": state_tensor,
            "returns": torch.FloatTensor(returns).to(self.device),
            "logprobs": logprobs_tensor,
            "probs": torch.FloatTensor(probs).to(self.device),
          },
          batch_size=len(states)
        )
        self.replay_buffer.extend(episode_dict)

      # update
      for _ in range(self.epochs):
        batch = self.replay_buffer.sample(batch_size=n_episodes*n_steps)
        costs = self.evaluate_cost(
          batch['states'],
          batch['returns'],
          batch['logprobs'],
          batch['probs']
        )
        loss = costs["value"] # test value net convergence
        # loss = costs["policy"] + 0.5*costs["value"] + 1e-4 * costs["l2"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      avg_reward = np.mean(episode_rewards)
      if self.debug:
        print(f"actor loss {costs['policy'].item():.3f} value loss {costs['value'].item():.3f} mean action {np.mean(abs(np.array(actions)))}")
      print(f"iter {i}, reward {avg_reward:.3f}, t {time.time()-self.start:.2f}")
      self.hist.append((i, avg_reward))

    print(f"Total time: {time.time() - self.start}")
    return self.model.actor, self.hist

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_iters", type=int, default=1000)
  parser.add_argument("--n_eps", type=int, default=10)
  parser.add_argument("--n_steps", type=int, default=30)
  parser.add_argument("--env_bs", type=int, default=1) # TODO: batch
  parser.add_argument("--save_model", default=False)
  parser.add_argument("--noise_mode", default=None)
  parser.add_argument("--debug", default=True)
  args = parser.parse_args()

  print(f"training a0c with max_iters {args.max_iters}") 
  env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=args.env_bs)
  model = ActorCritic(env.observation_space.shape[-1], {"pi": [32], "vf": [32]}, env.action_space.shape[-1]) #, act_bound=(-10, 10))
  a0c = A0C(env, model, env_bs=args.env_bs, debug=args.debug)
  best_model, hist = a0c.train(args.max_iters, args.n_eps, args.n_steps)

  # print(f"rolling out best model") 
  # env = gym.make("CartLatAccel-v0", noise_mode=args.noise_mode, env_bs=1, render_mode="human")
  # states, actions, rewards, dones, next_state, _= a0c.rollout(env, best_model, max_steps=200, deterministic=True)
  # print(f"reward {sum(rewards)}")

  # # single rollout
  # env = gym.make("CartLatAccel-v0", noise_mode=args.noise_mode, env_bs=1, render_mode="human")
  # state, _ = env.reset()
  # max_steps = 200
  # rewards = []
  # for _ in range(max_steps):
  #   state_tensor = torch.FloatTensor(state).unsqueeze(0)
  #   action = best_model.get_action(state_tensor, deterministic=True)
  #   next_state, reward, terminated, truncated, info = env.step(action)
  #   done = terminated or truncated
  #   state = next_state
  #   rewards.append(reward)
  #   if done:
  #     env.close()
  #     break
  # print(f"reward {sum(rewards)}")

  if args.save_model:
    os.makedirs('out', exist_ok=True)
    torch.save(best_model, 'out/best.pt')
