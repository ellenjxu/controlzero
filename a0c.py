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
  def __init__(self, model, exploration_weight=1e-3, gamma=0.99, k=1, alpha=0.5, device='cpu'): # small exploratoin weight for reward
    super().__init__(exploration_weight, gamma, k, alpha)
    self.model = model # f_theta(s) -> pi(s), V(s)
    self.device = device

  def _value(self, state): # override methods to use NN
    action, value = self.model(state.to_tensor().to(self.device))
    return value

  def _puct(self, state: State, action: np.ndarray):
    logprob, entropy = self.model.actor.get_logprob(
        state.to_tensor().to(self.device), 
        torch.tensor(action, device=self.device)
    )
    # Q values are small (1e-3) so exploration factor should be quite small too
    # print(self.Q[(state,action)], math.sqrt(self.Ns[state])/(self.N[(state,action)]+1))
    # TODO: check math.exp(logprob). Q values should be negative since returns are negative
    return self.Q[(state,action)] + self.exploration_weight * math.exp(logprob) * math.sqrt(self.Ns[state])/(self.N[(state,action)]+1)

class A0C:
  def __init__(self, env, model, lr=3e-4, gamma=0.99, lam=0.99, clip_range=0.2, epochs=1, n_steps=30, ent_coeff=0.01, bs=30, env_bs=1, device='cpu', debug=False):
  # def __init__(self, env, model, lr=3e-4, gamma=0.99, lam=0.99, clip_range=0.2, epochs=10, n_steps=2048, ent_coeff=0.001, bs=64, env_bs=1, device='cpu', debug=False):
    self.env = env
    self.env_bs = env_bs
    self.model = model.to(device)
    self.gamma = gamma
    self.lam = lam
    self.clip_range = clip_range
    self.epochs = epochs
    self.n_steps = n_steps
    self.ent_coeff = ent_coeff
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000, device=device), batch_size=bs) # TODO: 100K
    self.bs = bs
    self.hist = []
    self.start = time.time()
    self.device = device
    self.debug = debug
    # self.mcts = AlphaZero(model, device=device)
    self.mcts = MCTS() # TODO: use MCTS without NN for now

  def compute_return(self, rewards, dones, next_value):
    returns = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
    # if done, then return reward. Otherwise estimate future reward
      returns[t] = rewards[t] + self.gamma*(1-dones[t])*next_value 
      next_value = returns[t]
    return returns
  
  def evaluate_cost(self, states, returns, logprobs, probs): # probs = pi_tree (normalized counts)
    V = self.model.critic(states).squeeze()
    value_loss = nn.MSELoss()(V, returns)
    # A0C policy loss uses REINFORCE trick to move distribution closer to normalized visit counts
    # print(probs.shape, logprobs.shape)
    policy_loss = torch.sum(-probs * logprobs, dim=1).mean() # both probs and logprobs are (s,m) where m is the number of actions
    l2_loss = sum(torch.norm(param) for param in self.model.parameters())
    return {"policy": policy_loss, "value": value_loss, "l2": l2_loss}
    
  @staticmethod
  def rollout(env, model, max_steps=1000, deterministic=False, device='cpu'):
    states, actions, rewards, dones, probs = [], [], [], [], []
    state, _ = env.reset()

    for _ in range(max_steps):
      # select best action based on MCTS
      s = CartState.from_array(state[0])
      best_action, _ = model.get_action(s) # get single action and prob
      action, prob_dist = model.get_policy(s) # MCTS policy over all children actions
      next_state, reward, terminated, truncated, info = env.step(np.array([[best_action]]))
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

  def train(self, max_evals=1000):
    eps = 0
    while True:
      # rollout
      start = time.perf_counter()
      # use MCTS instead of actor to rollout. Has one extra probs state from on MCTS (pi_tree)
      states, actions, rewards, dones, next_state, probs = self.rollout(self.env, self.mcts, self.n_steps, device=self.device)
      rollout_time = time.perf_counter()-start
      # print(states, actions, rewards, dones, next_state, probs)

      # since we are not optimizing policy directly, we use returns for SL instead of calculating ratio using GAE
      start = time.perf_counter()
      with torch.no_grad():
        state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        next_value = self.model.critic(next_state_tensor).cpu().numpy().squeeze()
        action_tensor = torch.FloatTensor(np.array(actions)).unsqueeze(-1).to(self.device)
        # print("action_tensor", action_tensor.shape)
        logprobs_tensor, _ = self.model.actor.get_logprob(state_tensor, action_tensor)
      returns = self.compute_return(np.array(rewards), np.array(dones), next_value)
      gae_time = time.perf_counter()-start

      # add to buffer
      start = time.perf_counter()
      episode_dict = TensorDict( # add (s_i, pi_i, z_i) to buffer where s_i = state, pi_i = MCTS policy, z_i = return
        {
          "states": state_tensor,
          # "actions": action_tensor,
          "returns": torch.FloatTensor(returns).to(self.device),
          # "advantages": torch.FloatTensor(advantages).to(self.device),
          "logprobs": logprobs_tensor,
          "probs": torch.FloatTensor(probs).to(self.device),
        },
        batch_size=self.n_steps
      )
      self.replay_buffer.extend(episode_dict)
      buffer_time = time.perf_counter() - start

      # update
      start = time.perf_counter()
      for _ in range(self.epochs):
        for i, batch in enumerate(self.replay_buffer): # randomly sample a batch from buffer
          costs = self.evaluate_cost(
            batch['states'], 
            # batch['actions'],
            batch['returns'],
            # batch['advantages'],
            batch['logprobs'],
            batch['probs']
          )
          loss = 100*costs["policy"] + 1*costs["value"] + 1e-4 * costs["l2"] # highway change paper used (100,1,1e-4)
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          break
      self.replay_buffer.empty() # clear buffer
      update_time = time.perf_counter() - start

      # debug info
      if self.debug:
        print(f"policy loss {costs['policy'].item():.3f} value loss {costs['value'].item():.3f} mean action {np.mean(abs(np.array(actions)))}")
        print(f"Runtimes: rollout {rollout_time:.3f}, gae {gae_time:.3f}, buffer {buffer_time:.3f}, update {update_time:.3f}")

      eps += self.env_bs
      avg_reward = np.sum(rewards)/self.env_bs

      print(f"eps {eps:.2f}, reward {avg_reward:.3f}, t {time.time()-self.start:.2f}")
      self.hist.append((eps, avg_reward))
      if eps > max_evals:
        print(f"Total time: {time.time() - self.start}")
        break

    return self.model.actor, self.hist

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_evals", type=int, default=30000)
  parser.add_argument("--env_bs", type=int, default=1000)
  parser.add_argument("--save_model", default=False)
  parser.add_argument("--noise_mode", default=None)
  parser.add_argument("--debug", default=True)
  args = parser.parse_args()

  print(f"training a0c with max_evals {args.max_evals}") 
  env = gym.make("CartLatAccel-v0", noise_mode=args.noise_mode, env_bs=args.env_bs)
  model = ActorCritic(env.observation_space.shape[-1], {"pi": [32], "vf": [32]}, env.action_space.shape[-1], act_bound=(-10, 10))
  a0c = A0C(env, model, env_bs=args.env_bs, debug=args.debug)
  best_model, hist = a0c.train(args.max_evals)

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
