"""
MCTS online planner on cartlataccel. Simulates n_sims for each state and chooses the best action
"""

import torch
import argparse
import numpy as np
import gymnasium as gym
import gym_cartlataccel
from networks.mcts import MCTS, State

class CartState(State):
  def __init__(self, pos, vel, target):
    self.pos = pos
    self.vel = vel
    self.target = target
      
  def __hash__(self):
    return hash((self.pos, self.vel, self.target))
  
  def __eq__(self, other):
    if not isinstance(other, CartState):
      return False
    return (self.pos == other.pos and self.vel == other.vel and self.target == other.target)

  @classmethod
  def from_array(cls, array):
    return cls(array[0], array[1], array[2])
  
  def to_tensor(self, device='cpu'):
    return torch.FloatTensor([self.pos, self.vel, self.target]).to(device)
  
  def generate(self, action):
    # cartlataccel env.py
    force_mag = 50.0
    tau = 0.02
    max_x = 10.0
    max_episode_steps = 500
    
    action = np.clip(action, -1, 1)
    new_a = action * force_mag
    new_x = 0.5 * new_a * tau**2 + self.vel * tau + self.pos
    new_x = np.clip(new_x, -max_x, max_x)
    new_v = new_a * tau + self.vel
    
    next_state = CartState(new_x, new_v, self.target)
    error = abs(new_x - self.target)
    reward = -error/max_x
    
    return next_state, reward
  
  def sample_action(self):
    return np.random.uniform(-1, 1) # max_u

def run_mcts(mcts, env, max_steps, search_depth, n_sims, seed=42, deterministic=False):
  state, _ = env.reset(seed=seed)
  total_reward = 0
  for step in range(max_steps):
    s = CartState.from_array(state)
    action, _ = mcts.get_action(s, search_depth, n_sims, deterministic=deterministic)
    next_state, reward, terminated, truncated, _ = env.step(np.array([action])) # env expects batched
    # print(state, action, reward)
    total_reward += reward
    state = next_state
    if truncated:
      break
  return total_reward

if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument("--noise_mode", type=str, default="None")
  parser.add_argument("--search_depth", type=int, default=10)
  parser.add_argument("--n_sims", type=int, default=100)
  parser.add_argument("--render", type=str, default="human")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  env = gym.make("CartLatAccel-v1", render_mode=args.render, noise_mode=args.noise_mode)
  
  import time
  start = time.time()
  mcts = MCTS()
  reward = run_mcts(mcts, env, max_steps=200, search_depth=args.search_depth, n_sims=args.n_sims, seed=args.seed)
  end = time.time()
  print(f"total time: {end-start}")
  print(f"reward: {reward}")