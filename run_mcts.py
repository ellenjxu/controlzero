""" MCTS online planner on cartlataccel. Simulates n_sims for each state and chooses the best action """

import argparse
import numpy as np
import gymnasium as gym
import gym_cartlataccel
from networks.mcts import MCTS, State
from utils.cartstate import CartState

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
