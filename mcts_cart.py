import numpy as np
from mcts import MCTS, State
import gymnasium as gym
import gym_cartlataccel

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
  
  def generate(self, action):
    # cartlataccel env.py
    force_mag = 10.0
    tau = 0.02
    max_x = 10.0
    max_episode_steps = 500
    
    new_a = action * force_mag
    new_x = 0.5 * new_a * tau**2 + self.vel * tau + self.pos
    new_x = np.clip(new_x, -max_x, max_x)
    new_v = new_a * tau + self.vel
    
    next_state = CartState(new_x, new_v, self.target)
    error = abs(new_x - self.target)
    reward = -error/max_episode_steps
    
    return next_state, reward
  
  def sample_action(self):
    return np.random.uniform(-10.0, 10.0) # max_u

if __name__ == "__main__": 
  env = gym.make("CartLatAccel-v0", render_mode="human")
  mcts = MCTS(exploration_weight=2.0, gamma=0.99, k=2, alpha=0.3)
  
  max_steps = 500
  search_depth = 10
  n_sims = 100
  
  state, _ = env.reset()
  total_reward = 0
  for step in range(max_steps):
    s = CartState.from_array(state[0])
    for _ in range(n_sims):
      mcts.simulate(s, search_depth)
    action = mcts.get_best_action(s)
    next_state, reward, terminated, truncated, _ = env.step(np.array([[action]])) # env expects batched
    total_reward += reward[0]
    state = next_state
    if truncated:
      break

  print(f"reward: {total_reward}")
  env.close()