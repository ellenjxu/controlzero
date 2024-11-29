import torch
import numpy as np
from networks.mcts import State

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
