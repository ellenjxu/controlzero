import math
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

class MCTS:
  def __init__(self, exploration_weight=1, gamma=0.95, k=1, alpha=0.5):
    self.gamma = gamma
    self.exploration_weight = exploration_weight
    self.k = k
    self.alpha = alpha
    self.N = defaultdict(int)      # (state,a) -> visits
    self.Q = defaultdict(float)    # (state,a) -> mean value
    self.Ns = defaultdict(int)     # state -> total visits
    self.children = defaultdict(list)

  def simulate(self, s, d=10):
    """runs a MCTS simulation from state to depth d"""
    if d<=0:
      return 0 # TODO: value net

    m = self.k * self.Ns[s] ** self.alpha # progressive widening
    if s not in self.children or len(self.children[s]) < m:
      a = s.sample_action()
      self.children[s].append(a)
    else:
      a = self.select_action(s)                           # selection
    next_state, r = s.generate(a)                         # expansion
    q = r + self.gamma * self.simulate(next_state, d-1)   # simulation
    self.N[(s,a)] += 1                                   
    self.Q[(s,a)] += (q-self.Q[(s,a)])/self.N[(s,a)]      # backpropagation
    self.Ns[s] += 1
    return q
  
  def select_action(self, s):
    def puct(a):
      return self.Q[(s,a)] + self.exploration_weight * math.sqrt(self.Ns[s])/(self.N[(s,a)]+1)
    return max(self.children[s], key=puct)

  def get_best_action(self, s):
    return max(self.children[s], key=lambda a: self.Q[(s,a)])

class State(ABC):
  @abstractmethod
  def __hash__(self):
    """Make state hashable for MCTS dictionaries"""
    pass
  
  @abstractmethod
  def __eq__(self, other):
    """Required for dictionary operations"""
    pass
  
  @classmethod
  @abstractmethod
  def from_array(cls, array):
    """Create state from numpy array"""
    pass
  
  @abstractmethod
  def generate(self, action):
    """Generate next state and reward given action
    Returns: (next_state, reward)
    """
    pass
  
  @abstractmethod
  def sample_action(self):
    """Sample random action to take from this state"""
    pass