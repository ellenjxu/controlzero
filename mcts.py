import math
import torch
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

class MCTS:
  def __init__(self, exploration_weight=1e-3, gamma=0.99, k=1, alpha=0.5, d=10, n_sims=100):
    self.gamma = gamma
    self.exploration_weight = exploration_weight
    self.k = k
    self.alpha = alpha
    self.reset()

  def reset(self):
    self.N = defaultdict(int)      # (state,a) -> visits
    self.Q = defaultdict(float)    # (state,a) -> mean value
    self.Ns = defaultdict(int)     # state -> total visits
    self.children = defaultdict(list)

  def _value(self, state):
    """Value estimation at leaf nodes"""
    return 0

  def _puct(self, state, action):
    """PUCT score for action selection"""
    # compare magnitude of Q and sqrt(Ns[state])/(N[(state,action)]+1)
    # print(self.Q[(state,action)], math.sqrt(self.Ns[state])/(self.N[(state,action)]+1))
    return self.Q[(state,action)] + self.exploration_weight * math.sqrt(self.Ns[state])/(self.N[(state,action)]+1)
 
  def puct_select(self, s):
    return max(self.children[s], key=lambda a: self._puct(s, a))

  def simulate(self, s, d=10):
    """Runs a MCTS simulation from state to depth d. Returns q, the value of the state"""
    if d <= 0:
      return self._value(s)

    m = self.k * self.Ns[s] ** self.alpha # progressive widening
    if s not in self.children or len(self.children[s]) < m:
      a = s.sample_action()
      self.children[s].append(a)
    else:
      a = self.puct_select(s)                             # selection
    next_state, r = s.generate(a)                         # expansion
    q = r + self.gamma * self.simulate(next_state, d-1)   # simulation
    self.N[(s,a)] += 1                                   
    self.Q[(s,a)] += (q-self.Q[(s,a)])/self.N[(s,a)]      # backpropagation
    self.Ns[s] += 1
    return q

  def get_action(self, s, d=10, n=100, temp=1, deterministic=False):
    """Gets action by sampling from normalized visit counts. Returns action and pi_tree, prob dist over visit counts"""
    for _ in range(n):
      self.simulate(s, d)
    visit_counts = np.array([self.N[(s,a)] for a in self.children[s]])
    probs = visit_counts ** (1/temp) / (visit_counts ** (1/temp)).sum()
    # print(visit_counts)
    action = max(self.children[s], key=lambda a: self.Q[(s,a)]) if deterministic else np.random.choice(self.children[s], p=probs)
    return action, probs

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