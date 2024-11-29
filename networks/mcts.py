import math
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

class State(ABC):
  @abstractmethod
  def __hash__(self): # required for MCTS dictionary
    pass
  @abstractmethod
  def __eq__(self, other):
    pass
  @classmethod
  @abstractmethod
  def from_array(cls, array):
    pass
  @abstractmethod
  def generate(self, action):
    """Generate next state and reward given action. similar to gym.step()"""
    pass
  @abstractmethod
  def sample_action(self):
    """Sample random action to take from this state"""
    pass

class MCTS:
  def __init__(self, exploration_weight=1e-3, gamma=0.99, k=1, alpha=0.5): # TODO: exploration weight
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

  def _value(self, state: State):
    """Value estimation at leaf nodes"""
    return 0

  def _puct(self, state: State, action: float):
    """PUCT score for action selection"""
    # compare magnitude of Q and sqrt(Ns[state])/(N[(state,action)]+1)
    # print(self.Q[(state,action)], math.sqrt(self.Ns[state])/(self.N[(state,action)]+1))
    return self.Q[(state,action)] + self.exploration_weight * math.sqrt(self.Ns[state])/(self.N[(state,action)]+1)
 
  def puct_select(self, s: State):
    return max(self.children[s], key=lambda a: self._puct(s, a))

  def search(self, s: State, d: int, is_root: bool = True, k_max: int = 10):
    """Runs a MCTS simulation from state to depth d. Returns q, the value of the state"""
    if d <= 0:
      return self._value(s)

    if is_root:
      if s not in self.children or len(self.children[s]) < k_max: # TODO: fixed number of children for root node
        a = s.sample_action()
        if a not in self.children[s]:
          self.children[s].append(a)
      else:
        a = self.puct_select(s)
    else:      # otherwise use progressive widening
      m = self.k * self.Ns[s] ** self.alpha
      if s not in self.children or len(self.children[s]) < m:
        a = s.sample_action()
        if a not in self.children[s]:
          self.children[s].append(a)
      else:
        a = self.puct_select(s)                                   # selection
    next_state, r = s.generate(a)                                 # expansion
    q = r + self.gamma * self.search(next_state, d-1, False)      # simulation
    self.N[(s,a)] += 1                                            # backpropagation
    self.Q[(s,a)] += (q-self.Q[(s,a)])/self.N[(s,a)]
    self.Ns[s] += 1
    return q

  def get_policy(self, s: State):
    actions = self.children[s]
    visit_counts = np.array([self.N[(s,a)] for a in actions])
    # print(visit_counts)
    norm_counts = visit_counts / visit_counts.sum()
    return actions, norm_counts

  def get_action(self, s: State, d: int, n: int, deterministic: bool=False):
    for _ in range(n):
      self.search(s, d)
    actions, norm_counts = self.get_policy(s)
    
    if deterministic:
      best_idx = np.argmax(norm_counts)
      return actions[best_idx], norm_counts[best_idx]
    else: # sample from distribution
      sampled_idx = np.random.choice(len(actions), p=norm_counts)
      return actions[sampled_idx], norm_counts[sampled_idx]
