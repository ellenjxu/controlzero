import math
import numpy as np
import torch
from collections import defaultdict
import copy

class Node:
  """A wrapper for state and action which makes them hashable for MCTS"""
  def __init__(self, value):
    self.value = value
    # convert value components to a hashable tuple
    if hasattr(value, '__dict__'):
      self._hashable = tuple(value.__dict__.values())
    elif isinstance(value, np.ndarray):
      self._hashable = tuple(value.flatten())
    elif isinstance(value, list):
      self._hashable = tuple(value)
    else:
      self._hashable = (value,)

  def __hash__(self):
    return hash(self._hashable)
  
  def __eq__(self, other):
    if not isinstance(other, Node):
      return False
    return self._hashable == other._hashable

  def to_tensor(self, device='cpu') -> torch.Tensor:
    """Convert value to tensor for neural network input"""
    if isinstance(self.value, torch.Tensor):
      return self.value.to(device)
    return torch.FloatTensor(self._hashable).to(device)

class MCTS:
  def __init__(self, exploration_weight=1e-2, gamma=0.99, k=1, alpha=0.5): # TODO: exploration weight. c_puct = 0.05
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

  def _value(self, state: Node):
    """Value estimation at leaf nodes"""
    return 0

  def _puct(self, state: Node, action: float):
    """PUCT score for action selection"""
    # compare magnitude of Q and sqrt(Ns[state])/(N[(state,action)]+1)
    # print(self.Q[(state,action)], math.sqrt(self.Ns[state])/(self.N[(state,action)]+1))
    return self.Q[(state,action)] + self.exploration_weight * math.sqrt(self.Ns[state])/(self.N[(state,action)]+1)
 
  def puct_select(self, s: Node):
    return max(self.children[s], key=lambda a: self._puct(s, a)) # takes each children action and finds the max PUCT score

  def sample_action(self, env, s: Node = None) -> float:
    """Sample an action using the environment's action space"""
    return env.action_space.sample()

  def search(self, env, s: Node, d: int, is_root: bool = True, k_max: int = 10):
    """Runs a MCTS simulation from state to depth d. Stores statstics and returns q, the value of the state"""
    if d <= 0:
      return self._value(s)

    if is_root:
      if s not in self.children or len(self.children[s]) < k_max: # TODO: fixed number of children for root node
        # now sampling action from env
        a = self.sample_action(env, s)
        a = Node(a)
        if a not in self.children[s]:
          self.children[s].append(a)
      else:
        a = self.puct_select(s)
    else:      
      m = self.k * self.Ns[s] ** self.alpha # progressive widening factor: when to explore new actions
      if s not in self.children or len(self.children[s]) < m:
        # now sampling action from env
        a = self.sample_action(env, s)
        a = Node(a)
        if a not in self.children[s]:
          self.children[s].append(a)
      else:
        a = self.puct_select(s)                                   # selection

    action = np.array([a.value])
    next_state, r, terminated, truncated, info = env.step(action)  # expansion
    # make sure both next_state and action are hashable
    next_state = Node(next_state)

    q = r + self.gamma * self.search(env, next_state, d-1, False)      # simulation
    
    # store statisticics
    self.N[(s,a)] += 1                                            # backpropagation
    self.Q[(s,a)] += (q-self.Q[(s,a)])/self.N[(s,a)]
    self.Ns[s] += 1
    return q

  def get_policy(self, s: Node):
    """MCTS policy as mapping actions to counts, max Q values"""
    actions = self.children[s]
    visit_counts = np.array([self.N[(s,a)] for a in actions])
    # print("visit counts", visit_counts)
    norm_counts = visit_counts / visit_counts.sum()
    q_values = [self.Q[(s,a)] for a in actions]
    max_q = max(q_values) if q_values else 0
    return actions, norm_counts, max_q

  def get_action(self, env, s: Node, d: int, n: int, deterministic: bool=False):
    """Choose the best action from current state."""
    # pass in env, for each rollout reset env to initial state
    # replaces s.generate()
    # s is just a hashable state
    for _ in range(n):
      mcts_env = copy.deepcopy(env)
      self.search(mcts_env, s, d)
    actions, norm_counts, max_q = self.get_policy(s)
    
    if deterministic:
      best_idx = np.argmax(norm_counts)
      return actions[best_idx].value
    else: # sample from distribution
      sampled_idx = np.random.choice(len(actions), p=norm_counts)
      return actions[sampled_idx].value

# A0C paper: https://arxiv.org/abs/1805.09613
class A0CModel(MCTS):
  """AlphaZero Continuous. Uses NN to guide MCTS search."""
  def __init__(self, model, exploration_weight=1e-2, gamma=0.99, k=1, alpha=0.5, device='cpu'):
    super().__init__(exploration_weight, gamma, k, alpha)
    self.model = model # f_theta(s) -> pi(s), V(s)
    self.device = device

  def _value(self, state): # override methods to use NN
    with torch.no_grad():
      _, value = self.model(state.to_tensor().to(self.device))
    return value

  def sample_action(self, env, s: Node = None) -> float:
    """Sample an action using the policy network"""
    a = self.model.actor.get_action(s.to_tensor().to(self.device)) # a is a float
    return a.item()
