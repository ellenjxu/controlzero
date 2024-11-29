# A0C paper: https://arxiv.org/abs/1805.09613
import torch
import math
import numpy as np
from networks.mcts import MCTS, State

class A0C(MCTS):
  """AlphaZero Continuous. Uses NN to guide MCTS search."""
  def __init__(self, model, exploration_weight=1e-1, gamma=0.99, k=1, alpha=0.5, device='cpu'):
    super().__init__(exploration_weight, gamma, k, alpha)
    self.model = model # f_theta(s) -> pi(s), V(s)
    self.device = device

  def _value(self, state): # override methods to use NN
    with torch.no_grad():
      _, value = self.model(state.to_tensor().to(self.device))
    return value

  def _batched_logprobs(self, s: State, actions: list[np.ndarray]):
    state_tensor = s.to_tensor().to(self.device)
    action_tensor = torch.tensor(actions, device=self.device)
    with torch.no_grad():
      logprobs, _ = self.model.actor.get_logprob(state_tensor, action_tensor)
    return logprobs

  def puct_select(self, state: State): # batched
    actions = self.children[state]
    logprobs = self._batched_logprobs(state, actions)
    q_values = torch.FloatTensor([self.Q[(state, a)] for a in actions]).to(self.device)
    visits = torch.FloatTensor([self.N[(state, a)] for a in actions]).to(self.device)
    
    # puct score
    sqrt_ns = math.sqrt(float(self.Ns[state]))
    exploration = self.exploration_weight * torch.exp(logprobs) * (sqrt_ns / (visits + 1))
    score = q_values + exploration
    
    return actions[torch.argmax(score).item()]