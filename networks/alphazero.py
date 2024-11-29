# A0C paper: https://arxiv.org/abs/1805.09613
import torch
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

  # def _puct(self, state: State, action: np.ndarray):
  #   state_tensor = state.to_tensor().to(self.device)
  #   action_tensor = torch.tensor([action], device=self.device)
  #   with torch.no_grad():
  #     logprob, _ = self.model.actor.get_logprob(state_tensor, action_tensor)
  #   return self.Q[(state,action)] + self.exploration_weight * torch.exp(logprob) * np.sqrt(self.Ns[state])/(self.N[(state,action)]+1)

  def _batched_logprobs(self, s: State, actions: list[np.ndarray]):
    state_tensor = s.to_tensor().to(self.device)
    action_tensor = torch.tensor(actions, device=self.device)
    with torch.no_grad():
      logprobs = self.model.actor.get_logprob(state_tensor, action_tensor)
    return logprobs

  def puct_select(self, s: State): # batched
    actions = self.children[s]
    if len(actions) == 1:
      return actions[0]
    logprobs = self._batched_logprobs(s, actions)
    puct_scores = [
      self.Q[(s,a)] + self.exploration_weight * np.exp(logprob) * np.sqrt(self.Ns[s]) / (self.N[(s,a)]+1)
      for a, logprob in zip(actions, logprobs)
    ]
    return actions[np.argmax(puct_scores)]