import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import gym_cartlataccel
import matplotlib.pyplot as plt
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from networks.alphazero import A0CModel
from networks.agent import ActorCritic
from networks.mcts import MCTS
from run_mcts import CartState

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class A0C:
  def __init__(self, env, model, lr=1e-1, epochs=10, ent_coeff=0.01, env_bs=1, device='cpu', debug=False):
    self.env = env
    self.env_bs = env_bs
    self.model = model.to(device)
    self.epochs = epochs
    self.ent_coeff = ent_coeff
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=100000, device=device))
    self.device = device
    self.debug = debug
    self.mcts = A0CModel(model, device=device)
    # self.mcts = MCTS()
    self.hist = {'iter': [], 'reward': [], 'value_loss': [], 'policy_loss': [], 'total_loss': []}

  def _compute_return(self, states):
    states = np.array(states)
    returns = []
    for state in states:
      s = CartState.from_array(state)
      q_values = [self.mcts.Q[(s,a)] for a in self.mcts.children[s]] # a0c uses max_a Q vs environment rewards
      v_target = max(q_values) if q_values else 0
      returns.append(v_target)
    return np.array(returns)

  def value_loss(self, states, returns):
    V = self.model.critic(states, out_norm=True).squeeze() # normalize output for loss, unnorm in mcts
    # normalize returns for loss
    returns = self.model.critic.return_normalizer.norm(returns)
    value_loss = nn.L1Loss()(V, returns)
    if self.debug:
      print(f"value {V[0]} return {returns[0]}")
      print(f"normalized value range: {V.min().item():.3f} {V.max().item():.3f}")
      print(f"normalized returns range: {returns.min().item():.3f} {returns.max().item():.3f}")
      normalized_states = self.model.critic.obs_normalizer.norm(states)
      print(f"state {states[0]}, normalized {normalized_states[0]}")
      print(f"normalized obs range: {normalized_states.min().item():.3f} {normalized_states.max().item():.3f}")
    return value_loss

  def policy_loss(self, mcts_states, mcts_actions, mcts_counts):
    # actor loss is KL divergence between MCTS policy and model policy
    logcounts = torch.log(torch.tensor(mcts_counts, device=self.device))
    logprobs, entropy = self.model.actor.get_logprob(mcts_states, mcts_actions.unsqueeze(-1))
    with torch.no_grad():
      error = logprobs - logcounts # self.tau * logcounts
    policy_loss = (error * logprobs).mean()
    policy_loss -= self.ent_coeff * entropy.mean()
    if self.debug:
      print(f"mean absolute error: {torch.mean(torch.abs(error))}")
      print(f"std: {self.model.actor.log_std}")
    return policy_loss
  
  def l2_loss(self):
    l2_loss = sum(torch.norm(param) for param in self.model.parameters())
    return l2_loss
  
  def mcts_rollout(self, max_steps=1000, deterministic=False):
    states, rewards, mcts_states, mcts_actions, mcts_counts = [], [], [], [], []
    state, _ = self.env.reset()

    for _ in range(max_steps):
      s = CartState.from_array(state) # TODO: get rid of cartstate
      best_action, _ = self.mcts.get_action(s, d=10, n=100, deterministic=True) # get single action and prob
      next_state, reward, terminated, truncated, info = self.env.step(np.array([[best_action]]))
      states.append(state)
      rewards.append(reward)
      done = terminated or truncated

      # mcts returns actions, counts
      actions, norm_counts = self.mcts.get_policy(s)
      mcts_counts.append(norm_counts)
      mcts_actions.append(actions)
      mcts_states.append([state]*len(actions))

      state = next_state
      if done:
        state, _ = self.env.reset()
        self.mcts.reset() # reset mcts tree
    return states, rewards, mcts_states, mcts_actions, mcts_counts

  def train(self, max_iters=1000, n_episodes=10, n_steps=30):
    start = time.time()
    for i in range(max_iters):
      # collect data
      for _ in range(n_episodes):
        states, rewards, mcts_states, mcts_actions, mcts_counts = self.mcts_rollout(n_steps)
        returns = self._compute_return(states)
        
        # update normalizers with new data
        self.model.update_normalizers(
          torch.FloatTensor(states).to(self.device),
          torch.FloatTensor(returns).to(self.device)
        )
        
        episode_dict = TensorDict({
          "states": states,
          "returns": returns,
          "mcts_states": np.array(mcts_states),
          "mcts_actions": np.array(mcts_actions),
          "mcts_counts": np.array(mcts_counts),
        }, batch_size=len(states))
        self.replay_buffer.extend(episode_dict)

      # update model
      for _ in range(self.epochs):
        batch = self.replay_buffer.sample(batch_size=n_steps*n_episodes) # TODO: taking entire batch of 300
        value_loss = self.value_loss(batch['states'], batch['returns'])
        policy_loss = self.policy_loss(batch['mcts_states'], batch['mcts_actions'], batch['mcts_counts'])
        l2_loss = self.l2_loss()
        loss = policy_loss + 0.5 * value_loss + 1e-4 * l2_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      
      # evaluate current actor net
      eps_rewards = []
      for _ in range(5):
        rewards = rollout(self.model.actor, self.env, max_steps=n_steps, deterministic=True)
        eps_rewards.append(sum(rewards))
      avg_reward = np.mean(eps_rewards)

      # log metrics
      self.hist['iter'].append(i)
      self.hist['reward'].append(avg_reward)
      self.hist['value_loss'].append(value_loss.item())
      self.hist['policy_loss'].append(policy_loss.item())
      self.hist['total_loss'].append(loss.item())

      print(f"actor loss {policy_loss.item():.3f} value loss {value_loss.item():.3f} l2 loss {l2_loss.item():.3f}")
      print(f"iter {i}, reward {avg_reward:.3f}, t {time.time()-start:.2f}")

    print(f"Total time: {time.time() - start}")
    return self.model, self.hist

def rollout(model, env, max_steps=1000, deterministic=False):
  state, _ = env.reset()
  rewards = []
  for _ in range(max_steps):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = model.get_action(state_tensor, deterministic=deterministic)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state
    rewards.append(reward)
    if done:
      env.close()
      break
  env.close()
  return rewards

def plot_losses(hist, save_path=None):
  plt.figure(figsize=(10, 6))
  # Create subplots for better visualization
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
  # Plot losses
  ax1.plot(hist['iter'], hist['total_loss'], label='Total Loss')
  ax1.plot(hist['iter'], hist['value_loss'], label='Value Loss')
  ax1.plot(hist['iter'], hist['policy_loss'], label='Policy Loss')
  ax1.set_xlabel('n_iters')
  ax1.set_ylabel('loss')
  ax1.legend()
  # Plot rewards
  ax2.plot(hist['iter'], hist['reward'], label='Average Reward', color='green')
  ax2.set_xlabel('n_iters')
  ax2.set_ylabel('reward')
  ax2.legend()
  
  plt.tight_layout()
  if save_path:
    plt.savefig(save_path)
  plt.show()
  plt.close(fig)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_iters", type=int, default=10)
  parser.add_argument("--n_eps", type=int, default=10)
  parser.add_argument("--n_steps", type=int, default=30)
  parser.add_argument("--env_bs", type=int, default=1) # TODO: batch
  parser.add_argument("--save", default=True)
  parser.add_argument("--noise_mode", default=None)
  parser.add_argument("--debug", default=True)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  print(f"training a0c with max_iters {args.max_iters}") 
  env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=args.env_bs)
  model = ActorCritic(env.observation_space.shape[-1], {"pi": [32], "vf": [32]}, env.action_space.shape[-1]) #, act_bound=(-1, 1))
  a0c = A0C(env, model, env_bs=args.env_bs, debug=args.debug)
  best_model, hist = a0c.train(args.max_iters, args.n_eps, args.n_steps)

  # run value net online planner
  from run_mcts import run_mcts
  env = gym.make("CartLatAccel-v1", render_mode="human", noise_mode=args.noise_mode)
  reward = run_mcts(a0c.mcts, env, max_steps=200, search_depth=10, n_sims=100, seed=args.seed)
  print(f"mcts reward {reward}")

  # run actor net model
  print("rollout out best actor")
  env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=1, render_mode="human")
  rewards = rollout(best_model.actor, env, max_steps=200, deterministic=True)
  print(f"actor reward {sum(rewards)}")

  if args.save:
    os.makedirs('out', exist_ok=True)
    torch.save(best_model, 'out/best.pt')
    # save history
    import pickle as pkl
    with open('out/history.pkl', 'wb') as f:
      pkl.dump(hist, f)
    plot_losses(hist, save_path="out/loss_curve.png")
  else:
    plot_losses(hist)