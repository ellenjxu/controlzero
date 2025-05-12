import numpy
import torch
import matplotlib.pyplot as plt

def sample_rollout(model, env, n_episodes=10, n_steps=200, seed=42, device='cpu'):
  eps_rewards = []
  for i in range(n_episodes):
    rewards = rollout(model, env, max_steps=n_steps, seed=seed+i, deterministic=True, device=device)
    eps_rewards.append(sum(rewards))
  return eps_rewards

def rollout(model, env, max_steps=1000, seed=42, deterministic=False, device='cpu'):
  state, _ = env.reset(seed=seed)
  rewards = []
  for _ in range(max_steps):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = model.get_action(state_tensor, deterministic=deterministic)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state
    rewards.append(reward)
    if done:
      state, _ = env.reset()
      break
  return rewards

def plot_losses(hist, save_path=None, title=None):
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
  
  if title:
    plt.suptitle(title)
  plt.tight_layout()
  if save_path:
    plt.savefig(save_path)
  plt.show()
  plt.close(fig)

# https://stackoverflow.com/a/17637351
class RunningStats:
  def __init__(self):
    self.n = 0
    self.old_m = 0
    self.new_m = 0
    self.old_s = 0
    self.new_s = 0

  def clear(self):
    self.n = 0

  def push(self, x):
    self.n += 1

    if self.n == 1:
      self.old_m = self.new_m = x
      self.old_s = 0
    else:
      self.new_m = self.old_m + (x - self.old_m) / self.n
      self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

      self.old_m = self.new_m
      self.old_s = self.new_s

  def mean(self):
    return self.new_m if self.n else 0.0

  def variance(self):
    return self.new_s / (self.n - 1) if self.n > 1 else 0.0

  def standard_deviation(self):
    return np.sqrt(self.variance())
    
