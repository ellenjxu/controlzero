import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import gym_cartlataccel
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from model import ActorCritic

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class PPO:
  def __init__(self, env, model, lr=3e-4, gamma=0.99, lam=0.99, clip_range=0.2, epochs=10, n_steps=2048, ent_coeff=0.001, bs=64, env_bs=1, device='cpu', debug=False):
    self.env = env
    self.env_bs = env_bs
    self.model = model.to(device)
    self.gamma = gamma
    self.lam = lam
    self.clip_range = clip_range
    self.epochs = epochs
    self.n_steps = n_steps
    self.ent_coeff = ent_coeff
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000, device=device), batch_size=bs)
    self.bs = bs
    self.hist = []
    self.start = time.time()
    self.device = device
    self.debug = debug

  def compute_gae(self, rewards, values, done, next_value):
    returns, advantages = np.zeros_like(rewards), np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
      delta = rewards[t] + self.gamma*next_value*(1-done[t]) - values[t]
      gae = delta + self.gamma*self.lam*(1-done[t])*gae
      advantages[t] = gae
      returns[t] = gae + values[t]
      next_value = values[t]
    return returns, advantages

  def evaluate_cost(self, model, states, actions, returns, advantages, logprob):
    new_logprob, entropy = model.actor.get_logprob(states, actions)
    ratio = torch.exp(new_logprob-logprob).squeeze()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = nn.MSELoss()(model.critic(states).squeeze(), returns.squeeze())
    entropy_loss = -self.ent_coeff * entropy.mean()
    return {"actor": actor_loss, "critic": critic_loss, "entropy": entropy_loss}

  @staticmethod
  def rollout(env, model, max_steps=1000, deterministic=False, device='cuda'):
    states, actions, rewards, dones  = [], [], [], []
    state, _ = env.reset()

    for _ in range(max_steps):
      state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
      action = model.get_action(state_tensor, deterministic=deterministic)
      next_state, reward, terminated, truncated, info = env.step(action)
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      done = terminated or truncated
      dones.append(done)

      state = next_state
      if done:
        state, _ = env.reset()
    return states, actions, rewards, dones, next_state

  def train(self, max_evals=1000):
    eps = 0
    while True:
      # rollout
      start = time.perf_counter()
      states, actions, rewards, dones, next_state = self.rollout(self.env, self.model.actor, self.n_steps, device=self.device)
      rollout_time = time.perf_counter()-start

      # compute gae
      start = time.perf_counter()
      with torch.no_grad():
        state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        values = self.model.critic(state_tensor).cpu().numpy().squeeze()
        next_values = self.model.critic(next_state_tensor).cpu().numpy().squeeze()
        logprobs_tensor, _ = self.model.actor.get_logprob(state_tensor, action_tensor)
        returns, advantages = self.compute_gae(np.array(rewards), values, np.array(dones), next_values)
        gae_time = time.perf_counter()-start
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
      # add to buffer
      start = time.perf_counter()
      episode_dict = TensorDict(
        {
          "states": state_tensor,
          "actions": action_tensor,
          "returns": torch.FloatTensor(returns).to(self.device),
          "advantages": torch.FloatTensor(advantages).to(self.device),
          "logprobs": logprobs_tensor,
        },
        batch_size=self.n_steps
      )
      self.replay_buffer.extend(episode_dict)
      buffer_time = time.perf_counter() - start

      # update
      start = time.perf_counter()
      for _ in range(self.epochs):
        for i, batch in enumerate(self.replay_buffer):
          costs = self.evaluate_cost(self.model, batch['states'], batch['actions'], batch['returns'], batch['advantages'], batch['logprobs'])
          loss = costs["actor"] + 0.5 * costs["critic"] + costs["entropy"]
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          break
      self.replay_buffer.empty() # clear buffer
      update_time = time.perf_counter() - start

      # debug info
      if self.debug:
        print(f"critic loss {costs['critic'].item():.3f} entropy {costs['entropy'].item():.3f} mean action {np.mean(abs(np.array(actions)))}")
        print(f"Runtimes: rollout {rollout_time:.3f}, gae {gae_time:.3f}, buffer {buffer_time:.3f}, update {update_time:.3f}")

      eps += self.env_bs
      avg_reward = np.sum(rewards)/self.env_bs

      print(f"eps {eps:.2f}, reward {avg_reward:.3f}, t {time.time()-self.start:.2f}")
      print(f"Runtimes: rollout {rollout_time:.3f}, gae {gae_time:.3f}, buffer {buffer_time:.3f}, update {update_time:.3f}")
      self.hist.append((eps, avg_reward))
      if eps > max_evals:
        print(f"Total time: {time.time() - self.start}")
        break

    return self.model.actor, self.hist

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_evals", type=int, default=30000)
  parser.add_argument("--env_bs", type=int, default=1000)
  parser.add_argument("--save_model", default=False)
  parser.add_argument("--noise_mode", default=None)
  args = parser.parse_args()

  print(f"training ppo with max_evals {args.max_evals}") 
  env = gym.make("CartLatAccel-v0", noise_mode=args.noise_mode, env_bs=args.env_bs)
  model = ActorCritic(env.observation_space.shape[-1], {"pi": [32], "vf": [32]}, env.action_space.shape[-1], shared_layers=True, act_bound=(-1, 1))
  ppo = PPO(env, model, env_bs=args.env_bs)
  best_model, hist = ppo.train(args.max_evals)

  print(f"rolling out best model") 
  env = gym.make("CartLatAccel-v0", noise_mode=args.noise_mode, env_bs=1, render_mode="human")
  states, actions, rewards, dones, next_state= ppo.rollout(env, best_model, max_steps=200, deterministic=True)
  print(f"reward {sum(rewards)}")

  if args.save_model:
    os.makedirs('out', exist_ok=True)
    torch.save(best_model, 'out/best.pt')
