import torch
import torch.nn as nn

class Normalizer(nn.Module):
  """Normalization layer that tracks running statistics"""
  def __init__(self, shape):
    super().__init__()
    self.register_buffer('mean', torch.zeros(shape))
    self.register_buffer('std', torch.ones(shape))
    self.register_buffer('count', torch.zeros(1))
    self.shape = shape

  def update(self, x):
    """Welford's online algorithm for updating mean and std"""
    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)
    batch_count = x.shape[0]

    delta = batch_mean - self.mean
    self.mean += delta * batch_count / (self.count + batch_count)
    m_a = self.std ** 2 * self.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta ** 2 * self.count * batch_count / (self.count + batch_count)
    self.std = torch.sqrt(M2 / (self.count + batch_count))
    self.count += batch_count

  def norm(self, x):
    return (x - self.mean) / (self.std + 1e-8)
  
  def denorm(self, x):
    return x * self.std + self.mean