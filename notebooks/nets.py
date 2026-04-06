import numpy as np

class MLP:

  def __init__(self, input_size, hidden_size, output_size):
    self.W1 = np.random.randn(hidden_size, input_size) * 0.1
    self.b1 = np.zeros((hidden_size, 1))
    self.W2 = np.random.randn(output_size, hidden_size) * 0.1
    self.b2 = np.zeros((output_size, 1))

  def forward(self, x):
    self.x = x
    self.z1 = self.W1 @ x + self.b1
    self.h = np.tanh(self.z1)
    self.y = self.W2 @ self.h + self.b2
    return self.y

  def backward(self, y_true):
    N = y_true.shape[1]

    dL_dy = (self.y - y_true) / N
    self.dL_dW2 = dL_dy @ self.h.T
    self.dL_db2 = np.sum(dL_dy, axis=1, keepdims=True)

    dL_dh = self.W2.T @ dL_dy
    dL_dz1 = (1 - self.h**2) * dL_dh

    self.dL_dW1 = dL_dz1 @ self.x.T
    self.dL_db1 = np.sum(dL_dz1, axis=1, keepdims=True)

  def update_weights(self, lr):
    self.W2 -= lr * self.dL_dW2
    self.b2 -= lr * self.dL_db2
    self.W1 -= lr * self.dL_dW1
    self.b1 -= lr * self.dL_db1

  def train(self, x, y, lr, epochs, batch_size):
    N = x.shape[1]

    for epoch in range(epochs):
      idx = np.random.permutation(N)

      for i in range(0, N, batch_size):
        batch = idx[i:i+batch_size]
        xb = x[:, batch]
        yb = y[:, batch]

        self.forward(xb)
        self.backward(yb)
        self.update_weights(lr)


class ValueNet:

  def __init__(self, input_size, hidden_size, output_size):
    self.W1 = np.random.randn(hidden_size, input_size) 
    self.b1 = np.zeros((hidden_size, 1))
    self.W2 = np.random.randn(output_size, hidden_size)
    self.b2 = np.zeros((output_size, 1))

  def forward(self, x):
    self.x = x.reshape(-1,1)
    self.z1 = self.W1 @ self.x + self.b1
    self.h = np.tanh(self.z1)
    self.y = self.W2 @ self.h + self.b2
    return self.y

  def backward(self, y_target):
    y_target = np.array([[y_target]])
    dL_dy = 2 * (self.y - y_target)
    self.dL_dW2 = dL_dy @ self.h.T
    self.dL_db2 = dL_dy

    dL_dh = self.W2.T @ dL_dy
    dL_dz1 = (1 - self.h**2) * dL_dh

    self.dL_dW1 = dL_dz1 @ self.x.T
    self.dL_db1 = dL_dz1

  def update_weights(self, lr):
    self.W2 -= lr * self.dL_dW2
    self.b2 -= lr * self.dL_db2
    self.W1 -= lr * self.dL_dW1
    self.b1 -= lr * self.dL_db1


class QNet:
    def __init__(self, hidden=32, n_a = 1, n_s = 1):
        self.W1 = np.random.randn(hidden, n_s) * np.sqrt(2)
        self.b1 = np.zeros((hidden, 1))
        self.W2 = np.random.randn(n_a, hidden) * np.sqrt(1/16)
        self.b2 = np.zeros((n_a, 1))

    def forward(self, x):
        self.x = np.array(x)
        self.z1 = self.W1 @ self.x + self.b1
        self.h = np.maximum(0, self.z1)
        self.q = self.W2 @ self.h + self.b2
        return self.q


    def backward(self, q_target, a_idx):

      
      batch_size = None
      if a_idx.ndim > 0:
        batch_size = a_idx.shape[0]

      q_target = np.array(q_target)
      dL_dy = np.zeros_like(self.q)

      if batch_size:
        dL_dy[a_idx, np.arange(batch_size)] = 2 * (self.q[a_idx, np.arange(batch_size)] - q_target[a_idx, np.arange(batch_size)])
        dL_dy /= batch_size
      else:
        dL_dy[a_idx] = 2 * (self.q[a_idx] - q_target[a_idx])

      self.dL_dW2 = dL_dy @ self.h.T
      self.dL_db2 = np.sum(dL_dy, axis=1, keepdims=True)
      dL_dh = self.W2.T @ dL_dy
      dL_dz1 = dL_dh.copy()
      dL_dz1[self.z1 <= 0] = 0

      self.dL_dW1 = dL_dz1 @ self.x.T
      self.dL_db1 = np.sum(dL_dz1, axis=1, keepdims=True)

    def update_weights(self, lr):
      self.W2 -= lr * self.dL_dW2
      self.b2 -= lr * self.dL_db2
      self.W1 -= lr * self.dL_dW1
      self.b1 -= lr * self.dL_db1





class ReplayBuffer():
  def __init__(self, size, action_dim, state_dim):
    self. max_size = size
    self.ptr = 0
    self.size = 0

    self.s = np.zeros((size, state_dim))
    self.a = np.zeros((size, action_dim))
    self.s_next = np.zeros((size, state_dim))
    self.r = np.zeros((size, 1))
    self.done = np.zeros((size, 1))

  def add(self, s, a, r, s_next, done):
    self.s[self.ptr] = s
    self.a[self.ptr] = a
    self.r[self.ptr] = r
    self.s_next[self.ptr] = s_next
    self.done[self.ptr] = done

    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def sample(self, batch_size):
    idx = np.random.randint(0, self.size, size=batch_size)
    return (
        self.s[idx],
        self.a[idx],
        self.r[idx],
        self.s_next[idx],
        self.done[idx]
    )

  