"""
cnn_numpy.py
------------
CNN kecil yang ditulis dari nol hanya dengan NumPy (tanpa PyTorch/TensorFlow),
supaya setiap langkah forward & backward bisa dibaca dan dipahami.

Arsitektur:
  Input (C, 16, 16)
  -> Conv 3x3 (8 filter)  -> ReLU -> MaxPool 2x2   => (8, 8, 8)
  -> Conv 3x3 (16 filter) -> ReLU -> MaxPool 2x2   => (16, 4, 4)
  -> Flatten (256) -> Dense (5 kelas) -> Softmax
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------- lapisan
class Conv3x3:
    def __init__(self, c_in, c_out, rng):
        self.W = rng.normal(0, np.sqrt(2 / (c_in * 9)), (c_out, c_in * 9))
        self.b = np.zeros(c_out)

    def forward(self, X):
        N, C, H, W = X.shape
        Xp = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)))
        jendela = sliding_window_view(Xp, (3, 3), axis=(2, 3))   # N,C,H,W,3,3
        self.cols = jendela.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * W, C * 9)
        self.shape = X.shape
        out = self.cols @ self.W.T + self.b
        return out.reshape(N, H, W, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        N, C, H, W = self.shape
        d = dout.transpose(0, 2, 3, 1).reshape(N * H * W, -1)
        self.dW = d.T @ self.cols
        self.db = d.sum(0)
        dcols = (d @ self.W).reshape(N, H, W, C, 3, 3)
        dXp = np.zeros((N, C, H + 2, W + 2))
        for i in range(3):
            for j in range(3):
                dXp[:, :, i:i + H, j:j + W] += dcols[..., i, j].transpose(0, 3, 1, 2)
        return dXp[:, :, 1:-1, 1:-1]

    def params(self):
        return [("W", self.W, "dW"), ("b", self.b, "db")]


class ReLU:
    def forward(self, X):
        self.m = X > 0
        return X * self.m

    def backward(self, d):
        return d * self.m

    def params(self):
        return []


class MaxPool2:
    def forward(self, X):
        N, C, H, W = X.shape
        r = X.reshape(N, C, H // 2, 2, W // 2, 2)
        out = r.max(axis=(3, 5))
        self.m = r == out[:, :, :, None, :, None]
        self.shape = X.shape
        return out

    def backward(self, d):
        return (self.m * d[:, :, :, None, :, None]).reshape(self.shape)

    def params(self):
        return []


class Dense:
    def __init__(self, n_in, n_out, rng):
        self.W = rng.normal(0, np.sqrt(1 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)

    def forward(self, X):
        self.shape = X.shape
        self.X = X.reshape(len(X), -1)
        return self.X @ self.W + self.b

    def backward(self, d):
        self.dW = self.X.T @ d
        self.db = d.sum(0)
        return (d @ self.W.T).reshape(self.shape)

    def params(self):
        return [("W", self.W, "dW"), ("b", self.b, "db")]


# ------------------------------------------------------------------ model
class CNNDriver:
    def __init__(self, n_kanal, n_kelas=5, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = [
            Conv3x3(n_kanal, 8, rng), ReLU(), MaxPool2(),
            Conv3x3(8, 16, rng), ReLU(), MaxPool2(),
            Dense(16 * 4 * 4, n_kelas, rng),
        ]
        self._adam = {}
        self.t = 0

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def predict_proba(self, X, batch=256):
        hasil = []
        for i in range(0, len(X), batch):
            z = self.forward(X[i:i + batch])
            z = np.exp(z - z.max(1, keepdims=True))
            hasil.append(z / z.sum(1, keepdims=True))
        return np.concatenate(hasil)

    def train_step(self, X, y, lr=0.003):
        z = self.forward(X)
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        loss = -np.log(p[np.arange(len(y)), y] + 1e-9).mean()

        d = p.copy()
        d[np.arange(len(y)), y] -= 1
        d /= len(y)
        for layer in reversed(self.layers):
            d = layer.backward(d)

        # Optimizer Adam
        self.t += 1
        b1, b2 = 0.9, 0.999
        for li, layer in enumerate(self.layers):
            for nama, w, gnama in layer.params():
                g = getattr(layer, gnama)
                m, v = self._adam.get((li, nama), (0, 0))
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g ** 2
                self._adam[(li, nama)] = (m, v)
                mh, vh = m / (1 - b1 ** self.t), v / (1 - b2 ** self.t)
                w -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss

    # simpan / muat bobot
    def simpan(self, path):
        bobot = {f"{i}_{n}": w for i, l in enumerate(self.layers) for n, w, _ in l.params()}
        np.savez(path, **bobot)

    def muat(self, path):
        data = np.load(path)
        for i, l in enumerate(self.layers):
            for n, w, _ in l.params():
                w[...] = data[f"{i}_{n}"]
