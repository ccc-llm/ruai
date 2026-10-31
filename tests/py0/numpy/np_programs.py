"""numpy in programs: arrays through functions, classes and lists; regression, k-means, softmax."""
from dataclasses import dataclass

import numpy as np


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / x.std(axis=0)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1).reshape(-1, 1))
    return e / e.sum(axis=1).reshape(-1, 1)


@dataclass
class Linear:
    w: np.ndarray
    b: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b


def fit(x: np.ndarray, y: np.ndarray, steps: int, lr: float) -> Linear:
    model = Linear(np.zeros(x.shape[1]), 0.0)
    n = x.shape[0]
    for _ in range(steps):
        err = model.predict(x) - y
        model.w = model.w - lr * (x.T @ err) / n
        model.b = model.b - lr * float(err.mean())
    return model


def kmeans(points: np.ndarray, k: int, iters: int) -> np.ndarray:
    centers = points[:k].copy()
    for _ in range(iters):
        labels = np.zeros(points.shape[0], dtype="int64")
        for i in range(points.shape[0]):
            d = ((centers - points[i]) ** 2).sum(axis=1)
            labels[i] = d.argmin()
        for c in range(k):
            members = points[labels == c]
            if len(members) > 0:
                centers[c] = members.mean(axis=0)
    return centers


rng = np.random.default_rng(0)
x = rng.random((50, 3))
true_w = np.array([2.0, -1.0, 0.5])
y = x @ true_w + 0.3
m = fit(x, y, 2000, 0.5)
print(np.round(m.w, 3), round(m.b, 3))
print(np.round(standardize(x)[:2], 4))
print(np.round(softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])), 6))
pts = np.concatenate([rng.random((20, 2)), rng.random((20, 2)) + 5.0])
print(np.round(kmeans(pts, 2, 5), 4))
arrays: list[np.ndarray] = [np.arange(3), np.arange(3) * 2]
arrays.append(arrays[0] + arrays[1])
total = np.zeros(3, dtype="int64")
for a in arrays:
    total = total + a
print(total, [int(a.sum()) for a in arrays], len(arrays))
alias = arrays[0]
alias[0] = 99
print(arrays[0], alias[0])
for row in np.arange(6).reshape(3, 2):
    print(row, row.sum(), [float(v) * 2 for v in row])
