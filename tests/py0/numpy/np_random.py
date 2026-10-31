"""numpy: np.random.default_rng (PCG64): random, uniform, integers, shuffle, permutation."""
import numpy as np

rng = np.random.default_rng(42)
print(repr(rng.random(3)), rng.random(), repr(rng.random((2, 2))))
print(repr(rng.uniform(-1.0, 1.0, 4)), rng.uniform(5.0, 6.0))
print(repr(rng.integers(0, 10, 8)), rng.integers(100), rng.integers(5, 7), repr(rng.integers(0, 3, (2, 3))))
x = np.arange(10)
rng.shuffle(x)
print(x, repr(rng.permutation(6)), repr(rng.permutation(np.array([1.5, 2.5, 3.5]))))
m = np.arange(12).reshape(4, 3)
rng.shuffle(m)
print(m)
r2 = np.random.default_rng(7)
w = r2.random(1000)
print(repr(w.mean()), repr(w.max()), repr((w > 0.5).sum()))
