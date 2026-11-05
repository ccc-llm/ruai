# NumPy in the browser: arrays, broadcasting, reductions and random numbers.
# The arrays live in rupy_lib.wasm (rupy's numpy, in Rust).
import numpy as np

a = np.arange(12).reshape((3, 4))
print(a)
print("column sums:", a.sum(axis=0), "mean:", a.mean())
print("a[:, 1:3] =")
print(a[:, 1:3])
x = np.linspace(0.0, 1.0, 5)
print(x * 2 + 1, np.sqrt(x))
rng = np.random.default_rng(42)
print(rng.random(3))
m = np.array([[2.0, 1.0], [1.0, 3.0]])
print(np.linalg.solve(m, np.array([3.0, 5.0])))
