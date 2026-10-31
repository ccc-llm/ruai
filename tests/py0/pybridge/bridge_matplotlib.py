"""The Python bridge with matplotlib (the shipped stubs): figures, axes, lines, saving files."""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
print(matplotlib.get_backend())
x = np.linspace(0, 2 * np.pi, 50)
fig, ax = plt.subplots(figsize=(6.0, 4.0))
lines = ax.plot(x, np.sin(x), "r-", label="sin")
ax.plot(x, np.cos(x), label="cos", linestyle="--")
ax.scatter([1, 2, 3], [0.5, 0.0, -0.5], color="black")
ax.set_title("waves")
ax.set_xlabel("x")
ax.set_xlim(0.0, 6.5)
ax.legend()
lo, hi = ax.get_xlim()
print(fig, ax.get_title(), ax.get_xlabel(), float(lo), float(hi), len(ax.get_lines()), lines[0].get_label(), lines[0].get_color())
print(np.round(lines[0].get_ydata()[:3], 4), fig.get_size_inches())
path = "bridge_mpl_" + str(os.getpid()) + ".png"
fig.savefig(path, dpi=50)
print(os.path.exists(path), os.path.getsize(path) > 1000)
os.remove(path)
plt.figure(figsize=(3.0, 2.0))
plt.plot([1, 4, 9, 16])
plt.bar(["a", "b"], [3, 5])
plt.hist(np.arange(20.0), 5)
plt.title("pyplot")
print(plt.gca().get_title(), plt.gcf())
plt.close()
