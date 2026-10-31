"""ruai.plot writes figures as SVG and HTML files."""
import os

import numpy as np

import ruai.plot as plt

x = np.linspace(-3, 3, 61)
fig, ax = plt.subplots()
ax.plot(x, x * x, label="x²")
ax.plot(x, np.exp(-x * x), "k:", label="gauss")
ax.scatter([0.0, 1.0], [1.0, 1.0], marker="s", label="points")
ax.legend()
ax.grid(True)
ax.set_title("functions <&>")
plt.suptitle("ruai.plot")
for ext in ["svg", "html"]:
    path = "plot_files_" + str(os.getpid()) + "." + ext
    fig.savefig(path)
    with open(path) as f:
        text = f.read()
    print(ext, "<svg" in text or "<canvas" in text, "functions" in text)
    os.remove(path)
plt.close()
