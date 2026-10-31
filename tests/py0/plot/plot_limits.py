"""ruai.plot against matplotlib: limits, ticks, labels and objects as matplotlib's."""
import numpy as np

import ruai.plot as plt


def show_axes(ax: plt.Axes) -> None:
    lo, hi = ax.get_xlim()
    bottom, top = ax.get_ylim()
    print(ax, round(float(lo), 9), round(float(hi), 9), round(float(bottom), 9), round(float(top), 9))
    print(ax.get_xticks(), ax.get_yticks())


x = np.linspace(0, 2 * np.pi, 50)
fig, ax = plt.subplots()
lines = ax.plot(x, np.sin(x), "r-", label="sin")
ax.plot(x, np.cos(x), "--")
print(fig, lines[0], lines[0].get_color(), lines[0].get_linestyle(), lines[0].get_marker(), lines[0].get_linewidth())
line2 = ax.get_lines()[1]
print(line2, line2.get_color(), line2.get_linestyle())
show_axes(ax)
ax.set_title("waves")
ax.set_xlabel("t")
print(ax.get_title(), ax.get_xlabel(), len(ax.get_lines()))
plt.close()

# integers, a single point, large and small values
for ys in [[1.0, 4.0, 9.0, 16.0], [3.0, 3.0], [0.001, 0.002, 0.0035], [120000.0, 250000.0, 990000.0], [-7.5, 2.25]]:
    fig, ax = plt.subplots()
    ax.plot(ys)
    show_axes(ax)
    plt.close()

# markers only, scatter, reference lines
fig, ax = plt.subplots(figsize=(4.0, 3.0))
marks = ax.plot([1.0, 2.0, 3.0], [2.0, 1.0, 3.0], "go")
print(fig, marks[0].get_linestyle(), marks[0].get_marker(), marks[0].get_color())
ax.scatter([0.5, 4.0], [1.5, -1.0], color="black")
ax.axhline(0.0, color="gray")
show_axes(ax)
plt.close()

# bars stick to zero; a categorical axis; a histogram
fig, ax = plt.subplots()
bars = ax.bar([1, 2, 3], [5.0, 3.0, 8.0])
print(bars, len(bars))
show_axes(ax)
plt.close()
fig, ax = plt.subplots()
ax.bar(["apples", "pears", "plums"], [3, 7, 2])
lo, hi = ax.get_xlim()
bottom, top = ax.get_ylim()
print(round(float(lo), 9), round(float(hi), 9), float(bottom), float(top))
plt.close()
data = np.array([1.0, 2.0, 2.5, 3.0, 3.1, 4.0, 7.5, 8.0])
counts, edges, patches = plt.hist(data, 5)
print(counts, edges, len(patches))
show_axes(plt.gca())
plt.close()

# an image: limits at the pixel edges, y downward
fig, ax = plt.subplots()
ax.imshow(np.arange(12.0).reshape(3, 4))
show_axes(ax)
plt.close()

# subplots in a row; subplot grid
fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0))
print(fig, axs[0], axs[2], len(fig.axes))
axs[1].plot([0.0, 1.0], [0.0, 100.0])
show_axes(axs[1])
plt.close()
plt.figure(figsize=(8.0, 6.0))
a = plt.subplot(2, 2, 1)
b = plt.subplot(2, 2, 4)
print(a, b, plt.gca())
plt.plot([1, 2, 3])
plt.xlim(0.0, 5.0)
lo, hi = plt.xlim()
print(float(lo), float(hi), plt.gcf().get_size_inches())
plt.close()

# errors
fig, ax = plt.subplots()
try:
    ax.plot([1.0, 2.0], [1.0])
except ValueError as e:
    print("ValueError:", e)
try:
    ax.plot([1.0, 2.0], "q-")
except ValueError as e:
    print("ValueError:", e)
plt.close()
