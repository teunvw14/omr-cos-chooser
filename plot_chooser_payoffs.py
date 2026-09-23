import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# MatPlotLib theming
plt.rcParams.update({
    'text.color': 'white',          # Global text color
    'legend.facecolor': '#002A3E',
    'axes.labelcolor': 'white',      # X and Y label color
    'axes.titlecolor': 'white',     # Title color
    'xtick.color': 'white',         # X tick mark and label color
    'ytick.color': 'white',         # Y tick mark and label color
    'axes.edgecolor': 'gray'        # Border line color around the plot
})


# Parameters
x = np.linspace(50, 150, 1001)
out_dir = Path("./data/")
out_dir.mkdir(exist_ok=True)

# Symmetric strike pairs: start together, then move apart in both directions.
pairs = [(100 - d, 100 + d) for d in range(-25, 26, 1)]

# Fixed y-range for all PNGs so they can be assembled into a GIF cleanly.
y_min, y_max = -10, 80

for i, (K_C, K_P) in enumerate(pairs):
    f = np.maximum(0, x - K_C)
    g = np.maximum(0, K_P - x)
    max_fg = np.maximum(f, g)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#001B29')
    ax.set_facecolor('#001B29')

    ax.plot(x, max_fg, label=r"$\max(f,g)$", color='#fe218b', linewidth=3)
    ax.plot(x, f, label=r"$f(x,K_C)=(x-K_C)^+$", linewidth=3, color='#fed700', linestyle=(0, (3, 5, 1, 5)))
    ax.plot(x, g, label=r"$g(x,K_P)=(K_P-x)^+$", linewidth=3, color='#21b0fe', linestyle=(0, (3, 5, 1, 5)))

    ax.set_xlim(50, 150)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("S(T)")
    ax.set_ylabel("Payoff")
    ax.set_title(f"Chooser Option Payoff $K_C={K_C},\\ K_P={K_P}$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = out_dir / f"frame_{i:02d}_KC_{K_C}_KP_{K_P}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

print(f"Created {len(pairs)} PNGs in {out_dir}")

# print(f"\nFrame order: {manifest}")
