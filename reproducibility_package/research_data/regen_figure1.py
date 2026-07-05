import sys
sys.path.insert(0, "src")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from run_experiments_v2 import generate_scenario, graph_search, genetic_search, qkg_qi_search

scenario = generate_scenario(12)  # identical scenario used in the original Figure 1

methods_paths = {
    "A*": graph_search(scenario, astar=True, mode="distance")[0],
    "Risk-Aware A*": graph_search(scenario, astar=True, mode="risk")[0],
    "GA": genetic_search(scenario, trials=55)[0],
    "QKG-QI": qkg_qi_search(scenario)[0],
}

colors = {
    "A*": "#5B9BD5",
    "Risk-Aware A*": "#ED7D31",
    "GA": "#A5A5A5",
    "QKG-QI": "#2E7D32",
}
linestyles = {
    "A*": (0, (4, 2)),
    "Risk-Aware A*": (0, (1, 1)),
    "GA": (0, (6, 2, 1, 2)),
    "QKG-QI": "solid",
}
linewidths = {"A*": 2.0, "Risk-Aware A*": 2.0, "GA": 2.0, "QKG-QI": 3.0}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10.5,
    "font.family": "DejaVu Sans",
})

fig, ax = plt.subplots(figsize=(8.2, 7.2))

im = ax.imshow(scenario["threat"].T, origin="lower", cmap="YlOrRd", vmin=0, vmax=1,
                extent=(0, 35, 0, 35), aspect="equal")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Threat intensity T(v)", fontsize=11)

nf = np.argwhere(scenario["no_fly"] == 1)
if len(nf) > 0:
    ax.scatter(nf[:, 0] + 0.5, nf[:, 1] + 0.5, marker="s", s=22, c="#3B3B3B",
               alpha=0.75, label="No-fly cells", linewidths=0, zorder=3)

for name, path in methods_paths.items():
    if path:
        xs = [p[0] + 0.5 for p in path]
        ys = [p[1] + 0.5 for p in path]
        ax.plot(xs, ys, linewidth=linewidths[name], linestyle=linestyles[name],
                 color=colors[name], label=name, zorder=4,
                 path_effects=[pe.Stroke(linewidth=linewidths[name] + 1.5, foreground="white"), pe.Normal()],
                 solid_capstyle="round")

start, goal = scenario["start"], scenario["goal"]
ax.scatter([start[0] + 0.5], [start[1] + 0.5], marker="o", s=140, c="#1565C0",
           edgecolors="white", linewidths=1.5, label="Start", zorder=5)
ax.scatter([goal[0] + 0.5], [goal[1] + 0.5], marker="*", s=260, c="#C62828",
           edgecolors="white", linewidths=1.2, label="Goal", zorder=5)

ax.set_title("Contested UAV Mission Scenario: Threat Heatmap and Route Comparison", fontsize=13.5, pad=12)
ax.set_xlabel("Grid X (cell index)")
ax.set_ylabel("Grid Y (cell index)")
ax.set_xlim(0, 35)
ax.set_ylim(0, 35)
ax.set_xticks(range(0, 36, 5))
ax.set_yticks(range(0, 36, 5))

handles, labels = ax.get_legend_handles_labels()
order = ["QKG-QI", "Risk-Aware A*", "A*", "GA", "Start", "Goal", "No-fly cells"]
ordered = sorted(zip(handles, labels), key=lambda hl: order.index(hl[1]) if hl[1] in order else 99)
handles, labels = zip(*ordered)
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.11),
          ncol=4, frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig("figs/figure1_risk_heatmap_route_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved")
