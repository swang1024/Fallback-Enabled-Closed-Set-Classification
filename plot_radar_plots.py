import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

categories = ['A', 'B', 'C', 'D', 'E', 'F']
labels = ['TN label prediction accuracy', 'TN percentage of prediction \n label outside of list', 'TP percentage of prediction \n label outside of list', \
'FP label prediction accuracy', 'FP percentage of prediction \n label outside of list', 'FN percentage of prediction \n label outside of list']
N = len(categories)

data_list = [
    [0.923, 0.002, 0.360, 0.612, 0.105, 0.500],   # Plot 1
    [0.942, 0.001, 0.358, 0.774, 0.049, 0.501],   # Plot 2
    [0.917, 0.002, 0.375, 0.663, 0.099, 0.350],   # Plot 3
]
titles = ['DomainNet (Painting)', 'DomainNet (Real)', 'DomainNet (Sketch)']

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

colors = sns.color_palette("Set2", N)

fig, axs = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(15, 5))

for i, ax in enumerate(axs):
    values = data_list[i] + data_list[i][:1]
    ax.plot(angles, values, color='grey', alpha=0.5, linewidth=1)
    ax.fill(angles, values, alpha=0.15, color='grey')
    # Draw lines for each edge (for visual, not for legend)
    for j in range(N):
        ax.plot([angles[j], angles[j]], [0, values[j]], color=colors[j], linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(titles[i], fontsize=14, pad=20)
    ax.set_yticklabels([])

legend_handles = [Line2D([0], [0], color=colors[j], lw=2) for j in range(N)]
# Legend to the right
fig.legend(legend_handles, labels, loc='right', bbox_to_anchor=(1.0, 0.5))

plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space for legend
plt.savefig('radar_plots.pdf', dpi=300)