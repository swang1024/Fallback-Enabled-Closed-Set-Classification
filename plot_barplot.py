import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Dummy data setup
num_models = 4
categories = ['pred1 outside of list', 'pred1 within list \n & pred1==pred2', 'pred1 within list \n & pred1!=pred2']
num_subplots = 6
model_names = ['Qwen', 'Llama3.2', 'ChatGPT 4o-mini', '']
titles = ['DomainNet (Painting)', 'DomainNet (Real)', 'DomainNet (Sketch)', 'VisDA', 'INaturalist (Phylum)', 'INaturalist (Class)']

# Generate dummy known/unknown data
np.random.seed(42)

# results[0, 0, :, :] = np.array([[1.72, 40.30], [64.08, 12.92], [34.2, 46.78]])  # Qwen on painting
# results[0, 1, :, :] = np.array([[24.55, 69.19], [48.02, 11.64], [27.43, 19.17]])  # Qwen on painting
# results[1, 0, :, :] = np.array([[1.02, 48.68], [80.53, 13.2], [12.68, 38.12]])  # Qwen on real
# results[1, 1, :, :] = np.array([[29.06, 78.68], [55.30, 9.01], [15.64, 12.31]])  # Qwen on real
# results[2, 0, :, :] = np.array([[1.64, 42.06], [71.34, 14.20], [27.01, 43.74]])  # Qwen on sketch
# results[2, 1, :, :] = np.array([[43.76, 79.39], [41.90, 8.21], [14.34, 12.4]])  # Qwen on sketch
# results[3, 0, :, :] = np.array([[0.035, 8.78], [82.76, 38.82], [17.21, 52.4]])  # Qwen on sketch
# results[3, 1, :, :] = np.array([[36.26, 54.33], [55.32, 25.10], [8.42, 20.57]])  # Qwen on sketch

# results[4, 0, :, :] = np.array([[0.58, 78.71], [96.45, 0], [3.55, 21.29]])  # Qwen on INaturalist (Phylum)

# results[5, 0, :, :] = np.array([[2.87, 25.99], [80.89, 7.70], [16.24, 66.31]])  # Qwen on INaturalist (Phylum)
# results[5, 1, :, :] = np.array([[36.50, 51.07], [54.5, 6.50], [9, 42.43]])  # Qwen on INaturalist (Phylum)

# 1. Hand input data
hand_input_data = [
    {'Plot': titles[0], 'Model': model_names[0], 'Category': categories[0], 'Known': 1.72, 'Unknown': 40.30},
    {'Plot': titles[0], 'Model': model_names[0], 'Category': categories[1], 'Known': 64.08, 'Unknown': 12.92},
    {'Plot': titles[0], 'Model': model_names[0], 'Category': categories[2], 'Known': 34.2, 'Unknown': 46.78},
    {'Plot': titles[0], 'Model': model_names[1], 'Category': categories[0], 'Known': 24.55, 'Unknown': 69.19},
    {'Plot': titles[0], 'Model': model_names[1], 'Category': categories[1], 'Known': 48.02, 'Unknown': 11.64},
    {'Plot': titles[0], 'Model': model_names[1], 'Category': categories[2], 'Known': 27.43, 'Unknown': 19.17},
    {'Plot': titles[0], 'Model': model_names[2], 'Category': categories[0], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[0], 'Model': model_names[2], 'Category': categories[1], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[0], 'Model': model_names[2], 'Category': categories[2], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[0], 'Model': model_names[3], 'Category': categories[0], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[0], 'Model': model_names[3], 'Category': categories[1], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[0], 'Model': model_names[3], 'Category': categories[2], 'Known': 0, 'Unknown': 0},
    
    {'Plot': titles[1], 'Model': model_names[0], 'Category': categories[0], 'Known': 1.02, 'Unknown': 48.68},
    {'Plot': titles[1], 'Model': model_names[0], 'Category': categories[1], 'Known': 80.53, 'Unknown': 13.2},
    {'Plot': titles[1], 'Model': model_names[0], 'Category': categories[2], 'Known': 12.68, 'Unknown': 38.12},
    {'Plot': titles[1], 'Model': model_names[1], 'Category': categories[0], 'Known': 29.06, 'Unknown': 78.68},
    {'Plot': titles[1], 'Model': model_names[1], 'Category': categories[1], 'Known': 55.30, 'Unknown': 9.01},
    {'Plot': titles[1], 'Model': model_names[1], 'Category': categories[2], 'Known': 15.64, 'Unknown': 12.31},
    {'Plot': titles[1], 'Model': model_names[2], 'Category': categories[0], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[1], 'Model': model_names[2], 'Category': categories[1], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[1], 'Model': model_names[2], 'Category': categories[2], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[1], 'Model': model_names[3], 'Category': categories[0], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[1], 'Model': model_names[3], 'Category': categories[1], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[1], 'Model': model_names[3], 'Category': categories[2], 'Known': 0, 'Unknown': 0},
    
    {'Plot': titles[2], 'Model': model_names[0], 'Category': categories[0], 'Known': 1.64, 'Unknown': 42.06},
    {'Plot': titles[2], 'Model': model_names[0], 'Category': categories[1], 'Known': 71.34, 'Unknown': 14.20},
    {'Plot': titles[2], 'Model': model_names[0], 'Category': categories[2], 'Known': 27.01, 'Unknown': 43.74},
    {'Plot': titles[2], 'Model': model_names[1], 'Category': categories[0], 'Known': 43.76, 'Unknown': 79.39},
    {'Plot': titles[2], 'Model': model_names[1], 'Category': categories[1], 'Known': 41.90, 'Unknown': 8.21},
    {'Plot': titles[2], 'Model': model_names[1], 'Category': categories[2], 'Known': 14.34, 'Unknown': 12.4},
    {'Plot': titles[2], 'Model': model_names[2], 'Category': categories[0], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[2], 'Model': model_names[2], 'Category': categories[1], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[2], 'Model': model_names[2], 'Category': categories[2], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[2], 'Model': model_names[3], 'Category': categories[0], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[2], 'Model': model_names[3], 'Category': categories[1], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[2], 'Model': model_names[3], 'Category': categories[2], 'Known': 0, 'Unknown': 0},

    {'Plot': titles[3], 'Model': model_names[0], 'Category': categories[0], 'Known': 0.035, 'Unknown': 8.78},
    {'Plot': titles[3], 'Model': model_names[0], 'Category': categories[1], 'Known': 82.76, 'Unknown': 38.82},
    {'Plot': titles[3], 'Model': model_names[0], 'Category': categories[2], 'Known': 17.21, 'Unknown': 52.4},
    {'Plot': titles[3], 'Model': model_names[1], 'Category': categories[0], 'Known': 36.26, 'Unknown': 54.33}, 
    {'Plot': titles[3], 'Model': model_names[1], 'Category': categories[1], 'Known': 55.32, 'Unknown': 25.10},
    {'Plot': titles[3], 'Model': model_names[1], 'Category': categories[2], 'Known': 8.42, 'Unknown': 20.57},
    {'Plot': titles[3], 'Model': model_names[2], 'Category': categories[0], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[3], 'Model': model_names[2], 'Category': categories[1], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[3], 'Model': model_names[2], 'Category': categories[2], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[3], 'Model': model_names[3], 'Category': categories[0], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[3], 'Model': model_names[3], 'Category': categories[1], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[3], 'Model': model_names[3], 'Category': categories[2], 'Known': 0, 'Unknown': 0},

    {'Plot': titles[4], 'Model': model_names[0], 'Category': categories[0], 'Known': 0.58, 'Unknown': 78.71},
    {'Plot': titles[4], 'Model': model_names[0], 'Category': categories[1], 'Known': 96.45, 'Unknown': 0},
    {'Plot': titles[4], 'Model': model_names[0], 'Category': categories[2], 'Known': 3.55, 'Unknown': 21.29},
    {'Plot': titles[4], 'Model': model_names[1], 'Category': categories[0], 'Known': 6.96, 'Unknown': 11.3},
    {'Plot': titles[4], 'Model': model_names[1], 'Category': categories[1], 'Known': 88.5, 'Unknown': 4.54},
    {'Plot': titles[4], 'Model': model_names[1], 'Category': categories[2], 'Known': 45.4, 'Unknown': 43.3},
    {'Plot': titles[4], 'Model': model_names[2], 'Category': categories[0], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[4], 'Model': model_names[2], 'Category': categories[1], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[4], 'Model': model_names[2], 'Category': categories[2], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[4], 'Model': model_names[3], 'Category': categories[0], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[4], 'Model': model_names[3], 'Category': categories[1], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[4], 'Model': model_names[3], 'Category': categories[2], 'Known': 0, 'Unknown': 0},

    {'Plot': titles[5], 'Model': model_names[0], 'Category': categories[0], 'Known': 2.87, 'Unknown': 25.99},
    {'Plot': titles[5], 'Model': model_names[0], 'Category': categories[1], 'Known': 80.89, 'Unknown': 7.70},
    {'Plot': titles[5], 'Model': model_names[0], 'Category': categories[2], 'Known': 16.24, 'Unknown': 66.31},
    {'Plot': titles[5], 'Model': model_names[1], 'Category': categories[0], 'Known': 36.50, 'Unknown': 51.07},
    {'Plot': titles[5], 'Model': model_names[1], 'Category': categories[1], 'Known': 54.50, 'Unknown': 6.50},
    {'Plot': titles[5], 'Model': model_names[1], 'Category': categories[2], 'Known': 9.00, 'Unknown': 42.43},
    {'Plot': titles[5], 'Model': model_names[2], 'Category': categories[0], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[5], 'Model': model_names[2], 'Category': categories[1], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[5], 'Model': model_names[2], 'Category': categories[2], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[5], 'Model': model_names[3], 'Category': categories[0], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[5], 'Model': model_names[3], 'Category': categories[1], 'Known': 0, 'Unknown': 0},
    {'Plot': titles[5], 'Model': model_names[3], 'Category': categories[2], 'Known': 0, 'Unknown': 0},
    # ... add more rows for each model, category, and plot
]

# 2. Build DataFrame and Diff column
df = pd.DataFrame(hand_input_data)
df['Diff'] = df['Known'] - df['Unknown']

# Color palette: 3 colors for 3 categories
category_colors = {
    categories[0]: '#4e79a7',
    categories[1]: '#f28e2c',
    categories[2]: '#59a14f',
}

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey='row')
plt.subplots_adjust(wspace=0.3, hspace=0.3, right=0.80)

bar_width = 0.8
gap = 1.7  # gap between model groups

for idx, ax in enumerate(axes.flat):
    plot_name = titles[idx]
    plot_data = df[df['Plot'] == plot_name]
    bar_positions = []
    bar_heights = []
    bar_colors = []
    xtick_positions = []
    xtick_labels = []

    # Compute bar positions so that each model group is spaced
    for m_idx, model_name in enumerate(model_names):
        model_data = plot_data[plot_data['Model'] == model_name]
        for c_idx, cat in enumerate(categories):
            bar_positions.append(m_idx * (len(categories) + gap) + c_idx)
            bar_heights.append(
                model_data[model_data['Category'] == cat]['Diff']
            )
            bar_colors.append(category_colors[cat])
        # Set x-tick in the middle of each model cluster
        xtick_positions.append(m_idx * (len(categories) + gap) + (len(categories)-1)/2)
        xtick_labels.append(model_name)

    bar_positions = np.asarray(bar_positions).ravel()      # (N,)
    bar_heights   = np.asarray(bar_heights).ravel()        # (N,)
    bars = ax.bar(bar_positions, bar_heights, color=bar_colors, width=bar_width, edgecolor='black')

    # X axis: only show model names at group centers
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=13, rotation=15)

    # Y axis: ticks only for leftmost plots
    if idx % 3 == 0:
        ax.tick_params(axis='y', labelsize=13)
    else:
        ax.set_yticklabels([])
        ax.set_yticks([])

    ax.set_title(plot_name, fontsize=15)
    if idx % 3 == 0:
        ax.set_ylabel('Known - Unknown', fontsize=14)
    else:
        ax.set_ylabel('')

# Legend: one for all plots, on the right
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=category_colors[cat], edgecolor='black', label=cat) for cat in categories]
fig.legend(handles=legend_patches,
           loc='center left', bbox_to_anchor=(0.85, 0.5),
           fontsize=13, title="Category", title_fontsize=14)

plt.tight_layout(rect=[0, 0, 0.80, 1])
plt.savefig('barplot_grid_known_unknown_diff.png', dpi=300, bbox_inches='tight')