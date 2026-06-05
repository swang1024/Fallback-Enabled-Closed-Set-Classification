import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as patches
import argparse

# Example confusion matrices (replace with your actual data)
cm1 = np.array([[512, 2588],
                [21, 3169]])
cm2 = np.array([[1827, 5879],
                [51, 7157]])
cm3 = np.array([[618, 1950],
                [36, 3625]])

matrices = [cm1, cm2, cm3]
titles = ['DomainNet (Painting)', 'DomainNet (Real)', 'DomainNet (Sketch)']  # Optional titles for each plot
labels = ['Known', 'Unknown']

# CLI: optional dataset name to prefix titles (e.g. VisDA, INaturalist)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--dataset', type=str, default=None, help='Optional dataset name to prefix plot titles')
parser.add_argument('--out', type=str, default='confusion_matrices.png', help='Output filename for saved figure')
args, _ = parser.parse_known_args()
if args.dataset:
    titles = [f"{args.dataset}: {t}" for t in titles]

# Example per-matrix subcategory proportions for the two top cells (True=Known, Pred=Known and Pred=Unknown).
# Each entry is a pair: (top-left proportions, top-right proportions). Each is an array-like of length 3 (A,B,C) and should sum to 1.
# Replace these with your actual proportions for each matrix.
subcat_props = [
    (np.array([0.002, 0.923, 1-0.002-0.923]), np.array([0.105, 0.612, 1-0.105-0.612])),
    (np.array([0.001, 0.942, 1-0.001-0.942]), np.array([0.049, 0.774, 1-0.049-0.774])),
    (np.array([0.002, 0.917, 1-0.002-0.917]), np.array([0.099, 0.663, 1-0.099-0.663])),
]

# Colors for subcategories A, B, C
# Top two subcells should be blue shades (top/middle), bottom subcell green
sub_colors = ['#6495ED',  # lighter blue (top)
              '#1f77b4',  # blue (middle)
              '#2ca02c']  # green (bottom)

# Colors for bottom-row subcategories (2 categories per bottom cell)
# other cell blue, bottom-most subcell green
bottom_colors = ['#1f77b4', '#2ca02c']

# Example per-matrix bottom-row proportions for the two bottom cells (True=Unknown row, Pred=Known and Pred=Unknown).
# Each entry is a pair: (bottom-left proportions, bottom-right proportions). Each is array-like of length 2 and should sum to 1.
bottom_subcat_props = [
    (np.array([0.5, 1-0.5]), np.array([0.612, 1-0.612])),
    (np.array([0.5, 0.5]), np.array([0.7, 0.3])),
    (np.array([0.25, 0.75]), np.array([0.6, 0.4])),
]

# Find global vmin/vmax for shared color scale
vmin = min(mat.min() for mat in matrices)
vmax = max(mat.max() for mat in matrices)

# Create subplots with extra room between them
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=True)
plt.subplots_adjust(wspace=0.6)

for i, ax in enumerate(axes):
    cm = matrices[i]

    # Draw the 2x2 heatmap (no colorbar)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                cbar=False, vmin=vmin, vmax=vmax, ax=ax,
                xticklabels=labels, yticklabels=labels, square=False)

    ax.set_xlabel('Predicted Binary Label', fontsize=20)
    if i == 0:
        ax.set_ylabel('True Binary Label', fontsize=20)
    else:
        ax.set_ylabel('')
    ax.set_title(titles[i], fontsize=14)

    # Compute PPV and NPV for the class 'Unknown' being the positive class
    # Layout of cm: [[TP, FN], [FP, TN]] where rows=true, cols=pred
    TP = cm[0, 0]
    FP = cm[1, 0]
    FN = cm[0, 1]
    TN = cm[1, 1]

    def safe_div(a, b):
        return a / b if b != 0 else np.nan

    ppv = safe_div(TP, (TP + FP))
    npv = safe_div(TN, (TN + FN))

    # Determine where to place the small right-side cells (data coordinates)
    # After a seaborn heatmap of shape (2,2), the data x range is [0,2] and y range is [0,2]
    # We'll draw two rectangles at x=2.05 to the right of the matrix, each height=1
    rect_x = cm.shape[1]  # this equals 2
    rect_w = 0.4
    rect_h = 1.0

    # Top cell: NPV (row 0). Note: heatmap origin='upper', so row 0 is at y=0..1
    top_y = 0
    bottom_y = 1

    # Add rectangles (white background with black edge) to the right of the matrix
    rect_top = patches.Rectangle((rect_x + 0.05, top_y), rect_w, rect_h,
                                 linewidth=1, edgecolor='black', facecolor='white', transform=ax.transData, zorder=5)
    rect_bot = patches.Rectangle((rect_x + 0.05, bottom_y), rect_w, rect_h,
                                 linewidth=1, edgecolor='black', facecolor='white', transform=ax.transData, zorder=5)
    ax.add_patch(rect_top)
    ax.add_patch(rect_bot)

    # Annotate numeric values centered in the small cells
    # Use percent format for clarity
    ppv_text = 'n/a' if np.isnan(ppv) else f"{ppv:.1%}"
    npv_text = 'n/a' if np.isnan(npv) else f"{npv:.1%}"

    # Show PPV on the top small cell and NPV on the bottom small cell
    ax.text(rect_x + 0.05 + rect_w / 2, top_y + rect_h / 2, ppv_text,
        ha='center', va='center', fontsize=11, zorder=6, transform=ax.transData)
    ax.text(rect_x + 0.05 + rect_w / 2, bottom_y + rect_h / 2, npv_text,
        ha='center', va='center', fontsize=11, zorder=6, transform=ax.transData)

    # Draw proportional stacked subcells inside the top two cells (True=Known row)
    # Get proportions for this matrix
    props_left, props_right = subcat_props[i]

    # Coordinates for the top-left cell (row 0, col 0)
    cell_x0 = 0
    cell_y0 = 0
    cell_w = 1.0
    cell_h = 1.0

    # Draw stacked rectangles within top-left cell from bottom->top proportionally
    y_cursor = cell_y0
    for j, prop in enumerate(props_left):
        h = cell_h * prop
        subrect = patches.Rectangle((cell_x0, y_cursor), cell_w, h,
                                    linewidth=0.5, edgecolor='black', facecolor=sub_colors[j], transform=ax.transData, zorder=4)
        ax.add_patch(subrect)
        # Label percentage in the middle of the sub-rect if there's enough space
        try:
            win_h = ax.get_window_extent().height
        except Exception:
            win_h = 100
        if h * win_h > 6:  # rough check; may be small
            ax.text(cell_x0 + cell_w / 2, y_cursor + h / 2, f"{prop:.0%}",
                    ha='center', va='center', fontsize=9, zorder=6, transform=ax.transData)
        y_cursor += h

    # Coordinates for the top-right cell (row 0, col 1)
    cell_x1 = 1
    cell_y1 = 0
    y_cursor = cell_y1
    for j, prop in enumerate(props_right):
        h = cell_h * prop
        subrect = patches.Rectangle((cell_x1, y_cursor), cell_w, h,
                                    linewidth=0.5, edgecolor='black', facecolor=sub_colors[j], transform=ax.transData, zorder=4)
        ax.add_patch(subrect)
        try:
            win_h = ax.get_window_extent().height
        except Exception:
            win_h = 100
        if h * win_h > 6:
            ax.text(cell_x1 + cell_w / 2, y_cursor + h / 2, f"{prop:.0%}",
                    ha='center', va='center', fontsize=9, zorder=6, transform=ax.transData)
        y_cursor += h

    # Now draw proportional stacked subcells inside the bottom two cells (True=Unknown row)
    bprops_left, bprops_right = bottom_subcat_props[i]

    # Coordinates for the bottom-left cell (row 1, col 0)
    bcell_x0 = 0
    bcell_y0 = 1
    bcell_w = 1.0
    bcell_h = 1.0

    y_cursor = bcell_y0
    for j, prop in enumerate(bprops_left):
        h = bcell_h * prop
        subrect = patches.Rectangle((bcell_x0, y_cursor), bcell_w, h,
                                    linewidth=0.5, edgecolor='black', facecolor=bottom_colors[j], transform=ax.transData, zorder=4)
        ax.add_patch(subrect)
        try:
            win_h = ax.get_window_extent().height
        except Exception:
            win_h = 100
        if h * win_h > 6:
            ax.text(bcell_x0 + bcell_w / 2, y_cursor + h / 2, f"{prop:.0%}",
                    ha='center', va='center', fontsize=9, zorder=6, transform=ax.transData)
        y_cursor += h

    # Coordinates for the bottom-right cell (row 1, col 1)
    bcell_x1 = 1
    bcell_y1 = 1
    y_cursor = bcell_y1
    for j, prop in enumerate(bprops_right):
        h = bcell_h * prop
        subrect = patches.Rectangle((bcell_x1, y_cursor), bcell_w, h,
                                    linewidth=0.5, edgecolor='black', facecolor=bottom_colors[j], transform=ax.transData, zorder=4)
        ax.add_patch(subrect)
        try:
            win_h = ax.get_window_extent().height
        except Exception:
            win_h = 100
        if h * win_h > 6:
            ax.text(bcell_x1 + bcell_w / 2, y_cursor + h / 2, f"{prop:.0%}",
                    ha='center', va='center', fontsize=9, zorder=6, transform=ax.transData)
        y_cursor += h

    # Add vertical labels to the right of the small cells, reading bottom-to-top
    # We place them slightly to the right of the rectangles; rotation=90 reads bottom->top
    label_x = rect_x + 0.05 + rect_w + 0.08
    # Label the small cells: ppv on the top, npv on the bottom (vertical, bottom->top)
    ax.text(label_x, top_y + rect_h / 2, 'ppv', rotation=90,
        va='center', ha='left', fontsize=10, transform=ax.transData)
    ax.text(label_x, bottom_y + rect_h / 2, 'npv', rotation=90,
        va='center', ha='left', fontsize=10, transform=ax.transData)

    # Expand x limits so the right-side cells and labels are visible
    ax.set_xlim(0, rect_x + rect_w + 0.4)
    # Ensure ticks align correctly
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(labels)

plt.tight_layout()
plt.savefig(args.out, dpi=300)