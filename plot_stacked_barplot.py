"""
stacked_bar_seaborn.py

Produces a stacked bar plot with seaborn/matplotlib:
- X-axis categories: GT/Pred combinations
- Y-axis: Percentage
- Bars: for the first two categories, two stacked sections (outside + correct),
  sharing the same base color with outside darker and correct lighter.
  For the last two categories, only outside is shown.
- Annotations: one decimal place for percentages
- Legend: "outside" shown as dark grey, "correct" shown as light grey

Run: python stacked_bar_seaborn.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

sns.set(style="whitegrid")

def lighten_color(color, amount=0.5):
    """
    Lighten the given color by blending it with white.
    amount=0 gives original color, amount=1 gives white.
    color: matplotlib color string or RGB tuple
    """
    try:
        c = mcolors.to_rgb(color)
    except Exception:
        c = color  # assume already an RGB tuple
    white = np.array([1.0, 1.0, 1.0])
    return tuple((1 - amount) * np.array(c) + amount * white)

def text_contrast_color(rgb):
    """
    Return 'white' or 'black' depending on luminance for legible text.
    rgb: tuple of floats 0..1
    """
    r, g, b = rgb
    # Perceived luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if lum > 0.6 else "white"

def make_stacked_bar_plot(categories, outside_vals, correct_vals, save_path=None):
    n = len(categories)
    x = np.arange(n)

    # pick distinct base colors for each category
    base_colors = sns.color_palette("tab10", n_colors=n)

    # prepare figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw bars
    bar_width = 0.6
    for i, cat in enumerate(categories):
        base = base_colors[i]
        outside = outside_vals[i]
        correct = correct_vals[i]

        # outside: darker / base color
        outside_color = base  # use base as the darker shade
        # correct: lighter shade (blend with white)
        correct_color = lighten_color(base, amount=0.55)

        # Plot outside section
        ax.bar(x[i], outside, width=bar_width, color=outside_color, edgecolor="none")

        # Plot correct stacked on top if > 0
        if correct > 0:
            ax.bar(x[i], correct, width=bar_width, bottom=outside, color=correct_color, edgecolor="none")

        # Annotate outside segment
        if outside > 0:
            y_text = outside / 2.0
            txt_color = text_contrast_color(mcolors.to_rgb(outside_color))
            ax.text(x[i], y_text, f"{outside:.1f}%", ha="center", va="center", color=txt_color, fontsize=10)

        # Annotate correct segment if present
        if correct > 0:
            y_text = outside + correct / 2.0
            txt_color = text_contrast_color(mcolors.to_rgb(correct_color))
            ax.text(x[i], y_text, f"{correct:.1f}%", ha="center", va="center", color=txt_color, fontsize=10)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=10, ha="right", fontsize=14)
    ax.set_ylabel("Percentage", fontsize=20)
    ax.set_ylim(0, max(np.array(outside_vals) + np.array(correct_vals)) * 1.1)  # add a bit headroom
    ax.set_title("GT/Pred categories — DomainNet (Painting)", fontsize=26)

    # Legend: user requested outside dark grey and correct light grey
    legend_handles = [
        Patch(facecolor="dimgray", edgecolor="none", label="outside"),
        Patch(facecolor="lightgrey", edgecolor="none", label="correct"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=14)

    sns.despine(offset=10, trim=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")

    plt.show()

if __name__ == "__main__":
    # Example data — replace these with your real numbers (percentages).
    categories = [
        "GT Known / Pred Known",
        "GT Known / Pred Unknown",
        "GT Unknown / Pred Known",
        "GT Unknown / Pred Unknown",
    ]

    # Percentages for 'outside' for each category
    outside_vals = [0.2, 10.5, 50.0, 61.2]  # example numbers

    # Percentages for 'correct' for each category (0 for the last two)
    correct_vals = [92.3, 61.2, 0.0, 0.0]  # example numbers

    # Sanity check: percentages should be in reasonable range
    for o, c in zip(outside_vals, correct_vals):
        if o < 0 or c < 0:
            raise ValueError("Percentages must be non-negative")

    make_stacked_bar_plot(categories, outside_vals, correct_vals, save_path="stacked_bar_plot.png")