#!/usr/bin/env python3
"""Plot sample counts for a target class list from a CSV of predictions.

Usage examples:
  python3 plot_class_counts.py \
      --csv llm_data/DomainNet_target_domain0_4o-mini_v13_summary_pred.csv \
      --out class_counts.png

Options:
  --csv       Path to CSV (must contain a ground truth column: 'ground truth' or 'ground_truth')
  --out       Output PNG path (default: class_counts.png)
  --include-private   Include rows with any private value (default: only private==False)
  --sort-by   'count' (default) or 'name'

The script normalizes labels before counting using Unicode NFKC, strips surrounding brackets/quotes
repeatedly, and collapses internal whitespace. It maps normalized ground-truth values to the
canonical entries in TARGET_CLASS_LIST.
"""

import argparse
import unicodedata
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

sns.set(style="whitegrid")


def plot_phylum_and_class_counts(phylum_names, phylum_counts,
                 class_names, class_counts,
                 out_path="inaturalist_phylum_class_counts.png",
                 rotate_xticks=90,
                 annotate=True,
                 dpi=200):
  """
  Create two vertically stacked bar plots:
    - top: Phylum (phylum_names / phylum_counts)
    - bottom: Class  (class_names  / class_counts)

  Inputs:
    phylum_names: list[str]
    phylum_counts: list[int]
    class_names:  list[str]
    class_counts: list[int]
  """
  # defensive conversions
  phylum_names = list(map(str, phylum_names))
  class_names = list(map(str, class_names))
  phylum_counts = list(map(int, phylum_counts))
  class_counts = list(map(int, class_counts))

  max_bars = max(len(phylum_names), len(class_names), 8)
  fig_w = max(10, max_bars * 0.25)
  fig_h = 10
  fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), constrained_layout=False)

  # Top plot: Phylum
  ax = axes[0]
  # sort bars by count (big to small left->right)
  if len(phylum_counts) > 0:
    order = np.argsort(phylum_counts)[::-1]
    ph_names_sorted = [phylum_names[i] for i in order]
    ph_counts_sorted = [phylum_counts[i] for i in order]
  else:
    ph_names_sorted = phylum_names
    ph_counts_sorted = phylum_counts

  x = np.arange(len(ph_names_sorted))
  bars = ax.bar(x, ph_counts_sorted, color="tab:blue", edgecolor="black")
  ax.set_title("INaturalist — Phylum (sample counts)", fontsize=20)
  ax.set_ylabel("Sample count", fontsize=15)
  ax.set_xticks(x)
  ax.set_xticklabels(ph_names_sorted, rotation=rotate_xticks, ha="right", fontsize=12)

  if annotate:
    ymax = max(ph_counts_sorted) if ph_counts_sorted else 1
    for rect, val in zip(bars, ph_counts_sorted):
      h = rect.get_height()
      if h > 0:
        ax.text(rect.get_x() + rect.get_width() / 2., h + max(1, ymax * 0.01),
            f"{val}", ha="center", va="bottom", fontsize=8)

  # Bottom plot: Class
  ax = axes[1]
  # sort class bars by count (big to small left->right)
  if len(class_counts) > 0:
    order = np.argsort(class_counts)[::-1]
    class_names_sorted = [class_names[i] for i in order]
    class_counts_sorted = [class_counts[i] for i in order]
  else:
    class_names_sorted = class_names
    class_counts_sorted = class_counts

  x = np.arange(len(class_names_sorted))
  bars = ax.bar(x, class_counts_sorted, color="tab:green", edgecolor="black")
  ax.set_title("INaturalist — Class (sample counts)", fontsize=20)
  ax.set_ylabel("Sample count", fontsize=15)
  ax.set_xticks(x)
  ax.set_xticklabels(class_names_sorted, rotation=rotate_xticks, ha="right", fontsize=12)

  if annotate:
    ymax = max(class_counts_sorted) if class_counts_sorted else 1
    for rect, val in zip(bars, class_counts_sorted):
      h = rect.get_height()
      if h > 0:
        ax.text(rect.get_x() + rect.get_width() / 2., h + max(1, ymax * 0.01),
            f"{val}", ha="center", va="bottom", fontsize=8)

  plt.tight_layout(rect=(0, 0, 1, 0.97))
  out_path = Path(out_path)
  fig.savefig(out_path, dpi=dpi)
  plt.close(fig)
  print(f"Saved figure to {out_path}")


def _parse_list_arg(s):
  # Accept comma- or pipe-separated lists
  if s is None:
    return None
  if isinstance(s, (list, tuple)):
    return list(s)
  if '|' in s:
    parts = [p.strip() for p in s.split('|') if p.strip()]
  else:
    parts = [p.strip() for p in s.split(',') if p.strip()]
  return parts

phylum_names = ['Chlorophyta', 'Cnidaria', 'Ascomycota', 'Tracheophyta', 'Marchantiophyta', 'Mollusca', 'Rhodophyta', 'Arthropoda', 'Annelida']
phylum_counts = [40, 260, 840, 42180, 70, 1690, 70, 27520, 40]

class_names = ['Arthoniomycetes', 'Aves', 'Marchantiopsida', 'Ulvophyceae', 'Jungermanniopsida', 'Malacostraca', 'Hexanauplia', 'Pucciniomycetes', 'Polychaeta', 'Pinopsida', 'Sphagnopsida', 'Clitellata', 'Diplopoda', 'Dacrymycetes', 'Echinoidea', 'Mammalia', 'Polypodiopsida', 'Dothideomycetes', 'Agaricomycetes', 'Florideophyceae', 'Scyphozoa', 'Merostomata', 'Gnetopsida', 'Elasmobranchii', 'Bryopsida', 'Ophiuroidea', 'Leotiomycetes', 'Asteroidea', 'Gastropoda', 'Pezizomycetes', 'Sordariomycetes', 'Anthozoa', 'Arachnida', 'Polyplacophora', 'Magnoliopsida', 'Cephalopoda']
class_counts = [20, 14860, 50, 40, 20, 540, 70, 30, 30, 860, 10, 10, 60, 50, 70, 2460, 1400, 10, 2450, 70, 60, 10, 20, 160, 300, 10, 60, 110, 1220, 130, 70, 170, 1530, 90, 32950, 50]

plot_phylum_and_class_counts(phylum_names, phylum_counts, class_names, class_counts,
                             out_path="inaturalist_class_counts.pdf")