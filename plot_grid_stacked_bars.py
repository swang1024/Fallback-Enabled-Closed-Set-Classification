"""Create a 2x3 grid of grouped stacked bar charts using seaborn/matplotlib.

Each subplot shows 4 model groups. Each group contains two stacked bars: "Known" and "Unknown".
Each bar is stacked as: bottom = Predicted Correct, top = Predicted Incorrect.

Layout & style requirements implemented:
- 2 rows x 3 columns of subplots
- 4 model sets per subplot; each model set has 2 bars (Known, Unknown)
- Two stacked components per bar (Predicted Correct, Predicted Incorrect)
- 6 plots share a single legend placed to the right of the grid
- Each model set uses a distinct color pair (same pair for Known & Unknown of that model)
- Larger horizontal gap between different model groups
- Y-axis ticks only shown on the left-most column (fontsize 13)
- Bars have black outlines

This script generates sample data. Replace `data` with your real values if required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Default model names and plot titles (can be overridden by caller)
DEFAULT_MODEL_NAMES = ["LLaMa", "Qwen", "4o-mini", "2.0-flash"]
DEFAULT_PLOT_TITLES = ["DomainNet (Painting)", "DomainNet (Real)", "DomainNet (Sketch)",
                   "VisDA", "INaturalist (Phylum)", "INaturalist (Class)"]

# Editable default data array.
# Structure: data_array[plot_idx][model_idx] = [ [known_correct, known_incorrect], [unknown_correct, unknown_incorrect] ]
# So each model entry contains two lists: Known and Unknown, each with two components.
# Example for one plot with two models:
# [
#   [[50,10], [40,20]],  # model 0: Known (50 correct, 10 incorrect), Unknown (40 correct, 20 incorrect)
#   [[60,5],  [30,15]],  # model 1
# ]
DEFAULT_DATA_ARRAY = [
    # plot 1 domain 0
    [
        [[81.08* 9733/(9733+2271), 81.08*2271/(9733+2271)], [64.88* 18164/(18164+9832), 64.88* 9832/(18164+9832)]],
        [[83.05* 2244/(2244+458), 83.05* 458/(2244+458)], [76.15* 2733/(2733+856), 76.15* 856/(2733+856)]],
        [[65.31* 14520/(7711+14520), 65.31* 7711/(7711+14520)], [71.17* 12647/(5122+12647), 71.17* 5122/(5122+12647)]],
        [[68.78* 14093/(6397+14093), 68.78* 6397/(6397+14093)], [71.56* 13961/(5549+13961), 71.56* 5549/(5549+13961)]],
    ],
    # plot 2
    [
        [[87.27* 11548/(11548+1684), 87.27* 1684/(11548+1684)], [65.71* 17590/(17590+9178), 65.71* 9178/(17590+9178)]],
        [[87.64* 6422/(6422+906), 87.64* 906/(6422+906)], [83.07* 6302/(6302+1284), 83.07* 1284/(6302+1284)]],
        [[71.34* 17735/(7124+17735), 71.34* 7124/(7124+17735)], [79.31* 12008/(3133+12008), 79.31* 3133/(3133+12008)]],
        [[75.47* 13326/(4331+13326), 75.47* 4331/(4331+13326)], [72.15* 12138/(12138+4686), 72.15* 4686/(12138+4686)]],
    ],
    # plot 3
    [
        [[77.99* 7110/(2007+7110), 77.99* 2007/(2007+7110)], [68.65* 21200/(21200+9683), 68.65* 9683/(21200+9683)]],
        [[77.32* 12718/(12718+3730), 77.32* 3730/(12718+3730)], [82.95* 19537/(4015+19537), 82.95* 4015/(4015+19537)]],
        [[60.66* 13581/(8808+13581), 60.66* 8808/(8808+13581)], [82.10* 14459/(3152+14459), 82.10* 3152/(3152+14459)]],
        [[67.49* 12161/(12161+5858), 67.49* 5858/(12161+5858)], [79.2* 17409/(17409+4572), 79.2* 4572/(17409+4572)]],
    ],
    # plot 4
    [
        [[85.21* 15700/(15700+2724), 85.21* 2724/(15700+2724)], [41.85* 9030/(9030+12546), 41.85* 12546/(9030+12546)]],
        [[89.35* 2500/(2500+298), 89.35* 298/(2500+298)], [64.19* 839/(839+468), 64.19* 468/(839+468)]],
        [[78.48* 26659/(7312+26659), 78.48* 7312/(7312+26659)], [73.43* 4427/(1602+4427), 73.43* 1602/(1602+4427)]],
        [[82.94* 12854/(2644+12854), 82.94* 2644/(2644+12854)], [70.84* 3189/(1313+3189), 70.84* 1313/(1313+3189)]],
    ],
    # plot 5
    [
        [[75.01* 9178/(9178+3058), 75.01* 3058/(9178+3058)], [73.20* 4219/(4219+1545), 73.20* 1545/(4219+1545)]],
        [[90.73* 9023/(9023+922), 90.73* 922/(9023+922)], [78.34* 6310/(6310+1745), 78.34* 1745/(6310+1745)]],
        [[70.8* 10105/(4164+10105), 70.8* 4164/(4164+10105)], [82.2* 3068/(663+3068), 82.2* 663/(663+3068)]],
        [[72.00* 10466/(4071+10466), 72.00* 4071/(4071+10466)], [91.28* 3161/(302+3161), 91.28* 302/(302+3161)]],
    ],
    # plot 6
    [
        [[85.66* 3686/(3686+617), 85.66* 617/(3686+617)], [76.51* 10479/(3218+10479), 76.51* 3218/(3218+10479)]],
        [[89.96* 5621/(5621+627), 89.96* 627/(5621+627)], [88.28 * 10375/(10375+1377), 88.28* 1377/(10375+1377)]],
        [[52.11* 3045/(2798+3045), 52.11* 2798/(2798+3045)], [85.71* 2706/(451+2706), 85.71* 451/(451+2706)]],
        [[53.85* 5271/(5271+6150), 53.85* 6150/(5271+6150)], [87.11* 5731/(5731+848), 87.11* 848/(5731+848)]],
    ],
]


def build_plots_from_array(data_array, model_names=None):
    """Convert the editable nested list `data_array` into a list of DataFrames for plotting.

    data_array shape:
      list of plots, each plot is list of models, each model is [[known_correct, known_incorrect], [unknown_correct, unknown_incorrect]]

    model_names: optional list of names (length must match number of models per plot)

    Returns: list of pandas.DataFrame objects compatible with plot_grid_stacked_bars.
    """
    plots = []
    if not isinstance(data_array, list):
        raise ValueError("data_array must be a list of plots")

    for p_idx, plot in enumerate(data_array):
        if not isinstance(plot, list):
            raise ValueError(f"Each plot (index {p_idx}) must be a list of models")
        n_models = len(plot)
        if model_names is not None and len(model_names) < n_models:
            raise ValueError("model_names must contain at least as many names as there are models in each plot")

        rows = []
        for m_idx, model_entry in enumerate(plot):
            if not (isinstance(model_entry, list) and len(model_entry) == 2):
                raise ValueError(f"Model entry at plot {p_idx}, model {m_idx} must be [[known_correct, known_incorrect],[unknown_correct, unknown_incorrect]]")
            known = model_entry[0]
            unknown = model_entry[1]
            if not (isinstance(known, list) and len(known) == 2 and isinstance(unknown, list) and len(unknown) == 2):
                raise ValueError(f"Known/Unknown entries must be lists of two numbers at plot {p_idx}, model {m_idx}")

            model_name = model_names[m_idx] if model_names and m_idx < len(model_names) else f"Model {m_idx+1}"
            rows.append({"model": model_name, "category": "Known", "pred_correct": known[0], "pred_incorrect": known[1]})
            rows.append({"model": model_name, "category": "Unknown", "pred_correct": unknown[0], "pred_incorrect": unknown[1]})

        plots.append(pd.DataFrame(rows))

    return plots


def plot_grid_stacked_bars(plots, out_path=None, figsize=(14, 8), model_names=None, plot_titles=None):
    """Create the requested 2x3 grid plot from `plots` (list of DataFrames).

    plots: list of DataFrames (length 6)
    Each DataFrame: columns ['model','category','pred_correct','pred_incorrect']
    """
    sns.set(style="whitegrid")
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = axes.reshape(nrows, ncols)

    # Use blue for Known bars and orange for Unknown bars (each with two shades for
    # the stacked components: pred_correct (darker), pred_incorrect (lighter)).
    known_colors = ("#2b6cb0", "#9ec5ff")   # dark blue, light blue
    unknown_colors = ("#d97706", "#ffd7a6") # dark orange, light orange

    n_categories = 2  # Known, Unknown
    bar_width = 0.28
    inner_gap = 0.05  # gap between Known and Unknown within a model set
    group_gap = 0.45  # extra gap between model sets

    # Create legend handles for stacked components (pred_correct/pred_incorrect)
    from matplotlib.patches import Patch

    # Legend mapping: known correct/incorrect and unknown correct/incorrect
    legend_handles = [
        Patch(facecolor=known_colors[0], edgecolor="black", label="Known — \n predicted as Known"),
        Patch(facecolor=known_colors[1], edgecolor="black", label="Unknown — \n predicted Known"),
        Patch(facecolor=unknown_colors[0], edgecolor="black", label="Unknown — \n predicted Unknown"),
        Patch(facecolor=unknown_colors[1], edgecolor="black", label="Unknown — \n predicted Known"),
    ]

    # allow predefined titles and model name ordering
    if model_names is None:
        model_names = None  # preserve ordering from each DataFrame
    if plot_titles is None:
        plot_titles = DEFAULT_PLOT_TITLES

    # compute a global y-limit so all subplots share the same scale
    # find max stacked height across all plots
    global_max = 0.0
    for df in plots:
        stacked = (df['pred_correct'] + df['pred_incorrect']).max()
        if stacked > global_max:
            global_max = stacked
    if global_max <= 0:
        global_max = 100.0
    y_max = global_max * 1.08

    for idx, df in enumerate(plots):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        # compute x positions for groups
        # We'll layout groups as: for model i: x_base = i * (n_categories*bar_width + inner_gap + group_gap)
        group_width = n_categories * bar_width + inner_gap
        x_positions = []
        labels = []
        # determine model ordering: either provided model_names or order from the dataframe
        if model_names:
            models_in_plot = [m for m in model_names if m in list(df['model'].unique())]
        else:
            models_in_plot = list(df['model'].unique())
        for i, model in enumerate(models_in_plot):
            x_base = i * (group_width + group_gap)
            # Known bar center
            x_known = x_base
            x_unknown = x_base + bar_width + inner_gap
            x_positions.append((x_known, x_unknown))
            labels.append(model)

        # Plot each model group
        for i, (x_known, x_unknown) in enumerate(x_positions):
            model = models_in_plot[i]
            row_known = df[(df['model'] == model) & (df['category'] == 'Known')].iloc[0]
            row_unknown = df[(df['model'] == model) & (df['category'] == 'Unknown')].iloc[0]

            # Known uses blue shades, Unknown uses orange shades
            k_correct, k_incorrect = known_colors
            u_correct, u_incorrect = unknown_colors

            # Known bar (bottom: pred_correct, top: pred_incorrect)
            ax.bar(x_known, row_known['pred_correct'], width=bar_width, color=k_correct,
                   edgecolor='black')
            ax.bar(x_known, row_known['pred_incorrect'], width=bar_width, bottom=row_known['pred_correct'],
                   color=k_incorrect, edgecolor='black')

            # Unknown bar (use orange shades)
            ax.bar(x_unknown, row_unknown['pred_correct'], width=bar_width, color=u_correct,
                   edgecolor='black')
            ax.bar(x_unknown, row_unknown['pred_incorrect'], width=bar_width, bottom=row_unknown['pred_correct'],
                   color=u_incorrect, edgecolor='black')

        # Set x ticks to be centered across the model group (between known and unknown)
        group_centers = [(xk + xu) / 2.0 for (xk, xu) in x_positions]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(labels, rotation=0, fontsize=13)

        # give some padding on left/right
        ax.set_xlim(-group_width * 0.6, x_positions[-1][1] + group_width * 0.6)

        # set consistent y-limits across all subplots
        ax.set_ylim(0, y_max)

        # Only the left-most column shows y tick labels; keep ticks shared across rows
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        if c == 0:
            # ensure a reasonable number of ticks and format as percentages
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
            ax.tick_params(axis='y', labelsize=13)
            ax.set_ylabel('Percentage', fontsize=16)
        else:
            # keep tick positions but hide labels to preserve shared scale
            ax.tick_params(labelleft=False)

        # use provided plot_titles if available
        title = plot_titles[idx] if idx < len(plot_titles) else f"Plot {idx + 1}"
        # nudge the title upward so it doesn't get clipped by tight_layout
        ax.set_title(title, y=1.02, fontsize=20)

    # Place a shared legend above the subplot grid
    # Leave room at the top for the legend, then place it centered above plots
    fig.subplots_adjust(top=0.82)
    # Use multiple columns so legend fits horizontally; smaller font to save space
    lg = fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.99),
                    fontsize=15, ncol=4)
    # keep legend frame off for a cleaner look when on top
    lg.set_frame_on(False)

    # Tighten layout for the plotting area below the legend
    plt.tight_layout(rect=(0, 0, 1, 0.9))

    if out_path:
        fig.savefig(out_path, dpi=200)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


def main():
    plots = build_plots_from_array(DEFAULT_DATA_ARRAY, model_names=DEFAULT_MODEL_NAMES)
    out_path = "grid_stacked_bars.pdf"
    plot_grid_stacked_bars(plots, out_path=out_path)


if __name__ == '__main__':
    main()
