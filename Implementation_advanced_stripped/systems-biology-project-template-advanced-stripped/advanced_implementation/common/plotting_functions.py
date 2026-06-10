import json
import os
import re
import tkinter as tk
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter

from common.simulation import simulate_model, simulate_model_steady_state

FIG_WIDTH_A4_CM = 19.0
FIG_HEIGHT_A4_CM = (19/21)*29.7
LINE_WIDTH = 2
FONT_SIZE_TITLE = 14
FONT_SIZE_AXIS = 10
FONT_SIZE_LEGEND = 8
FONT_SIZE_ANNOTATION = 12

INPUT_RECTANGLE_HEIGHT = 0.01

rcParams['font.family'] = 'sans-serif'

ANNOTATIONS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# utility function to find the rows/columns that are the closest to a square
def close_to_square(n: int) -> tuple[int, int]:
    b = np.round(np.sqrt(n))
    a = np.ceil(n/b)
    return int(a), int(b)


# utility function to find the rows/columns that are the closest to a given aspect ratio
def close_to_aspect_ratio(n: int, width: float, height: float) -> tuple[int, int]:
    # if h = w*height/width, then n = w*h = w*w*height/width => w = sqrt(n*width/height)
    w = np.round(np.sqrt(n*width/height))
    h = np.ceil(n/w)

    return int(h), int(w)


# utility function to find the size of your monitor - if multi-screen setup the primary screen is considered
def get_curr_screen_geometry() -> tuple[float, float, float, float]:
    """
    Workaround to get the size of the current screen in a multi-screen setup.

    Returns:
        geometry (str): The standard Tk geometry string.
            [width]x[height]+[left]+[top]
    """
    root = tk.Tk()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    geometry = root.winfo_geometry()
    root.destroy()
    
    width, height, x0, y0 = [x/10 for x in map(int, re.findall(r'\d+', geometry))] # /10 converts units from mm to cm
    
    return width, height, x0, y0


# function to create and build the figure object by adding the axes defined in data file
def create_and_load_axis_object(
    figs: dict, fig_num: int, plot_info: dict, 
    fig_width: float = FIG_WIDTH_A4_CM, fig_height: float = FIG_HEIGHT_A4_CM) -> Axes:
    # create one figure per fig-number, with one subplot per observable panel
    if fig_num not in figs:
        figs[fig_num] = {"fig": plt.figure(
            # (1/2.54) scales the argument into inches (default length unit in matplotlib)
            figsize=(fig_width*(1/2.54), fig_height*(1/2.54))), "axs": {}}

    if plot_info["subplot_idx"] not in figs[fig_num]["axs"]:
        rows, cols = plot_info["shape"]
        figs[fig_num]["axs"][plot_info["subplot_idx"]] = figs[fig_num]["fig"].add_subplot(
            rows, cols, plot_info["subplot_idx"])

    return figs[fig_num]["axs"][plot_info["subplot_idx"]]


# convert 0-255 rgb values to 0-1 scale
def rgb_255_to_float(c: Sequence[float] | str | None) -> tuple[float, float, float] | str | None:
    if c is None or c == 'None':
        return 'none'
    c_values: Sequence[float] = c  # type: ignore[assignment]
    return (float(c_values[0])/255.0, float(c_values[1])/255.0, float(c_values[2])/255.0)


# add subplot index
def set_annotation(ax: Axes, annotation_id: int, x: float = 0.02, y: float = 1.0, fs: int = FONT_SIZE_ANNOTATION) -> None:
    ax.text(
        x, y, ANNOTATIONS[annotation_id], fontsize=fs,
        fontweight='bold', transform=ax.transAxes, va='top', ha='left')


# set the axis information according to the data file
def set_axis_information(
    ax: Axes, plot_info: dict, obs: dict, c_plot: tuple[float, float, float] | None = (0, 0, 0), 
    xlim: tuple[float, float] | None = None) -> None:
    if "Points" in obs:
        for _, ys in enumerate(obs["Points"]):
            ax.plot(obs["Time"], ys, marker="x", markersize=6,
                    linestyle='none', color=c_plot, lw=LINE_WIDTH)
    if "title" in plot_info:
        ax.set_title(plot_info["title"], fontsize=FONT_SIZE_TITLE)
    if "xlabel" in plot_info:
        ax.set_xlabel(plot_info["xlabel"], fontsize=FONT_SIZE_AXIS)
    if "ylabel" in plot_info:
        ax.set_ylabel(plot_info["ylabel"], fontsize=FONT_SIZE_AXIS)

    if "ylim" in plot_info:
        ax.set_ylim(plot_info["ylim"])
    else:
        highest_y = ax.get_ylim()[1]
        decimal = int(np.floor(np.log10(highest_y)))
        precision = 10**(decimal - 1)
        ymax = np.ceil(highest_y / precision) * precision
        ax.set_ylim((0, ymax))

    if xlim is not None:
        ax.set_xlim(xlim)
    elif "xlim" in plot_info:
        ax.set_xlim(plot_info["xlim"])

    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_AXIS)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# function to set tick precision based on the values
def custom_axis_precision(ax: Axes, values: list[float] | np.ndarray, axis: str = 'y') -> None:
    # check if there are any value with more than 0 decimal places
    precision_0_decimal_place = [bool(val == int(val)) for val in values]
    # check if there are any value with more than 1 decimal places
    precision_1_decimal_place = [
        bool(round(val, 2) == round(val, 1)) for val in values]

    if axis == 'y':
        if sum(precision_0_decimal_place) == len(values):
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        elif sum(precision_1_decimal_place) == len(values):
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        if sum(precision_0_decimal_place) == len(values):
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        elif sum(precision_1_decimal_place) == len(values):
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        else:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))


# round up ymax to get "nice" steps on the axis
def return_closest_dividable_number(min: float, max: float, num_values: int = 5) -> float:
    # step size
    number_of_steps = num_values - 1
    step = (max-min) / number_of_steps

    if step < 1:
        precision_value_number = 5
        corrector_value_numbers = 10**(abs(np.floor(np.log10(step)))+1)
    elif step < 10:
        precision_value_number = 10
        corrector_value_numbers = 10**(1 - np.floor(np.log10(step)))
    else:
        precision_value_number = 10
        corrector_value_numbers = 10**(2 - np.floor(np.log10(step)))

    # round up the step value using ceil function and the chosen precision value - the the step is < 1 use the corrector value to assign it correctly
    rounded_step = np.ceil(step*corrector_value_numbers / precision_value_number) * \
        precision_value_number / corrector_value_numbers

    # return the ymax value
    return min + rounded_step * number_of_steps


# set "num" numbers of y-ticks
def set_num_yticks(ax: Axes, num: int = 5) -> None:
    ymin, ymax = ax.get_ylim()
    ymax_updated = return_closest_dividable_number(ymin, ymax, num)

    # set axis ticks
    y_ticks = np.linspace(ymin, ymax_updated, num)
    ax.set_yticks(y_ticks)
    custom_axis_precision(ax, y_ticks, axis='y')


# set "num" numbers of x-ticks
def set_num_xticks(ax: Axes, num: int = 4) -> None:
    xmin, xmax = ax.get_xlim()
    xmax_updated = return_closest_dividable_number(xmin, xmax, num)

    # set axis ticks
    x_ticks = np.linspace(xmin, xmax_updated, num)
    ax.set_xticks(x_ticks)
    custom_axis_precision(ax, x_ticks, axis='x')


# add rectangles indicating input period
def add_input_patches(ax: Axes, input_values: dict) -> None:
    y0 = ax.get_ylim()[0]
    y1 = (ax.get_ylim()[1]-ax.get_ylim()[0])*INPUT_RECTANGLE_HEIGHT
    for t_id in range(1, len(input_values["t"]), 2):
        x0 = input_values["t"][t_id]
        x1 = input_values["t"][t_id+1] - input_values["t"][t_id]
        ax.add_patch(Rectangle((x0, y0), x1, y1, color='k', zorder=10))


# add lines where state manipulations are applied
def add_input_lines(ax: Axes, input_values: dict) -> None:
    y0 = ax.get_ylim()[0]
    y1 = ax.get_ylim()[1]*0.9
    for t_id in range(0, len(input_values["t"]), 1):
        x = input_values["t"][t_id]
        ax.plot([x, x], [y0, y1], color='grey', linestyle=(0,(5,10)), lw=1, zorder=10)


# Decide based on the input dictionary whether to add rectangles or lines to indicate the stimulus period and call the respective function
def show_stimulus_indicator(ax: Axes, input_dict: dict, fig_num: int, stim_type_in_figure: dict) -> None:
    for input_name, input_values in input_dict["input"].items():
        # show rectangles for inputs that requested it
        if "show_in_plot" in input_values.keys() and input_values["show_in_plot"] and "type" in input_values:
            add_input_patches(ax, input_values)
            if "type" in input_values:
                stim_type_in_figure[fig_num].append("type")
        elif "show_in_plot" in input_values.keys() and input_values["show_in_plot"] and "mode" in input_values:
            add_input_lines(ax, input_values)
            if "mode" in input_values:
                stim_type_in_figure[fig_num].append("mode")

        # special-case: always annotate 'intensity' input with a thin dashed line and label
        if input_name.lower() == "intensity" and "t" in input_values and "f" in input_values:
            try:
                add_intensity_lines(ax, input_values)
            except Exception:
                # don't break plotting on annotation errors
                continue


def add_intensity_lines(ax: Axes, input_values: dict) -> None:
    """
    Draw thin dashed horizontal lines above the plot for intensity intervals and
    label them with the intensity value.

    Expects input_values to contain 't' (list of time breakpoints, possibly
    starting with -inf) and 'f' (list of intensity values). The segment
    between t[i] and t[i+1] uses f[i]. For the final segment (after last t),
    end is taken from ax.get_xlim()[1].
    """
    times = list(input_values.get("t", []))
    vals = list(input_values.get("f", []))
    if len(times) < 2 or len(vals) == 0:
        return

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_range = x_max - x_min if (x_max - x_min) != 0 else 1.0
    dx = 0.01 * x_range
    y_text = y_max - 0.01 * (y_max - y_min)

    # draw a vertical dashed line at each change time (skip the -inf first entry)
    for i in range(1, len(times)):
        tchg = times[i]
        try:
            tval = float(tchg)
        except Exception:
            continue

        # skip if outside x-limits
        if tval < x_min or tval > x_max:
            continue

        intensity = vals[i] if i < len(vals) else vals[-1]
        try:
            intensity_val = float(intensity)
        except Exception:
            continue

        # draw vertical dashed line
        ax.vlines(tval, y_min, y_max, colors='grey', linestyles='--', linewidth=0.7, alpha=0.8)

        # place label slightly to the right of the line
        label_x = min(tval + dx, x_max)
        label = f"{intensity_val:g}"
        ax.text(label_x, y_text, label, ha='left', va='top', fontsize=7, color='grey')


# use an empty subplot slot to add the legends - this is needed to avoid interaction of the legend with the main plot when clicking on the legend
def empty_axis(ax: Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Remove ticks
    ax.tick_params(left=False, bottom=False)  # Remove tick marks on axes
    ax.set_xticks([])  # Remove x-axis tick labels
    ax.set_yticks([])  # Remove y-axis tick labels

    # remove the background
    ax.set_facecolor("none")
    
    # disable navigation for the legend axis to avoid interaction with the main plot when clicking on the legend
    ax.set_navigate(False)


# use an empty subplot slot to add the legends
def create_legend_empty_axis(
    ax: Axes, sim_labels: list[str] = ['default'], c_lines: list[list[float]] = [[0.0, 0.0, 0.0]], 
    c_errors: list[list[float]] = [[0.0, 0.0, 0.0]], c_individuals: list[list[float]] = [[0.0, 0.0, 0.0]], 
    uncertainties: bool = False, show_stimuli_patch: bool = False, show_stimuli_lines: bool = False) -> None:
    empty_axis(ax)
    
    for c_line, sim_label in zip(c_lines, sim_labels):
        if c_line != []:
            ax.plot([], [], color=rgb_255_to_float(c_line), linestyle=None, label=sim_label)
            if uncertainties:
                ax.fill_between(
                    [], [], [], color=rgb_255_to_float(c_line), alpha=0.3, label=f"{sim_label} uncertainty")
    for c_err, sim_label in zip(c_errors, sim_labels):
        if c_err != []:
            ax.errorbar(
                [], [], yerr=[], linestyle='', marker='o', markersize=4, capsize=6, 
                color=rgb_255_to_float(c_err), label=f"{sim_label} data")
    for c_marker, sim_label in zip(c_individuals, sim_labels):
        if c_marker != []:
            ax.plot(
                [], [], marker="x", markersize=4, linestyle='none', color=rgb_255_to_float(c_marker), 
                label=f"{sim_label} points")
    
    if show_stimuli_patch:
        ax.fill_between([], [], [], color=[0, 0, 0], label="Stimulus duration")
    if show_stimuli_lines:
        ax.plot([], [], color='grey', linestyle='--', label="Stimulus onset")


# load all available uncertainty sources for *model_name* and merge them into a single union-envelope dict
def load_union_UC(model_name: str, time_vector: np.ndarray) -> dict:
    """
    Load all available uncertainty sources for *model_name* and merge them into
    a single union-envelope dict keyed by experiment -> observable ->
    {"Min": array, "Max": array} on the supplied *time_vector*.

    Returns an empty dict if no UC files are found.
    """
    sources: list[tuple[str, str, str]] = [
        (f"results/{model_name}/UC/UC_reverse_PPL_{model_name}.json", "Min", "Max"),
        (f"results/{model_name}/UC/UC_PPL_{model_name}.json",         "Min", "Max"),
        (f"results/{model_name}/UC/UC_MCMC_{model_name}.json",        "Lower", "Upper"),
    ]

    # accumulate per (exp, obs) lists of (lower, upper) arrays
    lower_acc: dict[str, dict[str, list[np.ndarray]]] = {}
    upper_acc: dict[str, dict[str, list[np.ndarray]]] = {}

    for path, lower_key, upper_key in sources:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as fh:
            data = json.load(fh)
        for k_exp, obs_dict in data.items():
            for k_obs, vals in obs_dict.items():
                lower_acc.setdefault(k_exp, {}).setdefault(k_obs, []).append(
                    np.interp(time_vector, vals["Time"], vals[lower_key]))
                upper_acc.setdefault(k_exp, {}).setdefault(k_obs, []).append(
                    np.interp(time_vector, vals["Time"], vals[upper_key]))

    union_UC: dict = {}
    for k_exp in lower_acc:
        union_UC[k_exp] = {}
        for k_obs in lower_acc[k_exp]:
            union_UC[k_exp][k_obs] = {
                "Min": np.minimum.reduce(lower_acc[k_exp][k_obs]),
                "Max": np.maximum.reduce(upper_acc[k_exp][k_obs]),
            }
    return union_UC


# Plot figures based on the contents of the data file
def plot_figures(p: list[float] | np.ndarray, sims: dict, D: dict, model, fig_names: dict, simulate_steady_state: bool = False) -> None:

    p_array = np.array(p) if isinstance(p, list) else p

    figs = {}
    stim_type_in_figure = {}
    for k_exp, d in D.items():
        ic = model.state_values
        
        # High resolution time vector for smooth simulation curves
        t_start = 0
        t_end = d["all_times"][-1]
        num_steps = int((t_end - t_start) / 0.1) + 1
        t_high_res = np.linspace(t_start, t_end, num_steps)

        union_UC = load_union_UC(model.name, t_high_res)

        sim = sims[k_exp]
        sim.reset_states()
        if simulate_steady_state:
            simulate_model_steady_state(sim, ic, p_array)
            ic = sim.state_values

        simulate_model(sim, ic, p_array, time_vector=t_high_res)

        for k_obs, obs in d["Observables"].items():
            plot_info = obs["plotting_info"]
            fig_num = plot_info["fig"]
            if fig_num not in stim_type_in_figure:
                stim_type_in_figure[fig_num] = []

            if fig_num in fig_names:
                if "plot_colors" in plot_info:
                    c_plot, c_eb, c_fill = rgb_255_to_float(plot_info["plot_colors"][0]), rgb_255_to_float(
                        plot_info["plot_colors"][1]), rgb_255_to_float(plot_info["plot_colors"][2])
                else:
                    # Matplotlib default blue and orange colors (normalized to 0-1)
                    c_plot = (0.122, 0.467, 0.706)
                    c_eb   = (1.0, 0.498, 0.055)
                    c_fill = (1.0, 0.498, 0.055)

                idx = sim.feature_names.index(k_obs)
                y_sim = sim.feature_data[:, idx]

                # create the figure and axis object - then returns the axis object to use
                ax = create_and_load_axis_object(
                    figs, fig_num, plot_info, fig_width=FIG_WIDTH_A4_CM, fig_height=FIG_HEIGHT_A4_CM)

                # do the plotting of uncertainty, best simulation, and errorbar
                if union_UC and k_exp in union_UC and k_obs in union_UC[k_exp]:
                    ax.fill_between(
                        t_high_res,
                        union_UC[k_exp][k_obs]["Min"],
                        union_UC[k_exp][k_obs]["Max"],
                        alpha=0.3, color=c_plot)
                ax.plot(sim.time_vector, y_sim, color=c_plot, label=k_exp, lw=LINE_WIDTH)

                # shaded band for mean +/- SEM (light version of plot color)
                try:
                    mean = np.array(obs['Mean'])
                    sem = np.array(obs['SEM'])
                    lower = mean - sem
                    upper = mean + sem
                    # use a consistent blue for the SEM shaded band
                    ax.fill_between(obs['Time'], lower, upper, color=(0.122, 0.467, 0.706), alpha=0.25, zorder=3)
                except Exception:
                    pass

                # open circle markers for data points (edge blue, no fill)
                try:
                    ax.plot(obs['Time'], obs['Mean'], linestyle='-', color='blue', linewidth=1.0, zorder=5)
                    ax.plot(obs['Time'], obs['Mean'], linestyle='none', marker='o', markersize=6,
                            markerfacecolor='none', markeredgecolor='blue', zorder=6)
                except Exception:
                    pass

                # use the utility functions to format the axis and add rectangles to indicate input period
                set_axis_information(ax, plot_info, obs, c_plot)
                show_stimulus_indicator(ax, d, fig_num, stim_type_in_figure)
                set_num_yticks(ax, num=5)
                set_num_xticks(ax, num=4)

    for fig_num, figure in figs.items():
        for annotation_counter, pos in enumerate(sorted(figure["axs"].keys())):
            set_annotation(figure["axs"][pos], annotation_counter)

        # adjust the layout
        fig = figure["fig"]
        fig.tight_layout()

        if fig_num in fig_names:
            fig.savefig(f"./figures/{model.name}/{fig_names[fig_num]}.png", dpi=300)
        else:
            print(
                f"Not saving figure as figure number {fig_num} was not provided in input 'fig_names'")


def plot_PI_waterfall(model_name: str) -> None:
    """
    Read PI result JSON files from results_PI/{model_name}/ and plot a
    convergence waterfall showing how the min/max parameter bounds tighten
    over successive optimization runs.

    One subplot per parameter:
    - X-axis: run index (chronological by timestamp extracted from filename)
    - Y-axis: parameter value (log scale)
    - Blue line: running minimum (lower identifiability bound converging down)
    - Red line: running maximum (upper identifiability bound converging up)

    Saved to figures/{model_name}/PI_waterfall.png.
    """
    folder = f"./results_PI/{model_name}"
    if not os.path.exists(folder):
        print("No PI results found — skipping waterfall plot.")
        return

    prefix = f"{model_name}-"
    # group {param_name: [(timestamp_str, f_value), ...]}
    groups: dict[str, list[tuple[str, float]]] = {}

    for fname in os.listdir(folder):
        if not fname.endswith(".json") or fname == "PI_summary.json":
            continue
        rest = fname[len(prefix):]
        param_name = rest.split("-")[0]
        if not param_name:
            continue
        # extract timestamp if present (format: YYYYMMDD-HHMMSS at end of stem)
        stem = fname[:-5]
        ts_match = re.search(r'(\d{8}-\d{6})$', stem)
        ts = ts_match.group(1) if ts_match else "00000000-000000"
        try:
            with open(os.path.join(folder, fname)) as fh:
                data = json.load(fh)
            if "f" in data:
                groups.setdefault(param_name, []).append((ts, float(data["f"])))
        except (json.JSONDecodeError, KeyError):
            continue

    if not groups:
        print("No valid PI result files found — skipping waterfall plot.")
        return

    # sort each group chronologically
    for param_name in groups:
        groups[param_name].sort(key=lambda t: t[0])

    n = len(groups)
    n_rows, n_cols = close_to_square(n)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5), squeeze=False)
    fig.suptitle(f"PI Convergence — {model_name}", fontsize=FONT_SIZE_TITLE + 2)

    c_min = (0.122, 0.467, 0.706)   # blue
    c_max = (0.839, 0.153, 0.157)   # red

    for ax_idx, (param_name, runs) in enumerate(sorted(groups.items())):
        row, col = divmod(ax_idx, n_cols)
        ax = axes[row][col]

        f_vals = [r[1] for r in runs]
        xs = list(range(1, len(f_vals) + 1))

        # running min and max
        running_min = np.minimum.accumulate(f_vals)
        running_max = np.maximum.accumulate(f_vals)

        ax.step(xs, running_min, where='post', color=c_min, lw=LINE_WIDTH, label='running min')
        ax.step(xs, running_max, where='post', color=c_max, lw=LINE_WIDTH, label='running max')
        ax.scatter(xs, f_vals, color='gray', s=12, zorder=3, alpha=0.6)

        ax.set_yscale('log')
        ax.set_title(param_name, fontsize=FONT_SIZE_TITLE)
        ax.set_xlabel("Run index", fontsize=FONT_SIZE_AXIS)
        ax.set_ylabel(param_name, fontsize=FONT_SIZE_AXIS)
        ax.tick_params(labelsize=FONT_SIZE_AXIS)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # hide unused subplots
    for ax_idx in range(n, n_rows * n_cols):
        row, col = divmod(ax_idx, n_cols)
        axes[row][col].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=FONT_SIZE_LEGEND, frameon=False)
    fig.tight_layout()

    out_path = f"./figures/{model_name}/PI_waterfall.png"
    fig.savefig(out_path, dpi=300)
    print(f"PI waterfall plot saved to {out_path}")
