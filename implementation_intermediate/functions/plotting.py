import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter

from functions.simulation import simulate_model, simulate_model_steady_state

LINE_WIDTH = 2
FONT_SIZE_TITLE = 10
FONT_SIZE_AXIS = 8
FONT_SIZE_LEGEND = 8
FONT_SIZE_ANNOTATION = 12

INPUT_RECTANGLE_HEIGHT = 0.01

rcParams['font.family'] = 'sans-serif'

ANNOTATIONS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# function to create and build the figure object by adding the axes defined in data file
def create_and_load_axis_object(figs, fig_num, plot_info, fig_width = 6.4, fig_height = 4.8):
    # if the figure is provided - create the figure and add the figure object to the figure dictionary
    if fig_num not in figs:
        figs[fig_num] = {"fig": plt.figure(
            figsize=(fig_width, fig_height)), "axs": {}}

    # if the axis object, i.e. subplot, is not created - create it and add it to the figure dictionary
    if plot_info["subplot_idx"] not in figs[fig_num]["axs"]:
        m, n = plot_info["shape"]
        figs[fig_num]["axs"][plot_info["subplot_idx"]] = figs[fig_num]["fig"].add_subplot(
            m, n, plot_info["subplot_idx"])

    # return the current axis object
    return figs[fig_num]["axs"][plot_info["subplot_idx"]]


# convert 0-255 rgb values to 0-1 scale
def rgb_255_to_float(c_values):
    if c_values is None or c_values == 'None':
        return None
    return (float(c_values[0])/255.0, float(c_values[1])/255.0, float(c_values[2])/255.0)


# add subplot index
def set_annotation(ax, annotation_id, x = 0.02, y = 1.02, fs = FONT_SIZE_ANNOTATION):
    ax.text(
        x, y, ANNOTATIONS[annotation_id], fontsize=fs,
        fontweight='bold', transform=ax.transAxes, va='top', ha='left')


# set the axis information according to the data file
def set_axis_information(ax, plot_info, obs, c_plot = (0, 0, 0)):
    if "Points" in obs:
        for i, ys in enumerate(obs["Points"]):
            label = 'Individual data points' if i == 0 else None
            ax.plot(obs["Time"], ys, marker="x", markersize=1,
                linestyle='none', color=c_plot, lw=LINE_WIDTH, label=label)
    if "title" in plot_info:
        ax.set_title(plot_info["title"], fontsize=FONT_SIZE_TITLE)
    if "xlabel" in plot_info:
        ax.set_xlabel(plot_info["xlabel"], fontsize=FONT_SIZE_AXIS)
    if "ylabel" in plot_info:
        ax.set_ylabel(plot_info["ylabel"], fontsize=FONT_SIZE_AXIS)
    if "ylim" in plot_info:
        ax.set_ylim(plot_info["ylim"])
    if "xlim" in plot_info:
        ax.set_xlim(plot_info["xlim"])

    # set font size on the x- and y-axis
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_AXIS)
    
    # set the x-axis to show integers and y-axis to show 2 decimal points
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # remove the top and right spines for better aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# add rectangles indicating input period
def add_input_patches(ax, input_dict):
    # get the y-axis limits to set the height of the rectangle
    y0 = ax.get_ylim()[0]
    y1 = (ax.get_ylim()[1]-ax.get_ylim()[0])*INPUT_RECTANGLE_HEIGHT
    
    # loop through the inputs 
    for _, input_values in input_dict["input"].items():
        if "show_in_plot" in input_values.keys() and input_values["show_in_plot"] and "mode" not in input_values:
            # add rectangles for every second time point in the input time vector (assuming that the input is a step function)
            for t_id in range(1, len(input_values["t"]), 2):
                x0 = input_values["t"][t_id]
                x1 = input_values["t"][t_id+1] - input_values["t"][t_id]
                # add the rectangle to the plot
                ax.add_patch(Rectangle((x0, y0), x1, y1, color='k'))


# add vertical dashed lines at intensity changes with annotations
def add_intensity_annotations(ax, input_dict):
    # Check if intensity input exists
    if "intensity" not in input_dict["input"]:
        return
    
    intensity_input = input_dict["input"]["intensity"]
    if intensity_input["type"].lower() != "piecewise_constant":
        return
    
    t_vals = intensity_input["t"]
    f_vals = intensity_input["f"]
    
    # Get axis limits for text positioning
    y_min, y_max = ax.get_ylim()
    
    # Plot vertical dashed lines at intensity change times
    # For piecewise_constant: at time t[i], the intensity changes to f[i]
    for i in range(1, len(t_vals)):
        t = t_vals[i]
        # Only plot times > 0 (skip -1000/-Infinity initial conditions)
        if t > 0 and np.isfinite(t):
            intensity_val = f_vals[i]
            
            # Draw vertical dashed line
            ax.axvline(x=t, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add text annotation at top of line showing intensity value
            y_pos = y_max * 0.95
            ax.text(t, y_pos, f'{int(intensity_val)}%',
                   fontsize=5, ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, 
                            edgecolor='gray', linewidth=0.5))


def plot_figures(p, sims, data, model, fig_names, simulate_steady_state = False):
    figs = {}
    #%% loop through the experiments and observables to plot the data and simulations
    for k_exp, d in data.items():
        ic = model.state_values
        # number of steps = (t_stop - t_start)/step_size + 1
        t_high_res = np.linspace(
            0, d["all_times"][-1], int((d["all_times"][-1]-0)/0.1 + 1))

        sim = sims[k_exp]
        sim.reset_states()
        if simulate_steady_state:
            simulate_model_steady_state(sim, ic, p)
            ic = sim.state_values

        simulate_model(sim, ic, p, time_vector=t_high_res)

        for k_obs, obs in d["Observables"].items():
            plot_info = obs["plotting_info"]
            fig_num = plot_info["fig"]

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
                ax = create_and_load_axis_object(figs, fig_num, plot_info)

                # plot the simulation model
                ax.plot(sim.time_vector, y_sim, color=c_plot, label=k_exp, lw=LINE_WIDTH)
                
                # plot data as a connected line with shaded uncertainty region
                ax.plot(obs['Time'], obs['Mean'], color=c_eb, linestyle='-',
                    marker=plot_info.get("marker", 'o'), markerfacecolor='none',
                    markeredgecolor=c_eb, markersize=5, label="Data")
                # add shaded uncertainty region around the data line
                ax.fill_between(obs['Time'], 
                               obs['Mean'] - obs['SEM'], 
                               obs['Mean'] + obs['SEM'],
                               alpha=0.2, color=c_eb)

                # use the utility functions to format the axis and add rectangles to indicate input period
                set_axis_information(ax, plot_info, obs, c_plot)
                add_input_patches(ax, d)
                add_intensity_annotations(ax, d)
                #ax.legend(fontsize=FONT_SIZE_LEGEND)

    #%% add annotations and save the figures
    for fig_num, figure in figs.items():
        for annotation_counter, pos in enumerate(sorted(figure["axs"].keys())):
            set_annotation(figure["axs"][pos], annotation_counter)

        # adjust the layout
        fig = figure["fig"]
        fig.tight_layout()

        # save the figure if the figure number is provided in the input 'fig_names'
        if fig_num in fig_names:
            fig.savefig(f"./figures/{fig_names[fig_num]}.png", dpi=300)
        else:
            print(
                f"Not saving figure as figure number {fig_num} was not provided in input 'fig_names'")
