"""style.py
Contains standard style for figures.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from matplotlib import rcParams
try:
    import seaborn as sns
    sns.set(style='ticks', palette='Set1') 
except:
    print('seaborn not installed')

save_fig    = False
plot_style  = ["pdf", "print", "presentation", "poster"][0]

# Figure size
# Only specify height in inch. Width is calculated from golden ratio.
height = 3.4   # inch
linewidth  = 1.0
cross_size = 9 # pt, size of cross markers

# Choose parameters for pdf or print
if plot_style == "pdf":
    figure_path = os.path.join(".", "figures")
#     axes_color = "#636363" 
#     axes_color = "#bdbdbd" 
    axes_color = "#959595" 
#     text_color = "#636363"
    text_color = "#363636"
    font_family = 'serif'
    linewidth  = 2
elif plot_style == "print":
    figure_path = os.path.join(".", "figures")
    axes_color = "#959595" 
    text_color = "#363636"
    # Font
    latex_preamble      = [r'\usepackage[T1,small,euler-digits]{eulervm}']
    font_family         = 'serif'
elif plot_style == "presentation":
    figure_path = os.path.join("..", "presentation", "figures")
    axes_color = "#959595" 
    text_color = "#363636"
    # Font
    latex_preamble      = [r'\usepackage[cmbright]{sfmath}']
    rcParams['text.latex.preamble'] = latex_preamble
    font_family         = 'sans-serif'
elif plot_style == "poster":
    figure_path = os.path.join("..", "poster", "figures") 
    axes_color = "#959595" 
    text_color = "#363636"
    # Font
    latex_preamble      = [r'\usepackage[cmbright]{sfmath}']
    rcParams['text.latex.preamble'] = latex_preamble
    font_family         = 'sans-serif'
    # Figure size
    #        cm  / 2.54 cm * 1.0 inch  
    height = 9.5 / 2.54     # inch
    linewidth  = 0.8
    cross_size = 9 # pt, size of cross markers

"""
if not save_fig:
    height = 8.
    linewidth  = 2.0
    cross_size = 12 # pt, size of cross markers
"""

# Figure size
from  scipy.constants import golden_ratio as gr
figsize  = (gr * height, 1. * height)

fontsize_labels         = 12    # pt, size used in latex document
fontsize_labels_axes    = fontsize_labels
fontsize_labels_title   = fontsize_labels
fontsize_plotlabel      = fontsize_labels       # for labeling plots with 'A', 'B', etc.
legend_ms  = 2  # scale of markers in legend

# # Use sans-serif with latex in matplotlib
# # https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlibhttps://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
# rcParams['text.latex.preamble'] = [
#     r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#     r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
#     r'\usepackage{helvet}',    # set the normal font here
#     r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#     r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
# ]

# Adapt the matplotlib.rc
rcParams['font.family']         = font_family
# rcParams['font.serif']          = 'Computer Modern'
rcParams['text.usetex']         = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
rcParams['figure.figsize']      = figsize
rcParams['font.weight']         = "light"
rcParams['font.size']           = fontsize_labels
rcParams['xtick.labelsize']     = fontsize_labels
rcParams['ytick.labelsize']     = fontsize_labels
rcParams['legend.fontsize']     = fontsize_labels
rcParams['axes.labelsize']      = fontsize_labels_axes
rcParams['axes.titlesize']      = fontsize_labels_title
rcParams['legend.markerscale']  = legend_ms
rcParams['text.color']          = text_color
rcParams['xtick.color']         = text_color
rcParams['ytick.color']         = text_color
rcParams['axes.labelcolor']     = text_color
rcParams['axes.edgecolor']      = axes_color
rcParams['axes.grid']           = False
rcParams['lines.linewidth']     = linewidth
rcParams['lines.markersize']    = 3
# rcParams['figure.autolayout']   = True # this disables tight layout!

tick_params = {
                ## TICKS
                # see http://matplotlib.org/api/axis_api.html#matplotlib.axis.Tick
                'xtick.major.size'     : 3,      # major tick size in points
                'xtick.minor.size'     : 2,      # minor tick size in points
                'xtick.major.width'    : 0.5,    # major tick width in points
                'xtick.minor.width'    : 0.5,    # minor tick width in points
                'xtick.major.pad'      : 4,      # distance to major tick label in points
                'xtick.minor.pad'      : 4,      # distance to the minor tick label in points
                'xtick.direction'      : 'out',    # direction: in, out, or inout
                
                'ytick.major.size'     : 3,      # major tick size in points
                'ytick.minor.size'     : 2,      # minor tick size in points
                'ytick.major.width'    : 0.5,    # major tick width in points
                'ytick.minor.width'    : 0.5,    # minor tick width in points
                'ytick.major.pad'      : 4,      # distance to major tick label in points
                'ytick.minor.pad'      : 4,      # distance to the minor tick label in points
                'ytick.direction'      : 'out'    # direction: in, out, or inout
                }
rcParams.update(tick_params)


# Convenience functions
def tick_manager(ax, max_tick_num=[5, 4]):
    """ Reduce the number of ticks to a bare minimum. """
    for max_n_ticks, ticks, lim, tick_setter in zip(
        max_tick_num,
        [ax.get_xticks(), ax.get_yticks()],
        [ax.get_xlim(), ax.get_ylim()],
        [lambda t: ax.set_xticks(t), lambda t: ax.set_yticks(t)]):
        # Reduce ticks to those anyways visible on the lower side
        ticks = ticks[(ticks >= lim[0]) * (ticks <= lim[1])]
        
        n_ticks = len(ticks)
        while n_ticks > max_n_ticks:
            ticks = ticks[::2]
            n_ticks = len(ticks)
        # Set new ticks
        tick_setter(ticks)
    return None
#     for max_n_ticks, labels, label_vals, lower_lim in zip(
#         max_tick_num,
#         [ax.get_xticklabels(), ax.get_yticklabels()],
#         [ax.get_xticks(), ax.get_yticks()],
#         [ax.get_xlim()[0], ax.get_ylim()[0]]):
        
# #         ### Sometimes buggy...
# #         # Reduce ticks to those anyways visible on the lower side
# #         min_idx = np.where(label_vals >= lower_lim)[0].min()
# #         labels = labels[min_idx:]
        
#         n_ticks = len(labels)
#         print(n_ticks)
#         print(label_vals, lower_lim)
#         for k, label in enumerate(labels):
#             print(label)
#             if k % 2 == 1:
#                 label.set_visible(False)
#                 n_ticks -= 1 

def fixticks(fig_or_ax, fix_spines=True, max_tick_num=[5, 4], manage_ticks=True):
    """ Polishes graphs.
    Input: figure, list of axes or single axes. 
    """
    if type(fig_or_ax) is matplotlib.figure.Figure:  
        axes = fig_or_ax.axes
    elif type(fig_or_ax) is list:
        axes = fig_or_ax
    else:
        axes = [fig_or_ax]
    for ax in axes:
        ax.grid(False)      # Turn off grid (distracts!)
        # Set spines to color of axes
        for t in ax.xaxis.get_ticklines(): t.set_color(axes_color)
        for t in ax.yaxis.get_ticklines(): t.set_color(axes_color)
        if fix_spines:
            # Remove top axes & spines
            ax.spines['top'].set_visible(False)
            #ax.xaxis.set_ticks_position('bottom') # this resets spines in case of sharex=True...
            # Remove axes & spines on the side not used
            active_side = ax.yaxis.get_ticks_position()
            if active_side in ['default', 'left']:
                inactive_side = 'right'
            elif active_side == 'right':
                inactive_side = 'left'
            if active_side in ['default', 'left', 'right']:
                ax.spines[inactive_side].set_visible(False)
                ax.yaxis.set_ticks_position(active_side)
            # ax.tick_params(axis='x', colors=axes_color) ## Too light
            # ax.tick_params(axis='y', colors=axes_color)
            # Note: otherwise active_side == 'unknown'
        # Take care of tick number
        if manage_ticks:
            tick_manager(ax, max_tick_num)
    return None

def add_textbox(ax, s, rel_x=0.95, rel_y=0.95, zorder=10):
    """ Add a textbox. """
    textbox = ax.text(rel_x, rel_y, s, fontsize=12,
                      horizontalalignment='right', verticalalignment='top',
                      bbox=dict(facecolor='none', edgecolor=axes_color, boxstyle='round'),
                      zorder=zorder,
                      transform = ax.transAxes)
    return textbox

def add_subplot(fig, 
        n_rows_cols=(1, 1), index_row_col=(0, 0), 
        rowspan=1, colspan=1, 
        projection=None, 
        width_ratios=None, 
        sharex=None, sharey=None):
    """Add subplot specific to figure."""
    gridspec=plt.GridSpec(n_rows_cols[0], n_rows_cols[1], width_ratios=width_ratios)
    subplotspec=gridspec.new_subplotspec(index_row_col, rowspan=rowspan, colspan=colspan)
    ax = fig.add_subplot(subplotspec, projection=projection, sharex=sharex, sharey=sharey)
    return ax

def saving_fig(fig, figure_path, fig_name, data_type="png", verbose=True, dpi=1000, save_fig=True):
    # DPI = 1000 is a simple recommendation for publication plots
    from pathlib import Path
    if not Path(figure_path).is_dir():
        os.makedirs(figure_path)
        print("Made new directory ", figure_path)        
    if save_fig:
        if verbose:
            print("Save figure to " + os.path.join(figure_path, fig_name) + "." + data_type)
        if data_type == "png":
            fig.savefig(os.path.join(figure_path, fig_name + ".png"), 
                    dpi=dpi, 
                    bbox_inches='tight', format="png") 
        elif data_type == "pdf":
            fig.savefig(os.path.join(figure_path, fig_name + ".pdf"), 
                    dpi=dpi, 
                    bbox_inches='tight', format="pdf")
        elif data_type=="both":
            # Both
            fig.savefig(os.path.join(figure_path, fig_name + ".png"), 
                    dpi=dpi, 
                    bbox_inches='tight', format="png") 
            fig.savefig(os.path.join(figure_path, fig_name + ".pdf"), 
                    dpi=dpi, 
                    bbox_inches='tight', format="pdf")
        elif data_type == "svg":
            fig.savefig(os.path.join(figure_path, fig_name + "." + data_type),
                    format="svg")
        else:
            fig.savefig(os.path.join(figure_path, fig_name + "." + data_type),
                    format=data_type)
    else:
        if verbose:
            print("Not saved. Path would be: " + os.path.join(figure_path, fig_name) + "." + data_type)


# Colors are layered: two types a four layers
# Source: http://colorbrewer2.org/
# microcircuit colors:
colors =   [
            "#08519c", # blue
            "#a63603", # red/brown
            "#54278f", # dark purple
            "#006d2c", # green
            "#9ecae1", # light blue
            "#fdae6b", # light red/brown
            "#bcbddc", # light purple
            "#a1d99b"  # light green
            ]
# # structural plasticity colors:
# colors =   [
#             "#00028c", # blue
#             "#cc0000", # red
#             "#008000", # green
#             "#8b00aa", # dark purple
#             "#aa5d00", # brown
#             "#00c4bd", # turquoise
#             "#cc7722", # ochre
#             "#666666"  # grey
#             ]

### Tango colors
tango = np.array([
    [252, 233,  79], #Butter 1
    [237, 212,   0], #Butter 2
    [196, 160,   0], #Butter 3
    [252, 175,  62], #Orange 1
    [245, 121,   0], #Orange 2
    [206,  92,   0], #Orange 3
    [233, 185, 110], #Chocolate 1
    [193, 125,  17], #Chocolate 2
    [143,  89,   2], #Chocolate 3
    [138, 226,  52], #Chameleon 1
    [115, 210,  22], #Chameleon 2
    [ 78, 154,   6], #Chameleon 3
    [114, 159, 207], #Sky Blue 1
    [ 52, 101, 164], #Sky Blue 2
    [ 32,  74, 135], #Sky Blue 3
    [173, 127, 168], #Plum 1
    [117,  80, 123], #Plum 2
    [ 92,  53, 102], #Plum 3
    [239,  41,  41], #Scarlet Red 1
    [204,   0,   0], #Scarlet Red 2
    [164,   0,   0], #Scarlet Red 3
#     [238, 238, 236], #Aluminium 1
    [211, 215, 207], #Aluminium 2
#     [186, 189, 182], #Aluminium 3
    [136, 138, 133], #Aluminium 4
    [ 85,  87,  83], #Aluminium 5
#     [ 46,  52,  54], #Aluminium 6
]) / 255
# Sort:
# blue, orange, green, red, violet, butter, alum, choc
sorter = np.array([4, 1, 3, 6, 5, 0, 7, 2])

tango = tango.reshape((-1, 3, 3))[sorter].reshape((-1, 3))

tango_0 = tango[0::3]
tango_1 = tango[1::3]
tango_2 = tango[2::3]

colors = tango_2


def color_iterator(n_iter, cmap='viridis'):
    """ Continuous line coloring. 

    n_iter: int
        Number of color levels = iterations.
        
    cmap: str
        Color map. Check matplotlib for possible values. 
        default: 'viridis'
        
    Returns: 
    scalarMap:
        Generator for colors. Usage:
        ```
            from helper_funcs import color_iterator
            c_iter = color_iterator(n_iter)
            ax.plot(x, y, c=c_iter(idx))
        ```    
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mplcolors
    import matplotlib.cm as cmx
    cm          = plt.get_cmap(cmap) 
    cNorm       = mplcolors.Normalize(vmin=0, vmax=n_iter-1)
    scalarMap   = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    c_iter = lambda idx: scalarMap.to_rgba(idx)
    
    return c_iter


