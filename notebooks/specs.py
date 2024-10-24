### Define a few global variables
import os
import numpy as np

# Path for figures, data
figure_path = "../figures/"
data_path = "../data/"

# Random state
rng = np.random.RandomState()

### Plotting
# Fontsize
fs = 12
# Figure labels
flbs = ["(%s)"%s for s in "abcdefghijkl"] 
flbs = ["%s"%s for s in "ABCDEFGHIJKL"] 
# Color for legend
c_leg = '0.7'

### Tango colors
def def_tango():
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
    return colors, tango
colors, tango = def_tango()


# Colors for different g
# Colors
cs = tango.copy().reshape((-1, 3, 3))[:, ::-1]
# Blues
# cs[0, 0] = np.array([ 32,  74, 135]) / 255 # Sky Blue 3
cs[0, 0] = np.array([ 17,  54, 110]) / 255 # Sky Blue 3 -> darker
# cs[0, 1] = np.array([ 52, 101, 164]) / 255 # Sky Blue 2
cs[0, 1] = np.array([ 55, 108, 178]) / 255 # Sky Blue 2 -> lighter
cs[0, 2] = np.array([114, 159, 207]) / 255 # Sky Blue 1
# Oranges
# cs[1, 0] = np.array([206,  92,   0]) / 255 # Orange 3
cs[1, 0] = np.array([206,  73,   0]) / 255 # Orange 3 -> darker
# cs[1, 0] = np.array([156,  73,   0]) / 255 # Orange 3 -> pretty brown...
cs[1, 1] = np.array([245, 121,   0]) / 255 # Orange 2
cs[1, 2] = np.array([252, 175,  62]) / 255 # Orange 1
# Greens
# cs[2, 0] = np.array([ 78, 154,   6]) / 255 # Chameleon 3
cs[2, 0] = np.array([ 40,  80,   6]) / 255 # Chameleon 3 -> make a bit darker
# cs[2, 1] = np.array([115, 210,  22]) / 255 # Chameleon 2
cs[2, 1] = np.array([ 78, 154,   6]) / 255 # use Chameleon 1 instead
cs[2, 2] = np.array([138, 226,  52]) / 255 # Chameleon 1 

#######################################################
# Specific colors for oblique
# Colors for the different initializations (scenarios)
cs_sce = cs[[0, 1, 0, 1], [2, 2, 0, 0]]
alpha_sce = np.array([0.8, 0.8, 1, 1])
# Colors for cycling contexts
cs_ctx = colors[[4, 2]]



def color_iterator(n_iter, cmap='viridis', vmin=0):
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
    cNorm       = mplcolors.Normalize(vmin=vmin, vmax=n_iter-1)
    scalarMap   = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    c_iter = lambda idx: scalarMap.to_rgba(idx)
    return c_iter

def set_mpl_rc():
    import matplotlib.pyplot as plt
    # from matplotlib.colors import colorConverter
    from matplotlib import rcParams
    try:
        import seaborn as sns
        sns.set(style='ticks', palette='Set1') 
    except:
        print('seaborn not installed')
        
    # Set default colors
    from cycler import cycler
    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    
    # Figure size
    # Only specify height in inch. Width is calculated from golden ratio.
    height = 3.4   # inch
    linewidth  = 1.0
    cross_size = 9 # pt, size of cross markers
    
    # Choose parameters for pdf or print
    figure_path = os.path.join(".", "figures")
    axes_color = "#959595" 
    text_color = "#363636"
    font_family = 'serif'
    linewidth  = 2
    
    # Figure size
    from  scipy.constants import golden_ratio as gr
    figsize  = (gr * height, 1. * height)
    
    fontsize_labels         = 12    # pt, size used in latex document
    fontsize_labels_axes    = fontsize_labels
    fontsize_labels_title   = fontsize_labels
    fontsize_plotlabel      = fontsize_labels       # for labeling plots with 'A', 'B', etc.
    legend_ms  = 2  # scale of markers in legend
    
    # Adapt the matplotlib.rc
    rcParams['font.family']         = font_family
    # rcParams['font.serif']          = 'Computer Modern'
    rcParams['text.usetex']         = False
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
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
    
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    # rcParams['figure.autolayout']   = True # this disables tight layout!
    
    # # Use sans-serif with latex in matplotlib
    # # https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlibhttps://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
    # rcParams['text.latex.preamble'] = [
    #     r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
    #     r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
    #     r'\usepackage{helvet}',    # set the normal font here
    #     r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
    #     r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
    # ]
    
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

    return None
set_mpl_rc()

### Subplots with always the same dim
def subplots(fig, n_rows, n_cols, **kwargs):
    axes = fig.subplots(n_rows, n_cols, **kwargs)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]
    return axes

### Helper function to save figures
def saving_fig(fig, figure_path, fig_name, data_type="png", verbose=True, dpi=1000, save_fig=True):
    # DPI = 1000 is a simple recommendation for publication plots
    from pathlib import Path
    if not Path(figure_path).is_dir():
        os.makedirs(figure_path)
        print("Made new directory ", figure_path)        
    if save_fig:
        if verbose:
            print("Save figure to " + os.path.join(figure_path, fig_name) + "." + data_type)
        if data_type in ("png", "both"):
            fig.savefig(os.path.join(figure_path, fig_name + ".png"), 
                    dpi=dpi, 
                    bbox_inches='tight', format="png") 
        if data_type in ("pdf", "both"):
            fig.savefig(os.path.join(figure_path, fig_name + ".pdf"), 
                    dpi=dpi, 
                    bbox_inches='tight', format="pdf")
        if not data_type in ("png", "pdf", "both"):
            fig.savefig(os.path.join(figure_path, fig_name + "." + data_type),
                    format=data_type)
    else:
        if verbose:
            print("Not saved. Path would be: " + os.path.join(figure_path, fig_name) + "." + data_type)

def fixticks(fig_or_ax, fix_spines=True, max_tick_num=[5, 4], manage_ticks=True):
    """ Reduce the number of ticks to a bare minimum. 
    Input: figure, list of axes or single axes. 
    """
    def tick_manager(ax, max_tick_num=[5, 4]):
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
        
    import matplotlib
    if type(fig_or_ax) is matplotlib.figure.Figure:  
        axes = fig_or_ax.axes
    elif type(fig_or_ax) is list:
        axes = fig_or_ax
    else:
        axes = [fig_or_ax]
    for ax in axes:
        ax.grid(False)      # Turn off grid (distracts!)
        # Set spines to color of axes
        from matplotlib import rcParams
        axes_color = rcParams["axes.edgecolor"]
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
            # Note: otherwise active_side == 'unknown'
        # Take care of tick number
        if manage_ticks:
            tick_manager(ax, max_tick_num)
    return None


def plot_samples(ax, x, y, format_str='-', c=None, label=None, alpha=0.2,
                 mode='mean_std', y_ge_0=False, std_fac=1., alpha_mean=1, 
                 y_ge=None, y_le=None, 
                 **kwargs):
    try:
        import torch
        if type(y) == torch.Tensor:
            y = y.numpy()
    except:
        "Torch not installed."
    if type(mode) == int:
        i_s = mode
        ax.plot(x, y[i_s], format_str, c=c, label=label, **kwargs)
    else: 
        if 'all' in mode:
            for i_s in range(len(y)):
                ax.plot(x, y[i_s], format_str, c=c, alpha=alpha, 
                        label=label if i_s == 0 and not 'mean' in mode else None, 
                       **kwargs)
                
        y_m = np.nanmean(y, axis=0)
        if 'mean' in mode:
            ax.plot(x, y_m, format_str, c=c, label=label, alpha=alpha_mean, **kwargs)
        if 'std' in mode or 'sd' in mode:
            y_s = np.nanstd(y, axis=0) 
            # Apply a factor (as for the standard error, but custom)
            y_s *= std_fac
            # Bound?
            if y_ge is not None:
                lower_bound = y_ge
            elif y_ge_0:
                lower_bound = 0
            else:
                lower_bound = -np.inf
            if y_le is not None:
                upper_bound = y_le
            else:
                upper_bound = np.inf
            ax.fill_between(x, np.maximum(y_m-y_s, lower_bound), np.minimum(y_m+y_s, upper_bound), 
                            color=c, alpha=alpha, linewidth=0)
        if 'se' in mode:
            y_s = np.nanstd(y, axis=0) / np.sqrt(len(y[~np.isnan(y)]) / y.shape[1])
            if y_ge_0:
                ax.fill_between(x, np.maximum(y_m-y_s, 0), y_m+y_s, color=c, alpha=alpha, linewidth=0)
            else:
                ax.fill_between(x, y_m-y_s, y_m+y_s, color=c, alpha=alpha, linewidth=0)
        

def plot_stats(ax, 
               ys_sce, 
               sce_plt = [3, 1],
               sce_lbls = None, 
               plot_type="violin",
               dx_fac=None,
              ):
    import torch
    if type(ys_sce) == torch.Tensor:
        ys_sce = ys_sce.numpy()
    n_task, n_s, n_sce = ys_sce.shape
    n_sce_plt = len(sce_plt)
    
    if n_sce_plt == 2 and sce_lbls is None:
        sce_lbls = ["Aligned", "Oblique"]
    if n_sce_plt == 4 and sce_lbls is None:
        sce_lbls = [
            "Small, decaying", 
            "Small, chaotic", 
            "Large, decaying", 
            "Large, chaotic", ]
    assert len(sce_plt) == len(sce_lbls), "sce_plt and sce_lbls don't agree"
    
    xs = np.arange(n_task)
    dxi = 0.7 / n_sce_plt
    # dxs = np.zeros(n_sce_plt) # plot on top of each other
    # dxs = dxi * (np.arange(n_sce_plt) - (n_sce_plt-1)/2)
    if dx_fac is None:
        dx_fac = 0.6
    dxs = dxi * (np.arange(n_sce_plt) - (n_sce_plt-1)/2) * dx_fac
    
    for ii_sce in range(n_sce_plt):
        i_sce = sce_plt[ii_sce]
        c = cs_sce[i_sce]
        alpha = alpha_sce[i_sce]
        ys = ys_sce[:, :, i_sce].T
        dx = dxs[ii_sce]
        lbl = sce_lbls[ii_sce]
        if plot_type == 'bar':
            ax.bar(xs+dx, np.nanmean(ys, 0), width=dxi, color=c, alpha=alpha, linewidth=0, label=lbl)
            ax.errorbar(xs+dx, np.nanmean(ys, 0), np.nanstd(ys, 0), c=c, alpha=alpha, 
                        marker="", mew=5, lw=0, elinewidth=1.5, capsize=0, capthick=1,)
        if plot_type == 'violin':
            violin_parts = ax.violinplot(ys, xs+dx, points=20, widths=dxi,
                          showmeans=True, showextrema=True, showmedians=False, )
            # Colors
            for vp in violin_parts['bodies']:
                vp.set_color(c)
                vp.set_facecolor(c)
                vp.set_edgecolor(c)
                vp.set_linewidth(0)
                vp.set_alpha(0.6)
            for partname in ('cbars','cmins','cmaxes','cmeans','cmedians',):
                if partname in violin_parts.keys():
                    vp = violin_parts[partname]
                    vp.set_edgecolor(c)
                    vp.set_linewidth(2)
                    vp.set_alpha(alpha)
            # Labels
            ax.bar(xs+dx, [0]*len(xs), width=dxi, bottom=ys.mean(0), 
                   color=c, alpha=alpha, linewidth=0, label=lbl) 
            
        ax.set_xticks(xs)
        ax.set_xlim(xs[0]+ dxs.min() - 0.3, xs[-1] + dxs.max() + 0.3 + 0.1)
        ax.xaxis.set_ticks_position('none')
        
    return xs, dxs
