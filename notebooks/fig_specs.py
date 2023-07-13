# Path for figures, data
figure_path = "../figures/"
data_path = "../data/"

# Golden ratio
gr = 1.618
# Width of figures (full page)
fig_width = 8
# import scipy
# fig_height = fig_width / scipy.constants.golden_ratio
fig_height = fig_width / gr
# Fontsize
fs = 12

# flbs = ["(%s)"%s for s in "abcdefghijkl"] 
flbs = ["%s"%s for s in "ABCDEFGHIJKL"] 

struc_str = "\Delta W"

# # Line width
# lw = 2 # set in style.py!

# Color for legend
c_leg = '0.7'

# Colors for different g
from imp import reload
import numpy as np
import style; reload(style)
from style import tango
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
# cs_sce = cs[[1, 1, 0, 0], [2, 1, 1, 0]]
# cs_sce = cs[[1, 1, 0, 0], [1, 0, 1, 0]]
# alpha_sce = np.array([0.7, 1., 0.7, 1.0])
# New: all orange
cs_sce = cs[[0, 1, 0, 1], [2, 2, 0, 0]]
alpha_sce = np.array([0.8, 0.8, 1, 1])
# Colors for cycling contexts
from style import colors
cs_ctx = colors[[4, 2]]


# Subplots with always the same dim
def subplots(fig, n_rows, n_cols, **kwargs):
    axes = fig.subplots(n_rows, n_cols, **kwargs)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]
    return axes




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
