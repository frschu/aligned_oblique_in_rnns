import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def res_str_to_dict(res_str, loc):
    """ Moves string of results separated by commata into a dictionary, where each 
    key in `res_str` is a variable. This can then be dumped into a pickle file...
    
    Needs as input:
        loc = locals() 
    """
    res_str_list = res_str.replace("\n", "").replace(' ', '').split(',')
    if res_str_list[-1] == '': res_str_list = res_str_list[:-1]
    res_dict = {i: loc[i] for i in res_str_list}
    return res_dict

def map_device(tensors, net):
    """
    Maps a list of tensors to the device used by the network net
    :param tensors: list of tensors
    :param net: nn.Module
    :return: list of tensors
    """
    if net.wi.device != torch.device('cpu'):
        new_tensors = []
        for tensor in tensors:
            new_tensors.append(tensor.to(device=net.wi.device))
        return new_tensors
    else:
        return tensors


def radial_distribution_plot(angles, N=80, bottom=.1, cmap_scale=0.05):
    """
    Plot a radial histogram of angles
    :param angles: a series of angles
    :param N: num bins
    :param bottom: radius of base circle
    :param cmap_scale: to adjust the colormap
    :return:
    """
    angles = angles % (2 * np.pi)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = [np.mean(np.logical_and(angles > theta[i], angles < theta[i+1])) for i in range(len(theta)-1)]
    radii.append(np.mean(angles > theta[-1]))
    width = (2*np.pi) / N
    offset = np.pi / N
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta + offset, radii, width=width, bottom=bottom)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / cmap_scale))
        bar.set_alpha(0.8)
    plt.yticks([])
    
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

def sort_complex(a, tol_exponent=-12):
    """Sort a complex array using the real part first, then the imaginary part.
    Both parts increase.
    Note: the coresponding numpy function seems to be buggy at the moment. (numpy.__version__ == 1.16.1)
>

    
    co_vecs:        an array of vectors to be sorted as well (e.g. the corresponding eigenvectors)
    tol_exponent:   np.unique does not allow for tolerance between entries. This is fixed by rounding.
    """
    # Sort real part first
    pre_sorter = np.argsort(a.real)
    a_sorted = a[pre_sorter]
    # Get counts of sorted unique real values
    _, n_counts = np.unique(np.round(a_sorted.real, -tol_exponent), return_counts=True)
    # Iteratively resort the array... slow but what else can we do?
    idx = 0
    for nc in n_counts:
        entries = a_sorted[idx:idx + nc]
        imag_sorter = np.argsort(entries.imag)
        a_sorted[idx:idx + nc] = entries[imag_sorter]
        # Update
        idx += nc
        
    return a_sorted


def corr_mat(v1, v2=None):
    """ Shape of input vectors: (M, dim_rec), (N, dim_rec)"""
    if v2 is None:
        v2 = v1
    denom = np.linalg.norm(v1, axis=-1)[:, None] * np.linalg.norm(v2, axis=-1)[None, :]
    cov = v1 @ v2.T
    cm = np.where(denom > 0., cov/denom, 0.)
    return cm


def comp_evs(w_recs, max_dim_single=512, n_chunks=10, verbose=False):
    n_rec_epochs, dim_rec, _ = w_recs.shape
    
    # Eigenvalues
    time0 = time.time()
    if dim_rec <= max_dim_single:
        ev_w = np.linalg.eigvals(w_recs)
        ev_w = np.array([sort_complex(evs)[::-1] for evs in ev_w])
    else:
        ev_w = np.zeros((n_rec_epochs, dim_rec), dtype=complex)
        idx_step = int(np.ceil(n_rec_epochs / n_chunks))
        n_chunks = min(n_chunks, n_rec_epochs)
        for i in range(n_chunks):
            if verbose:
                print(i, time.time() - time0)
            idx_low = i * idx_step
            idx_up = (i + 1) * idx_step
            ev_w_i = np.linalg.eigvals(w_recs[idx_low:idx_up])
            ev_w_i = np.array([sort_complex(evs)[::-1] for evs in ev_w_i])
            ev_w[idx_low:idx_up] = ev_w_i
    if verbose:
        print("Calculating EVs took %.1f sec." % (time.time() - time0))
    
    return ev_w

# Sort singular values by coherence between successive singular vectors
def sort_svs(u_dw, vT_dw, sv_dw, wo, wi,
             rank=6, 
             n_back=1, 
             swap_01=False):
    n_rec_epochs = len(u_dw)
    dim_out = wo.shape[1]
    # Truncate
    u = u_dw[:, :, :rank].copy()
    vT = vT_dw[:, :rank].copy()
    s = sv_dw[:, :rank].copy()

    # Go backwards in time, so assing identities consistently
    for i in reversed(range(n_rec_epochs - n_back)):
        u_i = u[i].copy()
        uus = []
        for j in range(n_back):
            u_ipj = u[i+1+j].copy()
            # Correlation between current an previous vectors
            uuj = np.abs(corr_mat(u_i.T, u_ipj.T))
            uus.append(uuj)

        max_uu = np.argmin([uuj.max() for uuj in uus])
        uum = uus[max_uu]

        # Permutation based on similarity between vectors
        perm = np.array([np.argmax(uu_i) for uu_i in uum.T])

        # Permute u, vT, s
        u[i, :, :] = u[i, :, perm].T
        vT[i, :] = vT[i, perm]
        s[i, :] = s[i, perm]

    if swap_01:
        # Swap 0, 1
        ids = [0, 1]
        swap = [1, 0]
        u[:, :, ids] = u[:, :, swap]
        vT[:, ids] = vT[:, swap]
        s[:, ids] = s[:, swap]

    # Choose sign such that the correlation between u_j and wo_j are positive
    for i in reversed(range(n_rec_epochs)):
        for j in range(dim_out):
            if u[i, :, j] @ wo[:, j] < 0:
                u[i, :, j] *= -1
                vT[i, j] *= -1

    n_vecs = 2 * rank
    ac_uv = np.zeros((n_rec_epochs, n_vecs, n_vecs))
    wI = np.r_[wo.T, wi]
    cc_uv_wI = np.zeros((n_rec_epochs, n_vecs, len(wI)))
    for i in range(n_rec_epochs):
        uT_i = u[i].T
        vT_i = vT[i]
        # Compute correlation
        uv_i = np.r_[uT_i, vT_i]
        ac_uv[i] = corr_mat(uv_i, uv_i)
        cc_uv_wI[i] = corr_mat(uv_i, wI)
    
    return u, vT, s, ac_uv, cc_uv_wI



# Find local minimal on array

# import scipy.ndimage.filters as filters
# import scipy.ndimage.morphology as morphology
# def detect_local_minima(arr):
#     # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
#     """
#     Takes an array and detects the troughs using the local maximum filter.
#     Returns a boolean mask of the troughs (i.e. 1 when
#     the pixel's value is the neighborhood maximum, 0 otherwise)
#     """
#     # define an connected neighborhood
#     # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
#     neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
#     # apply the local minimum filter; all locations of minimum value 
#     # in their neighborhood are set to 1
#     # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
#     local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
#     # local_min is a mask that contains the peaks we are 
#     # looking for, but also the background.
#     # In order to isolate the peaks we must remove the background from the mask.
#     # 
#     # we create the mask of the background
#     background = (arr==0)
#     # 
#     # a little technicality: we must erode the background in order to 
#     # successfully subtract it from local_min, otherwise a line will 
#     # appear along the background border (artifact of the local minimum filter)
#     # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
#     eroded_background = morphology.binary_erosion(
#         background, structure=neighborhood, border_value=1)
#     # 
#     # we obtain the final mask, containing only peaks, 
#     # by removing the background from the local_min mask
#     detected_minima = local_min ^ eroded_background
#     return np.where(detected_minima)       
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
def detect_local_minima(arr, max_dist=2):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # Define an connected neighborhood.
    size = 2 * max_dist + 1
    neighborhood = np.bool_(np.ones((size, size)))
    # Find local minima in this neighborhood.
    local_min = filters.minimum_filter(arr, footprint=neighborhood)==arr
    # Remove background
    background = (arr==0)
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    detected_minima = local_min ^ eroded_background
    return detected_minima


# def move_leg(leg, xOffset, yOffset):
#     """ Move legend. Offsets are in coordinates :( """
#     ax = leg.axes
#     bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
#     bb.x0 += xOffset
#     bb.x1 += xOffset
#     bb.y0 += yOffset
#     bb.y1 += yOffset
#     leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
    
    
    
# Draw a circle
from matplotlib.patches import Arc, RegularPolygon
from numpy import radians as rad
def drawCirc(ax,radius,centX,centY,angle_,theta2_,color_='black',lw=2,arrowhead_scale=1.):
    #========Line
    if theta2_ > 0:
        arc = Arc([centX,centY],radius,radius,angle=angle_,
              theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=lw,color=color_)
    else:
        arc = Arc([centX,centY],radius,radius,angle=angle_,
              theta1=theta2_,theta2=0,capstyle='round',linestyle='-',lw=lw,color=color_)
    ax.add_patch(arc)


    #========Create the arrow head
    endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
    endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))

    
    if theta2_ > 0:
        orient = rad(angle_+theta2_)
    else:
        orient = rad(angle_+theta2_ + 180)
    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            radius/9 * lw/2 * arrowhead_scale,                # radius
            orient,     # orientation
            color=color_
        )
    )
    ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius]) 
    # Make sure you keep the axes scaled or else arrow will distort
    
def drawArr(ax,length,centX,centY,angle_,color_='black',lw=2,arrowhead_scale=1.):
    #========Line
    if theta2_ > 0:
        arc = Arc([centX,centY],radius,radius,angle=angle_,
              theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=lw,color=color_)
    else:
        arc = Arc([centX,centY],radius,radius,angle=angle_,
              theta1=theta2_,theta2=0,capstyle='round',linestyle='-',lw=lw,color=color_)
    ax.add_patch(arc)

    #========Create the arrow head
    endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
    endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))
    
    if theta2_ > 0:
        orient = rad(angle_+theta2_)
    else:
        orient = rad(angle_+theta2_ + 180)
    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            radius/9 * lw/2 * arrowhead_scale,                # radius
            orient,     # orientation
            color=color_
        )
    )
    ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius]) 
    # Make sure you keep the axes scaled or else arrow will distort
    
# 3d vector
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)
        
    

def plot_samples(ax, x, y, format_str='-', c=None, label=None, alpha=0.2,
                 mode='mean_std', y_ge_0=False, std_fac=1., alpha_mean=1, 
                 y_ge=None, y_le=None, 
                 **kwargs):
    if type(y) == torch.Tensor:
        y = y.numpy()
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
        
def scale_lbl(scale, size_str="n"):
    """ Generate a label from a scale. """
    nscale= -scale
    if nscale == 0:
        lbl = r"$%s^{0}$" % size_str
    elif nscale % 1 == 0:
        lbl = r"$%s^{-%d}$" % (size_str, nscale)
    elif (nscale * 6) % 1 == 0 and not (nscale * 2) % 1 == 0:
        num, den = (nscale * 3).as_integer_ratio()
        den *= 3
        lbl = r"$%s^{-%d/%d}$" % (size_str, num, den)
    else:
        lbl = r"$%s^{-%d/%d}$" % (size_str, *nscale.as_integer_ratio())
    return lbl




from collections import OrderedDict
def copy_sd(state_dict, device=None):
    sd = OrderedDict()
    for k, v in state_dict().items():
        if device is not None:
            sd[k] = v.detach().clone().to(device)
        else:
            sd[k] = v.detach().clone()
            #sd[k] = v.detach().clone().cpu()
    return sd

def res_to_cpu(res_list):
    for res_i in res_list:
        for mi, res_mi in np.ndenumerate(res_i):
            try:
                for k, v in res_mi.items():
                    res_mi[k] = v.cpu()
            except:
                res_i[mi] = res_mi.cpu()
                
def gen_corr(Xuc, Yuc, center=True):
    """ Generalized correlation of matrices X, Y with shapes
    (..., n), (..., n), respectively. 
    """
    if Xuc.ndim != 2:
        Xuc = Xuc.reshape(-1, Xuc.shape[-1])
    if Yuc.ndim != 2:
        Yuc = Yuc.reshape(-1, Yuc.shape[-1])
        
    # Do everything with torch
    if type(Xuc) == np.ndarray:
        Xuc = torch.from_numpy(Xuc)
    if type(Yuc) == np.ndarray:
        Yuc = torch.from_numpy(Yuc)
    
    # Center
    if center:
        X = Xuc - Xuc.mean(-1)[:, None]
        Y = Yuc - Yuc.mean(-1)[:, None]
    else:
        X = Xuc
        Y = Yuc
    
    if X.shape[0] > X.shape[1]:
        nXX = torch.linalg.norm(X.T @ X).item()
    else:
        nXX = torch.linalg.norm(X @ X.T).item()
    if Y.shape[0] > Y.shape[1]:
        nYY = torch.linalg.norm(Y.T @ Y).item()
    else:
        nYY = torch.linalg.norm(Y @ Y.T).item()
        
    if nXX == 0 or nYY == 0:
        return 0, 0
    else:
        nXY = torch.linalg.norm(X @ Y.T).item()
        # Naive version
        corr = nXY / np.sqrt(nXX * nYY)
        # PR corrected version:
        DX_14 = torch.linalg.norm(X).item() / np.sqrt(nXX)
        DY_14 = torch.linalg.norm(Y).item() / np.sqrt(nYY)
        corr_tilde = corr * max(DX_14 / DY_14, DY_14 / DX_14)
        return corr, corr_tilde
    
def comp_pr(Xuc, center=True):
    """ Compute the dimension based on the inverse participation ratio. 
    Note that some authors (eg Gao 2017) define this measure as the participation
    ratio...
    """
    if type(Xuc) == np.ndarray:
        Xuc = torch.from_numpy(Xuc)
    Xuc = Xuc.reshape(-1, Xuc.shape[-1])
    # Center
    if center:
        X = Xuc - Xuc.mean(0)
    else:
        X = Xuc
    if X.shape[0] > X.shape[1]:
        nXX = ((X.T @ X)**2).sum().item()
    else:
        nXX = ((X @ X.T)**2).sum().item()
    nX = (X**2).sum().item()
    if nX != 0:
        return nX**2 / nXX
    else:
        return 0.
    
    

from mpl_toolkits.mplot3d import axes3d
from matplotlib.transforms import Bbox
def add_inset_axes(rect, units="ax", ax_target=None, fig=None, projection=None, 
                   transparent_bg=True, no_ticks=True,
                   **kw):
    """
    Wrapper around `fig.add_axes` to achieve `ax.inset_axes` functionality
    that works also for insetting 3D plot on 2D ax/figures
    """
    assert ax_target is not None or fig is not None, "`fig` or `ax_target` must be provided!"
    _units = {"ax", "norm2ax", "norm2fig"}
    assert {units} <= _units, "`rect_units` not in {}".format(repr(_units))

    if ax_target is not None:
        # Inspired from:
        # https://stackoverflow.com/questions/14568545/convert-matplotlib-data-units-to-normalized-units
        bb_data = Bbox.from_bounds(*rect)
        trans = ax_target.transData if units == "ax" else ax_target.transAxes
        disp_coords = trans.transform(bb_data)
        fig = ax_target.get_figure()
        fig_coord = fig.transFigure.inverted().transform(disp_coords)
    elif fig is not None:
        if ax_target is not None and units != "norm2fig":
            bb_data = Bbox.from_bounds(*rect)
            trans = ax_target.transData if units == "ax" else ax_target.transAxes
            disp_coords = trans.transform(bb_data)
        else:
            fig_coord = Bbox.from_bounds(*rect)

    axin = fig.add_axes(
        Bbox(fig_coord),
        projection=projection, **kw)
    
    if transparent_bg:
        axin.patch.set_alpha(0.)
        axin.xaxis.set_alpha(0.)
        axin.yaxis.set_alpha(0.)
        if projection == '3d':
            axin.zaxis.set_alpha(0.)
    if no_ticks:
        axin.set_xticks([])
        axin.set_yticks([])
        if projection == '3d':
            axin.set_zticks([])

    return axin




##############################################################################################

def ridge_CCA(X, Y, alpha=0, use_full_inv_sqrt=False, normalize_trafo=False, eps=1e-5):
    """ Ridge CCA: Compute singular values of transformed X, Y. 
    Transformation is a mixture of mean-centering (CX) and 
    ZCA whitening.
    For details, see:
        Williams et al, "Generalized Shape Metrics on Neural Representations", Neurips 2021
    
    Input:
    X, Y:   matrices with shape (p, n), (p, m), respectively, where p = number of samples.
    alpha:  float, ridge parameter.
            alpha = 0: CCA, invariance to reversible linear transformations.
            alpha = 1: "Orthonormal partial least squares", invariance to orthogonal trafos.
    use_full_inv_sqrt: 
            bool, specifies whether inverse sqrt is applied to the 
            linear interpolation or the cov matrix alone. 
            
    Returns:
    U, S, VT: SVD of the transformed covariance matrix
    """
    def inv_sqrt(M, eps=eps):
        """ Inverse square root of hemitian matrix. """
        # Compute via eigenvalues
        ew, Q = torch.linalg.eigh(M)
        inv_sqrt_ew = torch.zeros_like(ew)
        mask = ew > ew.max() * eps
        inv_sqrt_ew[mask] = 1/ torch.sqrt(ew[mask])
        return Q * inv_sqrt_ew[None] @ Q.T
    
    p, n_X = X.shape
    p_Y, n_Y = Y.shape
    assert p == p_Y, "X, Y must have the same number of rows."
    # Mean-center the columns:
    CX = X - X.mean(0)
    CY = Y - Y.mean(0)
    if alpha == 1:
        # Only mean-center
        X_t = CX
        Y_t = CY
    elif alpha == 0:
        MX = inv_sqrt(CX.T @ CX)
        MY = inv_sqrt(CY.T @ CY)
        X_t = CX @ MX
        Y_t = CY @ MY
    else:
        if use_full_inv_sqrt:
            # Apply inverse sqrt to the full interpolation matrix (e.g. Nielsen et al., 1998)
            MX = inv_sqrt(alpha * torch.eye(n_X) + (1 - alpha) * (CX.T @ CX))
            MY = inv_sqrt(alpha * torch.eye(n_Y) + (1 - alpha) * (CY.T @ CY))
        else:
            # Different version: only apply inv sqrt to centered cov matrix (Eq. 9 in Williams et al.)
            MX = alpha * torch.eye(n_X) + (1 - alpha) * inv_sqrt(CX.T @ CX)
            MY = alpha * torch.eye(n_Y) + (1 - alpha) * inv_sqrt(CY.T @ CY)
            
            # One can play with other interpolations...
            # MX = alpha * torch.eye(n_X) + np.sqrt(1 - alpha**2) * inv_sqrt(X.T @ CX)
            # MY = alpha * torch.eye(n_Y) + np.sqrt(1 - alpha**2) * inv_sqrt(Y.T @ CY)
            
        X_t = CX @ MX
        Y_t = CY @ MY
        
    # Normalize before computing the cov. Else we get overflow
    norm_X_t = torch.linalg.norm(X_t)
    norm_Y_t = torch.linalg.norm(Y_t)
    if normalize_trafo:
        X_t = X_t / norm_X_t
        Y_t = Y_t / norm_Y_t
    
    # Compute transformed cov
    XY_t = X_t.T @ Y_t           # MX @ (CX.T @ CY) @ MY
    
    # Compute singular values
    U, S, VT = torch.linalg.svd(XY_t)
    argmin_XYQ = S.sum()
    
    
    if alpha == 1:
        # Rotation matrices
        WX = U
        WY = VT.T
    else:
        # Rotation matrices
        WX = MX @ U
        WY = MY @ VT.T
    
    
    # Euclidean distance
    if normalize_trafo:
        arg_eucl = 2 - 2 * argmin_XYQ
    else:
        arg_eucl = norm_X_t**2 + norm_Y_t**2 - 2 * argmin_XYQ
    arg_eucl = torch.nn.ReLU()(arg_eucl)
    d_eucl = torch.sqrt(arg_eucl)

    # Angle (-> cos similarity)
    if normalize_trafo:
        arg_ang = argmin_XYQ
    else:
        arg_ang = argmin_XYQ / (norm_X_t * norm_Y_t)
    arg_ang = torch.minimum(arg_ang, arg_ang*0 + 1)
    d_ang = torch.arccos(arg_ang)
    
    # return d_eucl, d_ang, U, S, VT, norm_X_t, norm_Y_t
    return d_eucl, d_ang, WX, S, WY, norm_X_t, norm_Y_t



def sort_corr(S, direc="max"):
    """Sort correlation matrix `S`. Parameter `direc` indicates direction, 
    starting with 'max' or 'min' entry of S. """
    # Remove diagonal
    Sn = S.copy()
    if direc == "max":
        Sn_min = Sn.min()
        np.fill_diagonal(Sn, Sn_min-1)
        # Start with largest correlation
        node_0 = np.argmax(Sn.max(axis=1))
    elif direc == "min":
        Sn_max = Sn.max()
        np.fill_diagonal(Sn, Sn_max+1)
        # Start with largest correlation
        node_0 = np.argmin(Sn.min(axis=1))
    # Iterate
    node_i = node_0
    sorter_corr = [node_i]
    for i in range(1, len(S)):
        S_i = Sn[node_i]
        if direc == "max":
            S_i[sorter_corr] = Sn_min-1
            node_i = np.argmax(S_i)
        elif direc == "min":
            S_i[sorter_corr] = Sn_max+1
            node_i = np.argmin(S_i)
        sorter_corr.append(node_i)
    return np.array(sorter_corr)



def confidence_ellipse(ax, cov, mean_xy=np.zeros(2), n_std=1.0, 
                       edgecolor='k', facecolor='None',
                       **kwargs):
    """ https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html"""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor,
                      **kwargs)
    # Scale
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(*mean_xy)
    ellipse.set_transform(transf + ax.transData)
    
    return ax.add_patch(ellipse)




################################################################################################
def comp_acc(output, target, mask, acc_thr = 0.2):
    """ Compute accuracy for decision making tasks (Romo and Mante).
    In contrast to the loss, this discards the fixation period and only concerns the
    decision period, which is inferred as times with nonzero loss. 
    We then ask the binary question whether the output was within `acc_thr` 
    of the target averaged over the decision period.
    The accuracy is computed over the batch.
    
    acc_thr defaults to 0.2.
    """
    batch_size = output.shape[0]
    # Mask: where is a nonzero target demanded (i.e. not fixation).
    mask_acc = mask * (target != 0.)
    # Difference between output and target, normalized by target amplitude.
    rel_dto = torch.Tensor([
        (torch.abs(target - output) / torch.abs(target))[i_b][mask_acc[i_b]].mean() 
        for i_b in range(batch_size)])
    # Correct decision if this difference is smaller than acc_thr.
    acc = (rel_dto < acc_thr).sum().item() / batch_size
    return acc


def to_dev(arr_or_list, device):
    to_dev_arr = lambda arr: (torch.from_numpy(arr).to(device)
                           if type(arr) == np.ndarray else arr.to(device))
    if type(arr_or_list) in [list, tuple]:
        return [to_dev_arr(arr) for arr in arr_or_list]
    else:
        return to_dev_arr(arr_or_list)