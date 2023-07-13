import os
import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm
from sklearn.decomposition import PCA
from fig_specs import data_path


# Helpers to transform matlab files to python
def to_dict(arr):
    keys = np.array(arr.dtype.names)
    return {key: arr[np.where(keys == key)[0][0]] for key in keys}
def to_arr(data_i):
    arr = np.stack(data_i[0]).transpose((2, 0, 1))
    return arr

def load_golub_2018(i_ba, 
                    move_only=True, 
                    file_name = "20120305.mat", 
                    verbose=False):
    """
    Load data from Golub et al. 2018 Nature Neuroscience "Learning by neural reassociation"
    """
    data_file = os.path.join(data_path, "bci", file_name)
    data = loadmat(data_file)
    # Data
    data_before = to_dict(data["beforeLearning"][0, 0][1][0, 0])
    data_after = to_dict(data["afterLearning"][0, 0][1][0, 0])
    # Parameters
    expParams = data["expParams"]
    monkey = expParams[0, 0][0][0]
    date = expParams[0, 0][1][0]
    params_before = to_dict(expParams[0, 0][2][0, 0])
    params_after = to_dict(expParams[0, 0][3][0, 0])
    del data, expParams
    # Join parameters. They are the same before and after for most.
    for key in params_before.keys():
        d_ba = np.linalg.norm(params_before[key] - params_after[key])
        if d_ba > 0:
            if verbose:
                print(key, d_ba)
    params = params_before.copy()
    params.pop("M2")
    M2_intuitive = params_before["M2"] / np.diag(params["Sigma_u"])
    M2_perturbed = params_after["M2"] / np.diag(params["Sigma_u"])
    params.pop("m0")
    params["m0_before"] = params_before["m0"]
    params["m0_after"] = params_after["m0"]
    params["eta"] = params_after["eta"]
    del params_before, params_after
    if verbose:
        print("Monkey, data:", monkey, date)
        print("Data keys:\n", data_before.keys())
        print("Params keys:\n", params.keys())

    # Mappings before and after perturbation
    beta = params['beta']
    Sigma_z = params['Sigma_z']
    B = params['K']
    c_bias = -(params['K'] @ params['d'])[:, 0]
    eta = params['eta']
    DVT = params['D'] @ params['V'].T
    w_out_intuitive = B @ Sigma_z @ beta
    w_out_perturbed = B @ eta @ Sigma_z @ beta
    # Consistency for BCI mappings
    assert np.allclose(w_out_intuitive, M2_intuitive)
    assert np.allclose(w_out_perturbed, M2_perturbed)
    
    # Neural activity, factors, output.
    # Some datasets don't include the neural activity. 
    # To allow for an equal footing, we only use the orthogonal factors ('latents').
    # These generate the activity on the manifold, and the output.
    z_orth_b = to_arr(data_before['factorActivity'])
    z_orth_a = to_arr(data_after['factorActivity']) 
    # Neural activity constrained to the manifold. 
    u_b = z_orth_b @ np.linalg.pinv(DVT @ beta).T
    u_a = z_orth_a @ np.linalg.pinv(DVT @ beta).T
    # Z-scored factors and output
    z_zs_b = u_b @ beta.T @ Sigma_z 
    z_zs_a = u_a @ beta.T @ Sigma_z 
    v_b_intuitive = u_b @ w_out_intuitive.T + c_bias 
    v_b_perturbed = u_b @ w_out_perturbed.T + c_bias 
    v_a_intuitive = u_a @ w_out_intuitive.T + c_bias 
    v_a_perturbed = u_a @ w_out_perturbed.T + c_bias 
    
    # Checks
    # Neural activity (if given)
    # u_b_full = (to_arr(data_before['spikes']) - params["mu_u"][:, 0]) * np.diag(params['Sigma_u'])
    # u_a_full = (to_arr(data_after['spikes']) - params["mu_u"][:, 0]) * np.diag(params['Sigma_u'])
    # VT = np.linalg.svd(DVT @ beta, full_matrices=False)[2]
    # assert np.allclose(u_b_full @ VT.T @ VT, u_b)
    # assert np.allclose(u_a_full @ VT.T @ VT, u_a)
    
    # Raw spike trains
    bin_width_ms = 45 
    u_b_full = to_arr(data_before['spikes']) / (bin_width_ms * 1e-3)
    u_a_full = to_arr(data_after['spikes']) / (bin_width_ms * 1e-3)
    
    # The output is not matched to numerical precision, but very close nonetheless.
    assert np.allclose(v_b_intuitive, to_arr(data_before['vraw_intuitive']))
    assert np.allclose(v_b_perturbed, to_arr(data_before['vraw_perturbed']))
    assert np.allclose(v_a_intuitive, to_arr(data_after['vraw_intuitive']))
    assert np.allclose(v_a_perturbed, to_arr(data_after['vraw_perturbed']))
    
    # Prepare all possible output and hids for simpler downstream analysis.
    ba_lbls = ["before", "after"]
    output_dict = {}
    hids_dict = {}
    for j_ba, ba_lbl in enumerate(ba_lbls):
        data_i = [data_before, data_after][j_ba]
        ctt_dir = np.stack(data_i['cursorToTargetDirection'][0]).transpose((2, 0, 1))
        n_repeats, n_target, dim_out = ctt_dir.shape
        ctt_ang = np.rad2deg(np.arctan2(*ctt_dir.reshape(-1, 2)[:, ::-1].T) % (2*np.pi) )
        target_ang = np.arange(0, 360, 45)
        if move_only:
            mask_m = ~np.any(np.isclose(ctt_ang[None], target_ang[:, None]), axis=0)
        else:
            mask_m = np.ones(ctt_ang.shape, dtype=bool)
        output_i = [v_b_intuitive, v_a_perturbed][j_ba].reshape(n_repeats * n_target, dim_out)[mask_m]
        hids_i = [u_b_full, u_a_full][j_ba].reshape(n_repeats * n_target, -1)[mask_m]
        output_dict[ba_lbl] = output_i
        hids_dict[ba_lbl] = hids_i
    
    return output_dict, hids_dict



def load_hennig_2018(i_ba, 
                     move_only=True,
                     file_name = "20131218.mat", 
                     use_raw_output=True,
                     verbose=False, 
                    ):
    """
    Load data from Hennig et al. 2018 eLife "Constraints on neural learning".
    """
    data_file = os.path.join(data_path, "bci", file_name)
    data = loadmat(data_file)
    date = data['D'][0, 0][0]
    # Decoding
    decoding = to_dict(data['D'][0, 0][1][0, 0])
    # Parameters
    FA_params = to_dict(decoding['FactorAnalysisParams'][0, 0])
    # Blocks
    block_1 = to_dict(data['D'][0, 0][2][0, 0])
    block_2 = to_dict(data['D'][0, 0][2][0, 1])
    if verbose:
        print("data", np.array(data['D'][0, 0].dtype.names))
        print("decoding:", decoding.keys())
        print("FA params: ", FA_params.keys())
        print("block_1: ", block_1.keys())
        print("nDecoder: ", to_dict(block_1['nDecoder'][0, 0]).keys())
        print("fDecoder: ", to_dict(block_1['fDecoder'][0, 0]).keys())

    # Variables from data
    L = FA_params['L']
    Psi = np.diag(FA_params['ph'].flatten())
    beta = L.T @ np.linalg.inv(L @ L.T + Psi) 
    Sigma_z = np.diag(1 / FA_params['factorStd'].flatten())
    DVT = FA_params['spikeRot'].T
    B = block_1['fDecoder']['M2'][0, 0] @ DVT * FA_params['factorStd'].T
    B_eta = block_2['fDecoder']['M2'][0, 0] @ DVT * FA_params['factorStd'].T
    # Kalman filter
    M0 = block_1['nDecoder']['M2'][0, 0] @ decoding['spikeCountMean'].T + block_1['nDecoder']['M0'][0, 0]
    M1 = block_1['nDecoder']['M1'][0, 0]
    M2_intuitive = block_1['nDecoder']['M2'][0, 0] * decoding['spikeCountStd']
    M2_perturbed = block_2['nDecoder']['M2'][0, 0] * decoding['spikeCountStd']
    # Consistency checks
    assert np.allclose(M0, block_2['nDecoder']['M2'][0, 0] @ decoding['spikeCountMean'].T + block_2['nDecoder']['M0'][0, 0])
    assert np.allclose(M0, block_1['fDecoder']['M0'][0, 0])
    assert np.allclose(M0, block_2['fDecoder']['M0'][0, 0])
    assert np.allclose(M1,  block_2['nDecoder']['M1'][0, 0])
    assert np.allclose(M1,  block_1['fDecoder']['M1'][0, 0])
    assert np.allclose(M1,  block_2['fDecoder']['M1'][0, 0])
    assert np.allclose(M2_intuitive, block_1['fDecoder']['M2'][0, 0] @ DVT @ beta)
    assert np.allclose(M2_perturbed, block_2['fDecoder']['M2'][0, 0] @ DVT @ beta)

    ########################################################################################
    # Obtain permutation from 2nd factor decoder
    dim_mani = beta.shape[0]
    perm = np.argmin((B_eta[0, :, None] - B[0, None, :])**2, axis=1)
    eta = np.eye(dim_mani, dtype=int)[:, perm]
    assert np.allclose(B @ eta, B[:, perm])

    # Output mappings, bias
    w_out_intuitive = B @ Sigma_z @ beta
    w_out_perturbed = B @ eta @ Sigma_z @ beta
    c_bias = M0.flatten()
    # Consistency for BCI mappings
    assert np.allclose(w_out_intuitive, M2_intuitive)
    assert np.allclose(w_out_perturbed, M2_perturbed)

    # Neural activity, factors, output.
    # Some datasets don't include the neural activity. 
    # To allow for an equal footing, we only use the orthogonal factors ('latents').
    # These generate the activity on the manifold, and the output.
    z_orth_b = block_1['latents']
    z_orth_a = block_2['latents']
    # Neural activity constrained to the manifold. 
    u_b = z_orth_b @ np.linalg.pinv(DVT @ beta).T
    u_a = z_orth_a @ np.linalg.pinv(DVT @ beta).T
    # Z-scored factors and output
    z_zs_b = u_b @ beta.T @ Sigma_z 
    z_zs_a = u_a @ beta.T @ Sigma_z 
    if use_raw_output:
        v_b_intuitive = u_b @ w_out_intuitive.T + c_bias 
        v_a_perturbed = u_a @ w_out_perturbed.T + c_bias 
        # The output is not matched to numerical precision, but very close nonetheless.
        assert np.max(v_b_intuitive - (block_1['velNext'] - block_1['vel'] @ M1.T - M0.T)) < 1e-4
        assert np.max(v_a_perturbed - (block_2['velNext'] - block_2['vel'] @ M1.T - M0.T)) < 1e-4
    else:
        ##################################################
        ### Use filtered velocity as output
        ##################################################
        v_b_intuitive = block_1['velNext']
        v_a_perturbed = block_2['velNext']

    # Checks
    # Neural activity (if given)
    # u_b_full = (block_1['spikes'] - decoding['spikeCountMean']) / decoding['spikeCountStd'].flatten()
    # u_a_full = (block_2['spikes'] - decoding['spikeCountMean']) / decoding['spikeCountStd'].flatten()
    # VT = np.linalg.svd(DVT @ beta, full_matrices=False)[2]
    # assert np.allclose(u_b_full @ VT.T @ VT, u_b)
    # assert np.allclose(u_a_full @ VT.T @ VT, u_a)
    
    # Raw spike trains
    bin_width_ms = 45 
    u_b_full = block_1['spikes'] / (bin_width_ms * 1e-3)
    u_a_full = block_2['spikes'] / (bin_width_ms * 1e-3)
    

    # Prepare all possible output and hids for simpler downstream analysis.
    ba_lbls = ["before", "after"]
    output_dict = {}
    hids_dict = {}
    for j_ba, ba_lbl in enumerate(ba_lbls):
        block_i = [block_1, block_2][j_ba]
        if move_only:
            mask_m = ~block_i["isFreezePeriod"].astype(bool).flatten()
        else:
            mask_m = np.ones(block_i['thetas'][:, 0].shape, dtype=bool)
        output_i = [v_b_intuitive, v_a_perturbed][j_ba][mask_m]
        hids_i = [u_b_full, u_a_full][j_ba][mask_m]
        output_dict[ba_lbl] = output_i
        hids_dict[ba_lbl] = hids_i
    
    return output_dict, hids_dict



def load_degenhart_2020(
    file_name="20160325.mat",
    verbose=False,
    move_only=True,
    fit_kalman=False,
    ):

    """
    Load data from Degenhard et al. 2020 Nature Biomed Engineering 
    "Stabilization of a brain–computer interface via the alignment of low-dimensional spaces of neural activity".
    
    The given velocity 'decodedVl' is 
        y_t = K x_t + B y_t-1
    Therefore, y_t cannot be directly obtained from x_t. 
    Set `fit_kalman` = True in order first fit K and B, and to then define
        output_t = y_t - B y_t-1
    This has a few less data points, but allows to yield R^2 = 100%.
    Without this option, one only obtains R^2 = 85%.
    """
    import mat73

    data_file = os.path.join(data_path, "bci", file_name)
    data_all = mat73.loadmat(data_file)
    if verbose:
        print("Top level keys", data_all.keys())
    # Trial data
    data = data_all["trialData"]
    if verbose:
        print("\nTrial data keys:\n", data.keys())
    # Analyze activity during evaluation trials
    trial_type = np.array(data['type'])
    trial_ids = np.where(trial_type == 'baselineEvaluation')[0]
    # Only take good electrodes. Matlab indexing is shifted...
    good_electrodes = data_all['goodElectrodes'].astype('int') - 1
    # Movement onset times
    moveBin = np.array(data['moveBin']).astype('int')
    
    if fit_kalman:
        # Regress output from hidden states after PC
        from sklearn.linear_model import RidgeCV
        # Possible rgularization parameters
        alpha_range = np.logspace(-3, 6, 20)
        output = []
        hids = []
        for trial_id in trial_ids:
            if move_only:
                i_move = moveBin[trial_id]
            else:
                i_move = 0
            # Output: the decoded velocity.
            # Not sure this is already filtered by the Kalman filter...
            output_i = data['decodedVl'][trial_id][i_move:]
            # Neural activity: spike count at the electrodes
            hids_i = data['binCounts'][trial_id][i_move:, good_electrodes]
            # Fit both the weights and the Kalman filter
            hids_i = np.c_[hids_i[1:], output_i[:-1]]
            output_i = output_i[1:]
            # Save
            output.append(output_i)
            hids.append(hids_i)
        # Fuse   
        output = np.concatenate(output)
        hids = np.concatenate(hids)
        # # z-score hidden variables
        ### This is necessary here because otherwise the output and hidden states
        ### may live on very different scales.
        output_mean = output.mean(0)
        output_std = output.std(0)
        output = (output - output_mean) / output_std
        hids[:, :-2] = (hids[:, :-2] - hids[:, :-2].mean(0)) / hids[:, :-2].std(0)
        hids[:, -2:] = (hids[:, -2:] - output_mean) / output_std
        # Full fit for reference
        ridge = RidgeCV(alphas=alpha_range).fit(hids, output)
        r_sq_full = ridge.score(hids, output)
        w_out_B = ridge.coef_ 
        w_out = w_out_B[:, :-2]
        B_kalman = w_out_B[:, -2:]
        # Actual output and hids: the corrected output, no past output for hidden states
        output_corr = output - hids[:, -2:] @ B_kalman.T
        output = (output_corr - output_corr.mean(0)) / output_corr.std(0)
        hids = hids[:, :-2]
    else:
        output = []
        hids = []
        for trial_id in trial_ids:
            if move_only:
                i_move = moveBin[trial_id]
            else:
                i_move = 0
            # Output: the decoded velocity.
            # Not sure this is already filtered by the Kalman filter...
            output_i = data['decodedVl'][trial_id][i_move:]
            # Neural activity: spike count at the electrodes
            bin_width_ms = 45 
            hids_i = (
                data['binCounts'][trial_id][i_move:, good_electrodes] 
                / (bin_width_ms * 1e-3)
            )
            # Save
            output.append(output_i)
            hids.append(hids_i)
        # Fuse   
        output = np.concatenate(output)
        hids = np.concatenate(hids).astype(int)

    return output, hids


############################################################################################
### Russo
############################################################################################
def load_russo_2018(file_name, subs_step=1):
    """
    See 
    https://data.mendeley.com/datasets/tfcwp8bp5j/1
    for information.
    """
    data_path_russo = os.path.join(data_path, "russo_monkey_data")
    data_file = os.path.join(data_path_russo, file_name)
    data_mat = loadmat(data_file)
    key_data = 'Pc' if 'Cousteau' in data_file else 'Pd'
    data = to_dict(data_mat[key_data][0, 0])
    mask_dict = to_dict(data['mask'][0, 0])
    # Mask for movement
    mask_m = ~np.isnan(mask_dict['cycleNum'][:, 0])
    
    #########################################################################
    # Join all three output modalities. This will save some time downstream, 
    # because we only need to do the preprocessing once. 
    output_mods = ["emg", "hand_pos", "hand_vel", "hand_acc"]
    t_delays = {
        "emg": -0,
        "hand_pos": -100,
        "hand_vel": -100,
        "hand_acc": -0,
    }
    t = mask_dict['time'][:, 0] * 1e3 # time should be in ms, but is given in sec.
    dt = int(t[1] - t[0]) 
    ids_mov = np.where(mask_m)[0]
    output_dict = {}
    hids_dict = {}
    for output_mod in output_mods:
        # Choose the output
        if output_mod == 'emg':
            z = data['zA'][ids_mov]
            # Project EMG to 2D output
            dim_out = 2
            pca = PCA(n_components=dim_out)
            output_i = pca.fit_transform(z)
        else:
            # For hand pos and vel, we have 4 examples. Let's take the mean of these...
            dim_out = 2
            n_ex = 4
            if output_mod == 'hand_pos':
                output_i = data['pA'].reshape(-1, n_ex, dim_out)[ids_mov].mean(1)
            if output_mod == 'hand_vel':
                output_i = data['vA'].reshape(-1, n_ex, dim_out)[ids_mov].mean(1)
            if output_mod == 'hand_acc':
                output_i = np.diff(data['vA'], axis=0).reshape(-1, n_ex, dim_out)[ids_mov].mean(1)

        # Firing rates with delay
        t_delay = t_delays[output_mod]
        ids_hids = ids_mov + int(t_delay / dt)
        # hids_i = data['xA'][ids_hids]
        hids_i = data['xA_raw'][ids_hids]
        # Subsampling step for data (to avoid overflow and/or too close points)
        output_dict[output_mod] = output_i[::subs_step]
        hids_dict[output_mod] = hids_i[::subs_step]
    
    return output_dict, hids_dict

############################################################################################
### Neural Latents Benchmark Datasets
############################################################################################
def load_nlb_maze(dataset_name, 
                verbose=False,
                bin_width_ms=45, #ms (BCI: 45ms, no filter)
                kern_sd_ms=45, #ms (25ms: Russo, but with much higher firing rates)
    ):
    from nlb_tools.nwb_interface import NWBDataset
    import pandas as pd
    filter_rates = kern_sd_ms > 0
    
    _, nlb_dataset = dataset_name.split('-')

    ## Load data from NWB file
    if nlb_dataset == "mc_maze":
        file_name = '000128'
    if nlb_dataset == "mc_maze_large":
        file_name = '000138'
    if nlb_dataset == "mc_maze_medium":
        file_name = '000139'
    if nlb_dataset == "mc_maze_small":
        file_name = '000140'
    data_path_nlb = os.path.join(data_path, "neural_latents")
    data_file = os.path.join(data_path_nlb, file_name, 'sub-Jenkins/') ### sub-Jenkins may not work here!
    dataset = NWBDataset(data_file)
    if verbose:
        print("Loaded from ", data_file)
        print("All data")
        print("dataset.data.shape", dataset.data.shape)

    ### Dataset preparation
    # Choose bin width and resample
    dataset.resample(bin_width_ms)
    if verbose:
        print("All data, binned")
        print(dataset.data.shape)

    ################################################################################
    ### Some preprocessing
    # The delays and reaction times are different in each trial. 
    # To get a better control over this, we will center our arrays
    # around the movement onset (which is already computed for us). 

    # Remove test trials, because these do not contain behavior
    trial_mask = np.array(dataset.trial_info["split"] != "test")
    # There are some catch trials which have no preparation time (delay < 0)
    # Let's remove these for now.
    min_delay = 448
    catch_trials = dataset.trial_info["delay"] < min_delay
    trial_mask *= ~catch_trials

    # Filter width for rates. It's useful to define this here, so we can set the margins appropriately.
    if filter_rates:
        import scipy.signal as signal
        def filt(x):
            kern_sd = int(round(kern_sd_ms / bin_width_ms))
            window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
            window /= np.sum(window)
            return np.convolve(x, window, 'same')
        margin = 2*kern_sd_ms
    else:
        margin = 0

    # Get trial data 
    # Time for aligment
    align_field = "move_onset_time"
    # How many ms before and after aligment time
    align_range = (-700, 785)
    # Obtain trial data: 
    trial_data = dataset.make_trial_data(
        align_field=align_field, align_range=align_range,
        ignored_trials=~trial_mask, 
        margin=margin)
    # Reduce trial info
    trial_info = dataset.trial_info[trial_mask]
    trial_info.set_index("trial_id", inplace=True)
    # Free the memory from the full dataset, which we won't use anymore
    del dataset
    # Merge heldout and non-heldout spikes, since we don't distinguish here. 
    trial_data.rename(columns={"heldout_spikes": "spikes"}, inplace=True)
    trial_data = trial_data.reindex(trial_data.columns.sortlevel(0)[0], axis=1)
    # Multiindex: use trial ID as index
    trial_data['t'] = trial_data["align_time"] / np.timedelta64(1, "ms")
    trial_data.set_index(["trial_id", "t"], inplace=True)
    # Add trial type and version as columns. This will be helpful for conditioned averaging
    trial_data["trial_type"] = np.int_(trial_info["trial_type"][trial_data.reset_index()["trial_id"]].values)
    trial_data["trial_version"] = np.int_(trial_info["trial_version"][trial_data.reset_index()["trial_id"]].values)
    # Make these columns indices
    trial_data.reset_index(inplace=True)
    trial_data.set_index(["trial_id", "trial_type", "trial_version", "t"], inplace=True)

    # Rates
    if filter_rates:
        rates = trial_data["spikes"].groupby(["trial_id"]).transform(filt).astype(np.float32)
    else:
        rates = trial_data["spikes"].astype(np.float32)
    # In Hz:
    rates = rates / (bin_width_ms * 1e-3)
    # Exclude the margins (these may be used if rates are filtered)
    rates = rates[~trial_data["margin"]]
    if verbose:
        print("rates, binned, removed chatch trials, test trials, margins")
        print("rates.shape", rates.shape)

    ################################################################################
    # Some other definitions that will be useful below.
    # Number of trials, and different trial types
    trial_ids = np.array(trial_info.index)
    n_trials = len(trial_info)
    # Number of time steps
    n_t = int(np.diff(align_range) / bin_width_ms)
    # Number of channels (neurons recorded)
    n_ch = rates.shape[1]
    # Number of different trial types and versions
    trial_types = np.int_(trial_info["trial_type"].unique())
    trial_types.sort()
    trial_versions = np.int_(trial_info["trial_version"].unique())
    trial_versions.sort()
    n_tt = len(trial_types)
    n_tv = len(trial_versions)
    # Trial time in ms
    ts = np.arange(n_t) * bin_width_ms + align_range[0]
    # Checks
    if verbose:
        print("Number of time steps:", n_t)
        print("Number of channels:", n_ch)
        print("Number of trials:", n_trials)
        print("Number of different trial types:", n_tt)
        print("Number of different trial versions:", n_tv)

    # Behavior: cursor, eyes, hand, hand vel
    obs_bh = ["cursor_pos", "eye_pos", "hand_pos", "hand_vel"]
    xy_bh = ["x", "y"]
    mi_bh = pd.MultiIndex.from_product([obs_bh, xy_bh])
    behav = trial_data[mi_bh][~trial_data["margin"]]

    # Add acceleration to behavior
    vel = behav["hand_vel"]
    dv = vel.groupby(["trial_id"]).diff(1)
    dv_str = "hand_acc"
    behav[dv_str, "x"] = dv["x"]
    behav[dv_str, "y"] = dv["y"]
    obs_bh.append(dv_str)

    # Labels
    mi_bh = behav.columns
    obs_bh = np.array(mi_bh.get_level_values(0))[::2]
    lbls_bh = [ob.replace("_", " ").capitalize() for ob in obs_bh]
    n_bh = len(lbls_bh)

    ################################################################################
    # For decoding, we limit ourselves to a smaller time interval, during which there is movement. 
    t_min = 0
    t_max = 300
    t = behav.reset_index()["t"] 
    mask_mov = (t >= t_min) * (t < t_max)
    
    # Firing rates (with delay time between firing rates and output).
    t_delays = {
        "hand_pos": -300,
        "hand_vel": -100,
        "hand_acc": -100,
    }
    
    # Join all three output modalities. This will save some time downstream, 
    # because we only need to do the preprocessing once. 
    output_mods = ["hand_pos", "hand_vel", "hand_acc"]
    output_dict = {}
    hids_dict = {}
    for output_mod in output_mods:
        output_i = behav.loc[:, :, :, mask_mov][output_mod]
        t_delay = t_delays[output_mod]
        mask_hids = (t >= t_min + t_delay) * (t < t_max + t_delay)
        hids_i = rates.loc[:, :, :, mask_hids]
        # Save normalized versions
        output_dict[output_mod] = output_i.to_numpy()
        hids_dict[output_mod] = hids_i.to_numpy()
        # Trial-cond avg
        output_i = output_i.groupby(["trial_type", "trial_version", "t"]).mean().to_numpy()
        hids_i = hids_i.groupby(["trial_type", "trial_version", "t"]).mean().to_numpy()
        output_dict[output_mod + "_tca"] = output_i 
        hids_dict[output_mod + "_tca"] = hids_i

    return output_dict, hids_dict

def load_nlb_rtt(dataset_name, 
                verbose=False,
                bin_width_ms=45, #ms (BCI: 45ms, no filter)
                kern_sd_ms=45, #ms (25ms: Russo, but with much higher firing rates)
    ):
    from nlb_tools.nwb_interface import NWBDataset
    import pandas as pd
    filter_rates = kern_sd_ms > 0
    _, nlb_dataset = dataset_name.split('-')

    ## Load data from NWB file
    data_path_nlb = os.path.join(data_path, "neural_latents")
    file_name = '000129'
    monkey_file = "sub-Indy"
    data_file = os.path.join(data_path_nlb, file_name, monkey_file) 
    dataset = NWBDataset(data_file)
    if verbose:
        print("Loaded from ", data_file)
        print("All data")
        print("dataset.data.shape", dataset.data.shape)
    ### Dataset preparation
    # Choose bin width and resample
    dataset.resample(bin_width_ms)
    if verbose:
        print("All data, binned")
        print(dataset.data.shape)
    ################################################################################
    ### Some preprocessing
    # Remove test trials, because these do not contain behavior
    trial_mask = np.array(dataset.trial_info["split"] != "test")
    # Get trial data
    trial_data = dataset.make_trial_data(ignored_trials=~trial_mask, )
    # Free the memory from the full dataset, which we won't use anymore
    del dataset
    # Merge heldout and non-heldout spikes, since we don't distinguish here. 
    trial_data.rename(columns={"heldout_spikes": "spikes"}, inplace=True)
    trial_data = trial_data.reindex(trial_data.columns.sortlevel(0)[0], axis=1)
    # Remove align and trial time. These are nonsense.
    trial_data.drop('trial_time', axis=1, inplace=True)
    trial_data.drop('align_time', axis=1, inplace=True)
    # Distance to target
    d_to_target = (trial_data.cursor_pos - trial_data.target_pos).apply(lambda x: norm(x), axis=1)
    trial_data["d_to_target"] = d_to_target
    # Where do targets change?
    target_pos = trial_data.target_pos.to_numpy()
    ids_shift_target = np.where(norm(target_pos[1:] - target_pos[:-1], axis=-1) != 0)[0] + 1
    # Obtain actual trial times.
    df_list = []
    id_st = ids_shift_target[0]
    for i_trial, id_st_p1 in enumerate(ids_shift_target[1:]):
        tdi = trial_data.iloc[id_st:id_st_p1].copy()
        tdi["trial_time"] = tdi["clock_time"] - tdi["clock_time"].iloc[0]
        tdi["trial_id"] = i_trial
        df_list.append(tdi)
        id_st = id_st_p1
    trial_data = pd.concat(df_list)
    # Multiindex: use trial ID as index
    trial_data['t'] = trial_data["trial_time"] / np.timedelta64(1, "ms")
    trial_data.set_index(["trial_id", "t"], inplace=True)
    # Remove unnecessary columns
    trial_data.drop('margin', axis=1, inplace=True)
    trial_data.drop('trial_time', axis=1, inplace=True)
    # Rates
    if filter_rates:
        # Filter width for rates. It's useful to define this here, so we can set the margins appropriately.
        import scipy.signal as signal
        def filt(x):
            kern_sd = int(round(kern_sd_ms / bin_width_ms))
            window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
            window /= np.sum(window)
            return np.convolve(x, window, 'same')
        rates = trial_data["spikes"].transform(filt).astype(np.float32)
    else:
        rates = trial_data["spikes"].astype(np.float32)
    # In Hz:
    rates = rates / (bin_width_ms * 1e-3)
    if verbose:
        print("rates, binned, removed test trials")
        print("rates.shape", rates.shape)
    ################################################################################
    # Some other definitions that will be useful below.
    trial_ids = trial_data.index.get_level_values(0).unique().to_numpy()
    n_trials = len(trial_ids)
    # Number of channels (neurons recorded)
    n_ch = rates.shape[1]
    # Checks
    if verbose:
        print("Number of channels:", n_ch)
        print("Number of trials:", n_trials)
    ##########################################################################################
    # Behavior: cursor, hand pos, hand vel, eyes
    obs_bh = ["target_pos", "cursor_pos", "finger_pos", "finger_vel"]
    xy_bh = ["x", "y"]
    mi_bh = pd.MultiIndex.from_product([obs_bh, xy_bh])
    behav = trial_data[mi_bh].copy()
    behav["clock_time"] = trial_data["clock_time"]
    behav["d_to_target"] = trial_data["d_to_target"]
    del trial_data
    # Add acceleration to behavior
    vel = behav["finger_vel"]
    dv = vel.groupby(["trial_id"]).diff(1)
    dv_str = "finger_acc"
    behav[dv_str, "x"] = dv["x"]
    behav[dv_str, "y"] = dv["y"]
    obs_bh.append(dv_str)
    # Labels
    behav = behav[["finger_pos", "finger_vel", "finger_acc", "cursor_pos", "target_pos", "clock_time", "d_to_target"]]
    mi_bh = behav.columns
    obs_bh = np.array(mi_bh.get_level_values(0))[::2]
    lbls_bh = [ob.replace("_", " ").capitalize() for ob in obs_bh]
    n_bh = len(lbls_bh)
    
    ################################################################################
    # For decoding, we limit ourselves to a smaller time interval, during which there is movement. 
    t_min = 300
    t_max = 600
    t = behav.reset_index()["t"] 
    dt = int(t[1] - t[0]) 
    mask_mov = (t >= t_min) * (t < t_max)
    ids_mov = np.where(mask_mov)[0]
    
    # Firing rates (with delay time between firing rates and output).
    t_delays = {
        "finger_pos": -300,
        "finger_vel": -100,
        "finger_acc": -150,
    }
        
    # Join all three output modalities. This will save some time downstream, 
    # because we only need to do the preprocessing once. 
    output_mods = ["finger_pos", "finger_vel", "finger_acc"]
    output_dict = {}
    hids_dict = {}
    for output_mod in output_mods:
        output_i = behav.iloc[ids_mov][output_mod].to_numpy()
        t_delay = t_delays[output_mod]
        ids_hids = ids_mov + int(t_delay / dt)
        hids_i = rates.iloc[ids_hids].to_numpy()
        # Save normalized versions
        output_dict[output_mod] = output_i
        hids_dict[output_mod] = hids_i

    return output_dict, hids_dict



