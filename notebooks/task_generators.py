import numpy as np
from warnings import warn

########################################################################################
def n_points(dim_in, dim_out, dt,
    t_cs=np.array([10]), 
    targets=None,
    input_amp=0., 
    target_amp=1., 
    t_max=None,
    i_in=0,
    i_out=0,
    fixate_idle_channels=False,
    rec_step_dt=1,
    mask_step_dt=1, ### not used!
     ):
    """ 
    Points at t_cs.
    """
    n_in = 1
    n_out = 1
    # Checks
    assert dim_in >= i_in + n_in, "dim_in too small."
    assert dim_out >= i_out + n_out, "dim_out too small."
    
    # Times
    if t_max is None:
        t_max = max(t_cs)
    assert t_max >= max(t_cs)
    
    if targets is None:
        targets = np.ones(len(t_cs))
        
    def task(batch_size):
        # Times (use linspace for better numerical stability)
        n_t_max = int(t_max / dt) + 1
        ts = np.linspace(0, t_max, n_t_max)
        # Input and target sequences
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)
        
        # Reduce the target and mask by rec step
        # Note that the input must not be subsampled, because it's needed for the forward pass!
        target_batch = target_batch[:, ::rec_step_dt]
        mask_batch = mask_batch[:, ::rec_step_dt]
        ts = ts[::rec_step_dt]
        # Target points
        for i, t_i in enumerate(t_cs):
            idx_t_i = np.argmin((ts - t_i)**2)
            target_batch[:, idx_t_i, i_out:i_out+n_out] = targets[i]
            mask_batch[:, idx_t_i, i_out:i_out+n_out] = True
        # Scale input and target
        input_batch *= input_amp
        target_batch *= target_amp
        # Set target of idle channels to zero?
        if fixate_idle_channels:
            mask_batch[:, :, :i_out] = 1
            mask_batch[:, :, i_out+n_out:] = 1
        return ts, input_batch, target_batch, mask_batch
    return task

########################################################################################
def fixed_points(dim_in, dim_out, dt,
    t_max=20, 
    t_loss_min=5,
    t_loss_max=None,
    input_amp=1., 
    target_amp=1., 
    corr_in=0., 
    corr_out=0., 
    idle_channels=False,
    catch_trial_fraction=None,
    choices=None,
    inputs=None, 
    targets=None, 
    only_pos=False,
    test=False,
    n_in=None,
    n_out=None,
    i_in=None,
    i_out=None,
    fixate_idle_channels=False,
    rec_step_dt=1,
    mask_step_dt=1,
     ):
    """ 
    Autonomous fixed point task (without transitions). 
    If `t_loss_min` is None, the loss is evaluated only at the last time point. 
    
    choices: indices to choose inputs and targets
    inputs: [len(choices) x dim_in] array of input values
    targets: [len(choices) x dim_out] array of target values 
        The purpose of 'inputs' and 'targets' is to allow  for correlated tasks. 
        Note that 'inputs' and 'targets' should be normalized, so that the amplitudes are determined 
        by 'input_amp' and 'target_amp', respectively.
    """
    if n_in is None:
        n_in = dim_in
    if i_in is None:
        i_in = 0
    if n_out is None:
        n_out = dim_out
    if i_out is None:
        i_out = 0
    # Checks
    assert dim_in >= i_in + n_in, "dim_in too small."
    assert dim_out >= i_out + n_out, "dim_out too small."
    
    # Pulse initializes the network
    t_pulse = dt
    pulse_amp = input_amp / t_pulse

    # Times
    # Loss interval
    if t_loss_min is None:
        t_loss_min = t_max - dt 
    else:
        t_loss_min = min(t_loss_min, t_max - dt)
    if t_loss_max is None:
        t_loss_max = t_max
    else:
        t_loss_max = max(t_loss_max, t_loss_min + dt)
    # Time steps
    n_t_max = int(t_max / dt)
    n_t_pulse = int(t_pulse / dt)
    n_t_loss_min = int(t_loss_min / dt)
    n_t_loss_max = int(t_loss_max / dt)
        
    # Choice indices
    if choices is None:
        choices = np.arange(n_out)
    n_choices = len(choices)
    
    # Inputs and targets
    assert inputs is None or corr_in == 0
    if inputs is None:
        inputs = np.eye(n_in)
    assert targets is None or corr_out == 0
    if targets is None:
        targets = np.eye(n_out)
    # Input and target correlation
    # We don't want correlated targets or inputs to have one of them aligned with the actual
    # input/output axis, while the other one is rotated towards it. 
    # Rather, we rotate both so that there is not bias. 
    R_mat = lambda ang: np.array(
        [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    if corr_in != 0.:
        assert n_in == 2, "For correlation model, we need two-dimensional input!"
        inputs[1] = corr_in * inputs[0] + np.sqrt(1 - corr_in**2) * inputs[1]
        ang_in = np.arccos(corr_in)
        inputs = inputs @ R_mat(np.pi/4 - ang_in/2).T
    if corr_out != 0.:
        assert n_out == 2, "For correlation model, we need two-dimensional output!"
        targets[1] = corr_out * targets[0] + np.sqrt(1 - corr_out**2) * targets[1]
        ang_out = np.arccos(corr_out)
        targets = targets @ R_mat(np.pi/4 - ang_out/2).T
    # Checks
    assert max(choices) < min(len(inputs), len(targets)), 'Too many choices for inputs and/or targets!'
    assert n_in == inputs.shape[-1], 'Input dimensions dont agree!'
    assert n_out == targets.shape[-1], 'Target and output dimensions dont agree!'
    if not np.allclose(np.linalg.norm(inputs, axis=-1), 1):
        warn("`inputs` are not normalized!")
    if not np.allclose(np.linalg.norm(targets, axis=-1), 1):
        warn("`targets` are not normalized!")
        
    # Scale input and target
    inputs *= pulse_amp
    targets *= target_amp

    def task(batch_size):
        # Input and target sequences
        ts = np.arange(0, t_max, dt)
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)

        for b_idx in range(batch_size):
            input_samp = np.zeros((n_t_max, n_in))
            target_samp = np.zeros((n_t_max, n_out))
            mask_samp = np.zeros((n_t_max, n_out))
            # Choose fixed point. 
            # Notice that this is not a Bit, since we want to be able to train sequentially
            if test and b_idx < 2 * n_choices:
                choice = b_idx % n_choices
                eo = int(b_idx / n_choices)
                sign = (-1)**eo
            else:
                choice = np.random.choice(choices)
                sign = np.random.choice([1, -1])
            if only_pos:
                sign = 1 # Overwrite the sign
            # Input and target
            input_samp[:n_t_pulse] = sign * inputs[choice]
            target_samp[n_t_loss_min:n_t_loss_max] = sign * targets[choice]
            # Catch trials (forcing zero to be stable fixed point)
            if catch_trial_fraction is not None:
                num_catch_trials = int(batch_size * catch_trial_fraction)
                if catch_trial_fraction > 0 and b_idx > (batch_size - num_catch_trials):
                    input_samp = np.zeros((n_t_max, n_in))
                    target_samp = np.zeros((n_t_max, n_out))
            # Mask
            mask_samp[n_t_loss_min:n_t_loss_max] = 1.
            # Mute target for idle dimensions (of the fixed point task only!)
            if idle_channels:
                for i in range(n_out):
                    if not i in choices:
                        mask_samp[:, i] = 0
            # Join
            input_batch[b_idx, :, i_in : i_in + n_in] = input_samp
            target_batch[b_idx, :, i_out : i_out + n_out] = target_samp
            mask_batch[b_idx, :, i_out : i_out + n_out] = mask_samp
        # Set target of idle channels to zero?
        if fixate_idle_channels:
            mask_batch[:, :, :i_out] = 1
            mask_batch[:, :, i_out+n_out:] = 1
        # Reduce the mask by mask_step_dt
        mask_batch_full = mask_batch
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)
        mask_batch[:, ::mask_step_dt] = mask_batch_full[:, ::mask_step_dt]
        # Reduce the target and mask by rec step
        # Note that the input must not be subsampled, because it's needed for the forward pass!
        target_batch = target_batch[:, ::rec_step_dt]
        mask_batch = mask_batch[:, ::rec_step_dt]
        ts = ts[::rec_step_dt]
        return ts, input_batch, target_batch, mask_batch
    return task

########################################################################################
def simple_sine(dim_in, dim_out, dt, 
    t_dec=21, 
    t_fix=0, 
    freq=0.1,
    target_amp=1., 
    input_amp=1., 
    t_max=None, 
    n_in=None,
    i_in=0,
    i_out=0,
    fixate_idle_channels=False,
    rec_step_dt=1,
    mask_step_dt=1,
):
    """ Sine wave with a single frequency (no input, hence needs non-zero initial state)."""
    # Number of input and output dimensions
    if n_in is None:
        n_in = dim_in
    n_out = 1
    # Checks
    assert dim_in >= i_in + n_in, "dim_in too small."
    assert dim_out >= i_out + n_out, "dim_out too small."
    # Trial times: 
    t_max_task = t_fix + t_dec
    if t_max is None or t_max < t_max_task:
        t_max = t_max_task
    n_t_max = int(t_max / dt)
    
    def task(batch_size):
        # Input and target sequences
        ts = np.arange(0, t_max, dt)
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)
        
        # No input
        
        # Target and mask
        target_batch[:, :, i_out:i_out+n_out] = np.sin(2 * np.pi * freq * ts)[None, :, None]
        response_begin = int(t_fix / dt)
        response_end = int(t_fix + t_dec / dt)
        mask_batch[:, response_begin:response_end, i_out:i_out+n_out] = 1
        # Scale by input and target amplitude
        input_batch *= input_amp
        target_batch *= target_amp
        # Set target of idle channels to zero?
        if fixate_idle_channels:
            mask_batch[:, :, :i_out] = 1
            mask_batch[:, :, i_out+n_out:] = 1
        # Reduce the mask by mask_step_dt
        mask_batch_full = mask_batch
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)
        mask_batch[:, ::mask_step_dt] = mask_batch_full[:, ::mask_step_dt]
        # Reduce the target and mask by rec step
        # Note that the input must not be subsampled, because it's needed for the forward pass!
        target_batch = target_batch[:, ::rec_step_dt]
        mask_batch = mask_batch[:, ::rec_step_dt]
        ts = ts[::rec_step_dt]
        return ts, input_batch, target_batch, mask_batch
    return task

########################################################################################
def complex_sine(dim_in, dim_out, dt,
    t_fix=0,
    t_dec=50,
    freq_min = 0.04,
    freq_max = 0.2,
    input_amp=1.,
    target_amp=1.,
    input_offset=0.25, # offset input so the network isn't getting zero input
    t_max=None, 
    i_in=0,
    i_out=0,
    fixate_idle_channels=False,
    test=False,
    rec_step_dt=1,
    mask_step_dt=1,
    ):
    """ 
    Sine wave task from black box and universality paper.
    Note that the max time and corresponding frequencies are different. 
    These are to a degree arbitrary...
    """
    # Number of input and output dimensions
    n_in = 1
    n_out = 1
    # Checks
    assert dim_in >= i_in + n_in, "dim_in too small."
    assert dim_out >= i_out + n_out, "dim_out too small."
    
    # Trial times: 
    if t_max is None:
        t_max = t_fix + t_dec
    ts = np.arange(0, t_max, dt)
    n_t = len(ts)
    n_t_fix = int(t_fix / dt)
    n_t_dec = int(t_dec / dt)
    response_begin = n_t_fix
    response_end = response_begin + n_t_dec
    
    # Input to frequency. x in [0, 1].
    freq_in = lambda x: (freq_max - freq_min) * x + freq_min

    def task(batch_size):
        ts = np.arange(0, t_max, dt)
        n_t_max = len(ts)
        # Relative input amplitudes
        x_min = 0.
        x_max = 1.
        x_in_test = np.r_[
            np.linspace(x_min, x_max, 5), 
            # np.linspace(x_min, x_max, batch_size - 5)]
            np.random.uniform(x_min, x_max, batch_size - 5)]
        
        # Input and target sequences
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        for b_idx in range(batch_size):
            input_samp = np.zeros((n_t_max, n_in))
            target_samp = np.zeros((n_t_max, n_out))
            mask_samp = np.zeros((n_t_max, n_out))
            # Random input amplitude
            if test:
                x_in = x_in_test[b_idx]
            else:
                x_in = np.random.uniform(x_min, x_max)
            # Set input
            input_samp[:] = x_in + input_offset
            # Target: sin starting at the input; frequency given by input * freq_factor
            freq = freq_in(x_in)
            signal = np.sin(2 * np.pi * freq * ts)
            target_samp[:, 0] = signal
            # Mask
            mask_samp[response_begin:response_end] = 1
            # Scale by input and target amplitude
            input_samp *= input_amp
            target_samp *= target_amp
            # Join
            input_batch[b_idx, :, i_in : i_in + n_in] = input_samp
            target_batch[b_idx, :, i_out : i_out + n_out] = target_samp
            mask_batch[b_idx, :, i_out : i_out + n_out] = mask_samp
        # Set target of idle channels to zero?
        if fixate_idle_channels:
            mask_batch[:, :, :i_out] = 1
            mask_batch[:, :, i_out+n_out:] = 1
        # Reduce the mask by mask_step_dt
        mask_batch_full = mask_batch
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)
        mask_batch[:, ::mask_step_dt] = mask_batch_full[:, ::mask_step_dt]
        # Reduce the target and mask by rec step
        # Note that the input must not be subsampled, because it's needed for the forward pass!
        target_batch = target_batch[:, ::rec_step_dt]
        mask_batch = mask_batch[:, ::rec_step_dt]
        ts = ts[::rec_step_dt]
        return ts, input_batch, target_batch, mask_batch
    return task

########################################################################################
def flipflop(dim_in, dim_out, dt,
    n_tasks=2, 
    choices=None,
    t_max=25,
    t_fix_min=0,
    t_fix_max=1,
    t_stim=1,
    t_dec_delay=2,
    t_stim_delay_min=3,
    t_stim_delay_max=10,
    input_amp=1.,
    target_amp=1.,
    fixate=False,
    bit_flip=True,
    test=False,
    i_in=0,
    i_out=0,
    fixate_idle_channels=False,
    rec_step_dt=1,
    mask_step_dt=1,
    ):
    """ 
    Flipflop task. This is different fromt the fixed point task, in that the
    dimensions not receiving the last pulse are supposed to remain at their 
    current value!
    """
    
    # Number of input and output dimensions
    n_in = n_tasks
    n_out = n_in
    # Checks
    assert dim_in >= i_in + n_in, "dim_in too small."
    assert dim_out >= i_out + n_out, "dim_out too small."
    if choices is None:
        choices = np.arange(n_tasks)
    assert np.max(choices) <= (n_tasks - 1), "The max choice must agree with number of tasks!"
        
    # Task times
    n_t_stim = int(t_stim / dt)
    n_t_dec_delay = int(t_dec_delay / dt)
    n_t_max = int(t_max / dt)
    
    def task(batch_size):
        # Input and target sequences
        ts = np.arange(0, t_max, dt)
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)

        for b_idx in range(batch_size):
            input_samp = np.zeros((n_t_max, n_tasks))
            target_samp = np.zeros((n_t_max, n_tasks))
            mask_samp = np.zeros((n_t_max, n_tasks))

            # Initial fixation time
            if test:
                t_fix_mean = 0.5 * (t_fix_min + t_fix_max)
                t_fix_test = [t_fix_mean, t_fix_min, t_fix_max]*(batch_size // 3 + 3)
                t_fix = t_fix_test[b_idx]
            else:
                t_fix = np.random.uniform(t_fix_min, t_fix_max)
            n_t_fix = int(t_fix / dt)
            idx_t = n_t_fix
            if fixate:
                # Mask
                mask_samp[:idx_t] = 1
            
            i_interval = 0
            test_intervals = np.array([16.55, 9.35, 14.80, 11.73, 12.17,  6.50, 13.06, 19.08, 13.19])
            test_signs = np.array([-1, 1, 1, 1, -1, -1, -1, -1, 1])
            if n_tasks == 1:
                test_choices = np.zeros(len(test_signs))
            elif n_tasks == 2:
                test_choices = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1])
            elif n_tasks == 3:
                test_choices = np.array([2, 0, 2, 1, 0, 2, 1, 0, 1])
            
            int_counter = 0
            init_pulses = np.random.choice(choices, n_tasks, replace=False)
            while True:
                # Choose interval
                if int_counter < n_tasks - 1:
                    # Initially, make sure that all inputs are triggered once.
                    interval = dt
                else:
                    if test and b_idx == 0 and i_interval < len(test_choices):
                        interval = test_intervals[i_interval]
                    else:
                        interval = np.random.uniform(t_stim_delay_min, t_stim_delay_max)
                # Choose input. 
                if int_counter < n_tasks:
                    # Make sure there's a target in each direction right away.
                    choice = init_pulses[int_counter]
                    sign = np.random.choice([1, -1])
                else:
                    if test and b_idx == 0 and i_interval < len(test_choices):
                        choice = test_choices[i_interval]
                        sign = test_signs[i_interval]
                    else:
                        choice = np.random.choice(choices)
                        sign = np.random.choice([1, -1])
                    
                
                # Add the decision delay
                interval += t_dec_delay
                # New index
                n_t_interval = int(interval / dt)
                idx_tp1 = idx_t + n_t_interval
                # Input
                input_samp[idx_t : idx_t + n_t_stim, choice] = sign
                # Target and mask
                for i_o in range(n_tasks):
                    if i_o == choice:
                        # New fixed point; with grace period
                        target_samp[idx_t + n_t_dec_delay : idx_tp1, i_o] = sign
                        mask_samp[idx_t + n_t_dec_delay : idx_tp1, i_o] = 1
                    else:
                        if bit_flip:
                            # Leave other channels as is; no grace period
                            target_samp[idx_t : idx_tp1, i_o] = target_samp[idx_t-1, i_o]
                            mask_samp[idx_t : idx_tp1, i_o] = mask_samp[idx_t-1, i_o]
                        else:
                            # Set other channels to zero, with grace period
                            target_samp[idx_t : idx_tp1, i_o] = 0
                            mask_samp[idx_t + n_t_dec_delay : idx_tp1, i_o] = mask_samp[idx_t-1, i_o]
                            
                # Update
                idx_t = idx_tp1
                i_interval += 1
                # Break
                if idx_t > n_t_max: break
                
                int_counter += 1
                    
            # Scale by input and target amplitude
            input_samp *= input_amp
            target_samp *= target_amp
            # Join
            input_batch[b_idx, :, i_in : i_in + n_in] = input_samp
            target_batch[b_idx, :, i_out : i_out + n_out] = target_samp
            mask_batch[b_idx, :, i_out : i_out + n_out] = mask_samp
            
        # Set target of idle channels to zero?
        if fixate_idle_channels:
            mask_batch[:, :, :i_out] = 1
            mask_batch[:, :, i_out+n_out:] = 1
        # Reduce the mask by mask_step_dt
        mask_batch_full = mask_batch
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)
        mask_batch[:, ::mask_step_dt] = mask_batch_full[:, ::mask_step_dt]
        # Reduce the target and mask by rec step
        # Note that the input must not be subsampled, because it's needed for the forward pass!
        target_batch = target_batch[:, ::rec_step_dt]
        mask_batch = mask_batch[:, ::rec_step_dt]
        ts = ts[::rec_step_dt]
        return ts, input_batch, target_batch, mask_batch
    return task
        

########################################################################################
def mante(dim_in, dim_out, dt,
    n_tasks=2,
    choices=None,
    t_fix=3,
    t_stim=20,
    t_delay=5,
    t_dec=20,
    input_amp=1.,
    target_amp=1.,
    context_amp=1., 
    rel_input_std=0.05,
    fixate=True,
    fraction_catch_trails=0.,
    coherences=None,
    test=False,
    t_max=None,
    i_in=0,
    i_out=0,
    fixate_idle_channels=False,
    rec_step_dt=1,
    mask_step_dt=1,
    ):
    """ 
    Mante task.
    """
    
    # Number of input and output dimensions
    n_in = 2 * n_tasks
    n_cont = n_tasks
    n_sens = n_tasks
    n_out = 1
    # Checks
    assert n_in % 2 == 0, "n_in must be even"
    assert dim_in >= i_in + n_in, "dim_in too small."
    assert dim_out >= i_out + n_out, "dim_out too small."

    if choices is None:
        choices = np.arange(n_tasks)
    assert np.max(choices) <= (n_tasks - 1), "The max choice must agree with number of tasks!"
        
    # Task times
    n_t_fix = int(t_fix / dt)
    stim_begin = n_t_fix
    n_t_stim = int(t_stim / dt)
    stim_end = stim_begin + n_t_stim
    n_t_delay = int(t_delay / dt)
    response_begin = stim_end + n_t_delay
    n_t_dec = int(t_dec / dt)
    response_end = response_begin + n_t_dec
    t_max_task = t_fix + t_stim + t_delay + t_dec
    if t_max is None or t_max < t_max_task:
        t_max = t_max_task
    n_t_max = int(t_max / dt)

    if coherences is None:
#         coherences = np.array([-4, -2, -1, 1, 2, 4]) / 4.
        coherences = np.array([-8, -4, -2, -1, 1, 2, 4, 8]) / 8.
#         coherences = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]) / 16.
    elif type(coherences) == list:
        coherences = np.array(coherences)
        
    def task(batch_size):
        # Input and target sequences
        ts = np.arange(0, t_max, dt)
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)
        
        for b_idx in range(batch_size):
            target_samp = np.zeros((n_t_max, n_out))
            mask_samp = np.zeros((n_t_max, n_out))
            input_cont_samp = np.zeros((n_t_max, n_cont))
            
            # Sensory input: noise and coherence
            input_sens_samp = np.random.randn(n_t_max, n_sens) * rel_input_std / np.sqrt(dt)
            
            # Coherence signal and target
            if b_idx < (1 - fraction_catch_trails) * batch_size:
                # Draw random sensory coherences and context
                if test and b_idx == 0:
                    coh_i = coherences[[-3, -1]]
                    context = 1
                else:
                    coh_i = np.random.choice(coherences, n_sens)
                    context = np.random.choice(choices)
                # Set input, context, target
                input_sens_samp[stim_begin:stim_end] += coh_i
                input_cont_samp[stim_begin:stim_end, context] = 1.
                target_samp[response_begin:response_end] = (-1)**int(coh_i[context] < 0)
            # Mask
            mask_samp[response_begin:response_end] = 1
            if fixate:
                mask_samp[:stim_end] = 1
            # Scale by input and target amplitude
            input_cont_samp *= context_amp
            input_sens_samp *= input_amp
            target_samp *= target_amp
            # Join context and sensory input
            input_samp = np.zeros((n_t_max, n_in))
            input_samp[:, :n_cont] = input_cont_samp
            input_samp[:, n_cont:] = input_sens_samp
            # Join
            input_batch[b_idx, :, i_in : i_in + n_in] = input_samp
            target_batch[b_idx, :, i_out : i_out + n_out] = target_samp
            mask_batch[b_idx, :, i_out : i_out + n_out] = mask_samp
        # Set target of idle channels to zero?
        if fixate_idle_channels:
            mask_batch[:, :, :i_out] = 1
            mask_batch[:, :, i_out+n_out:] = 1
        # Reduce the mask by mask_step_dt
        mask_batch_full = mask_batch
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)
        mask_batch[:, ::mask_step_dt] = mask_batch_full[:, ::mask_step_dt]
        # Reduce the target and mask by rec step
        # Note that the input must not be subsampled, because it's needed for the forward pass!
        target_batch = target_batch[:, ::rec_step_dt]
        mask_batch = mask_batch[:, ::rec_step_dt]
        ts = ts[::rec_step_dt]
        
        return ts, input_batch, target_batch, mask_batch
    return task
        
    
########################################################################################
def romo(dim_in, dim_out, dt,
    t_fix_min=1,
    t_fix_max=3,
    t_stim=1,
    t_stim_delay_min=2,
    t_stim_delay_max=12,
    t_dec_delay=4,
    t_dec=8,
    input_amp=1.,
    target_amp=1.,
    rel_input_amp_min=0.5,
    rel_input_amp_max=1.5,
    rel_min_input_diff=0.2,
    fixate=False,
    test=False,
    n_out=1,
    i_in=0,
    i_out=0,
    fixate_idle_channels=False,
    t_max=None,
    rec_step_dt=1,
    mask_step_dt=1,
       ):
    """ 
    Romo task.
    """
    # Number of input and output dimensions
    n_in = 1
    # Checks
    assert dim_in >= i_in + n_in, "dim_in too small."
    assert dim_out >= i_out + n_out, "dim_out too small."

    # Task times
    n_t_stim = int(t_stim / dt)
    n_t_dec_delay = int(t_dec_delay / dt)
    n_t_dec = int(t_dec / dt)
    # After stim_1, there is a random delay...
    t_max_task = (t_fix_max + 2 * t_stim + t_stim_delay_max
             + t_dec_delay + t_dec)
    if t_max is None or t_max < t_max_task:
        t_max = t_max_task
    n_t_max = int(t_max / dt)
    
    def task(batch_size):
        # Input and target sequences
        ts = np.arange(0, t_max, dt)
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)
        
        for b_idx in range(batch_size):
            input_samp = np.zeros((n_t_max, n_in))
            target_samp = np.zeros((n_t_max, n_out))
            mask_samp = np.zeros((n_t_max, n_out))
            # Fixation and delay intervals
            if test:
                stim_delay_mean = 0.5 * (t_stim_delay_min + t_stim_delay_max)
                stim_delay_test = [stim_delay_mean, t_stim_delay_min, t_stim_delay_max]*(batch_size // 3 + 3)
                stim_delay = stim_delay_test[b_idx]
                t_fix_mean = 0.5 * (t_fix_min + t_fix_max)
                t_fix_test = [t_fix_mean, t_fix_min, t_fix_max]*(batch_size // 3 + 3)
                t_fix = t_fix_test[b_idx]
            else:
                stim_delay = np.random.uniform(t_stim_delay_min, t_stim_delay_max)
                t_fix = np.random.uniform(t_fix_min, t_fix_max)
            # Set indices
            n_t_fix = int(t_fix / dt)
            stim_0_begin = n_t_fix
            stim_0_end = stim_0_begin + n_t_stim
            n_stim_delay = int(stim_delay / dt)
            stim_1_begin = stim_0_end + n_stim_delay
            stim_1_end = stim_1_begin + n_t_stim
            response_begin = stim_1_end + n_t_dec_delay
            response_end = response_begin + n_t_dec
            # Input amplitudes
            if test:
                rel_input_amps_test = [[1.1, 0.6], [1.0, 1.4], [0.5, 0.7], [0.9, 1.1]
                                 ]*(batch_size // 4 + 4)
                rel_input_amps = rel_input_amps_test[b_idx]
            else:
                while True:
                    rel_input_amps = np.random.uniform(
                        rel_input_amp_min, rel_input_amp_max, size=2)
                    rel_input_diff = np.abs(rel_input_amps[0] - rel_input_amps[1])
                    if rel_input_diff >= rel_min_input_diff:
                        break
            i_larger_input = np.argmax(rel_input_amps)
            # Set input, target
            input_samp[stim_0_begin:stim_0_end] = rel_input_amps[0]
            input_samp[stim_1_begin:stim_1_end] = rel_input_amps[1]
            if n_out==1:
                target_sign = (-1)**i_larger_input
                target_samp[response_begin:response_end] = target_sign
            else:
                target_samp[response_begin:response_end, i_larger_input] = 1
            # Mask
            mask_samp[response_begin:response_end] = 1
            if fixate:
                # Set target output to zero until the decision delay
                mask_samp[:stim_1_end] = 1
            # Scale by input and target amplitude
            input_samp *= input_amp
            target_samp *= target_amp
            # Join
            input_batch[b_idx, :, i_in : i_in + n_in] = input_samp
            target_batch[b_idx, :, i_out : i_out + n_out] = target_samp
            mask_batch[b_idx, :, i_out : i_out + n_out] = mask_samp
        # Set target of idle channels to zero?
        if fixate_idle_channels:
            mask_batch[:, :, :i_out] = 1
            mask_batch[:, :, i_out+n_out:] = 1
        # Reduce the mask by mask_step_dt
        mask_batch_full = mask_batch
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)
        mask_batch[:, ::mask_step_dt] = mask_batch_full[:, ::mask_step_dt]
        # Reduce the target and mask by rec step
        # Note that the input must not be subsampled, because it's needed for the forward pass!
        target_batch = target_batch[:, ::rec_step_dt]
        mask_batch = mask_batch[:, ::rec_step_dt]
        ts = ts[::rec_step_dt]
        return ts, input_batch, target_batch, mask_batch
    return task



###############################################################################################
def ready_set_go(dim_in, dim_out, dt,
    t_fix_min=1,
    t_fix_max=3,
    t_stim=1,
    t_int_min=5,
    t_int_max=25,
    t_dec=10,
    input_amp=1.,
    target_amp=1.,
    task_type="gauss",
    fixate=True,
    t_max=None,
    test=False,
    n_in=1,
    i_in=0,
    i_out=0,
    fixate_idle_channels=False,
    rec_step_dt=1,
    mask_step_dt=1,
    ):
    """ 
    Ready-set-go task. Pulses or ramp.
    """
    # Number of input and output dimensions
    n_out = 1
    # Checks
    assert dim_in >= i_in + n_in, "dim_in too small."
    assert dim_out >= i_out + n_out, "dim_out too small."

    # Task times
    n_t_stim = int(t_stim / dt)
    n_t_dec = int(t_dec / dt)
    
    # After stim_1, there is a random delay..
    t_max_task = (t_fix_max
                  + 2 * t_stim 
                  + 2 * t_int_max 
                  + t_dec)
    if t_max is None or t_max < t_max_task:
        t_max = t_max_task
    n_t_max = int(t_max / dt)
    
    
    def task(batch_size):
        # Input and target sequences
        ts = np.arange(0, t_max, dt)
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        
        if test:
            n_test_pts = 5
            t_int_mean = 0.5 * (t_int_min + t_int_max)
            t_int_test = np.r_[
                np.linspace(t_int_min, t_int_max, n_test_pts),
                [t_int_mean]*n_test_pts,
                np.random.uniform(t_int_min, t_int_max, batch_size-n_test_pts*2)]
            t_fix_mean = 0.5 * (t_fix_min + t_fix_max)
            t_fix_test = np.r_[
                [t_fix_mean]*n_test_pts,
                np.linspace(t_fix_min, t_fix_max, n_test_pts), 
                np.random.uniform(t_fix_min, t_fix_max, batch_size-n_test_pts*2)]
        
        for b_idx in range(batch_size):
            input_samp = np.zeros((n_t_max, n_in))
            target_samp = np.zeros((n_t_max, n_out))
            mask_samp = np.zeros((n_t_max, n_out))
            
            # Intervals
            if test:
                stim_interval = t_int_test[b_idx]
                t_fix = t_fix_test[b_idx]
            else:
                stim_interval = np.random.uniform(t_int_min, t_int_max)
                t_fix = np.random.uniform(t_fix_min, t_fix_max)
            # Set indices
            n_t_fix = int(t_fix / dt)
            n_t_int = int(stim_interval / dt)
            stim_0_begin = n_t_fix
            stim_0_end = stim_0_begin + n_t_stim
            stim_1_begin = stim_0_end + n_t_int
            stim_1_end = stim_1_begin + n_t_stim
            # Set input, target
            input_samp[stim_0_begin:stim_0_end, 0] = 1.
            input_samp[stim_1_begin:stim_1_end, -1] = 1.
            
            if task_type in ['pulse', 'unconstrained']:
                # Square pulse
                response_begin = stim_1_end + n_t_int
                response_end = response_begin + n_t_stim
                target_samp[response_begin:response_end] = 1.
            elif task_type == 'gauss':
                # Gaussian pulse
                t_pulse = 10
                sigma = t_pulse / 16
                n_t_pulse = int(t_pulse / dt)
                ts_pulse = np.arange(n_t_pulse) * dt
                gauss = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(ts_pulse - t_pulse/2)**2 / (2 * sigma**2))
                response_begin = stim_1_end + n_t_int - n_t_pulse//2
                response_end = response_begin + n_t_pulse
                target_samp[response_begin:response_end, 0] = gauss
            elif task_type == 'ramp':
                n_t_ramp = n_t_int + n_t_stim//2
                ramp = np.linspace(0, 1, n_t_ramp)
                response_end = stim_1_end + n_t_ramp
                target_samp[stim_1_end:response_end, 0] = ramp
                target_samp[response_end:] = 1
            # Mask
            if task_type == 'unconstrained':
                # Pulse, but unconstriained rise and fall
                t_no_loss = 3 # Time 
                assert t_no_loss < n_t_dec, 'No fall demanded!'
                n_t_no_loss = int(t_no_loss / dt)
                peak_begin = stim_1_end + n_t_int
                peak_end = peak_begin + n_t_stim
                mask_pre_begin = stim_1_end
                mask_pre_end = peak_begin - n_t_no_loss
                mask_post_begin = peak_end + n_t_no_loss
                mask_post_end = peak_end + n_t_dec
                mask_samp[mask_pre_begin:mask_pre_end] = 1
                mask_samp[peak_begin:peak_end] = 1
                mask_samp[mask_post_begin:mask_post_end] = 1
            else:
                mask_end = response_end + n_t_dec
                mask_samp[stim_1_end:mask_end] = 1
            if fixate:
                # Set target output to zero until the decision delay
                mask_samp[:stim_1_end] = 1

            # Scale by input and target amplitude
            input_samp *= input_amp
            target_samp *= target_amp

            # Join
            input_batch[b_idx, :, i_in : i_in + n_in] = input_samp
            target_batch[b_idx, :, i_out : i_out + n_out] = target_samp
            mask_batch[b_idx, :, i_out : i_out + n_out] = mask_samp
    
        # Set target of idle channels to zero?
        if fixate_idle_channels:
            mask_batch[:, :, :i_out] = 1
            mask_batch[:, :, i_out+n_out:] = 1
        # Reduce the mask by mask_step_dt
        mask_batch_full = mask_batch
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)
        mask_batch[:, ::mask_step_dt] = mask_batch_full[:, ::mask_step_dt]
        # Reduce the target and mask by rec step
        # Note that the input must not be subsampled, because it's needed for the forward pass!
        target_batch = target_batch[:, ::rec_step_dt]
        mask_batch = mask_batch[:, ::rec_step_dt]
        ts = ts[::rec_step_dt]
        return ts, input_batch, target_batch, mask_batch
    return task
    
########################################################################################
def cycling(dim_in, dim_out, dt,
    t_fix=0,
    t_stim=1,
    t_delay=1,
    t_dec=71,
    freq=0.1,
    target_amp=1.,
    input_amp=1.,
    t_max=None,
    ctx_pulse=True,
    i_in=0,
    i_out=0,
    fixate_idle_channels=False,
    rec_step_dt=1,
    mask_step_dt=5,
    ):
    """ 
    Rotation in two directions. After Russo et al., 2018
    
    If ctx_pulse is True, then the context input is only shown as a pulse
    at the beginning of the trial, and the network needs to generate two 
    separate limit cycles (default).
    Else, there is a constant context signal, and the network can in principle
    reuse the same limit cycle. This then overwrites `t_stim`.
    """
    n_in = 2
    n_out = 2
    
    # Trial times: 
    if t_max is None:
        t_max = t_fix + t_delay + t_dec
    assert t_max >= t_fix + t_delay + t_dec
    n_t_max = round(t_max / dt)
    n_t_fix = int(t_fix / dt)
    n_t_stim = int(t_stim / dt)
    n_t_delay = int(t_delay / dt)
    n_t_dec = int(t_dec / dt)
    stim_begin = n_t_fix
    if ctx_pulse:
        stim_end = stim_begin + n_t_stim
    else:
        stim_end = n_t_max
    
    response_begin = stim_begin + n_t_delay
    response_end = response_begin + n_t_dec
    
    # Frequency
    freq_T = freq * t_dec
    
    def task(batch_size):
        assert batch_size % n_in == 0
        # Input and target sequences
        ts = np.arange(0, t_max, dt)
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        
        if input_amp > 0:
            input_batch[:, stim_begin:stim_end, i_in:i_in+n_in] = (
                np.tile(np.eye(n_in), batch_size//n_in).T.reshape(batch_size, n_in)[:, None, :])
        # Target and mask
        arg = 2 * np.pi * freq * ts[:n_t_dec]
        for i_b in range(batch_size):
            sign = (-1)**i_b
            target_batch[i_b, response_begin:response_end, i_out] = np.sin(sign * arg)
            target_batch[i_b, response_begin:response_end, i_out+1] = np.cos(arg)
        mask_batch[:, response_begin:response_end] = 1
        # Scale
        input_batch *= input_amp
        target_batch *= target_amp
        # Set target of idle channels to zero?
        if fixate_idle_channels:
            mask_batch[:, :, :i_out] = 1
            mask_batch[:, :, i_out+n_out:] = 1
        # Reduce the mask by mask_step_dt
        mask_batch_full = mask_batch
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=bool)
        mask_batch[:, ::mask_step_dt] = mask_batch_full[:, ::mask_step_dt]
        # Reduce the target and mask by rec step
        # Note that the input must not be subsampled, because it's needed for the forward pass!
        target_batch = target_batch[:, ::rec_step_dt]
        mask_batch = mask_batch[:, ::rec_step_dt]
        ts = ts[::rec_step_dt]
        return ts, input_batch, target_batch, mask_batch
    return task
    
    
    
