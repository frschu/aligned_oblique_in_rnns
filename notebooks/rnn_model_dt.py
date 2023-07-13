import math
import torch
from torch import nn
import time
from warnings import warn
from collections import OrderedDict
import numpy as np

class RNN_Net(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, 
                 n_layers=2, nonlin=torch.nn.Tanh(), 
                 bias=False, out_scale="small", g=None,
                 gaussian_init=True, 
                 dt=0.2, 
                 rec_step_dt=1,
                 train_layers=None, h_0=None,
                 scale_input_weights=False,
                ):
        super().__init__()
        self.dt = dt
        self.rec_step_dt = rec_step_dt
        self.n_layers = n_layers
        
        # Nonlinearity
        self.nonlin_str = torch.typename(nonlin).split('.')[-1].lower()
        if self.nonlin_str in ["identity"]:
            # torch.nn.Identity leads to an in-place operation somewhere -> error
            self.nonlin = lambda x: x + 0.
        if self.nonlin_str in ["softplus"]:
            # torch.nn.Softplus leads to an in-place operation somewhere -> error
            self.nonlin = lambda x: torch.log(1 + torch.exp(x))
        else:
            self.nonlin = nonlin

        # Define layers
        if self.nonlin_str in ["tanh", "relu"]:
            self.rnn = nn.RNN(dim_in, dim_hid, n_layers - 1, 
                         nonlinearity=self.nonlin_str, bias=bias, batch_first=True)
        else:
            # RNN class only allows for tanh or relu. Define the nonlinearity by hand
            self.rnn = nn.RNN(dim_in, dim_hid, n_layers - 1, 
                         nonlinearity='tanh', bias=bias, batch_first=True)
            
        self.decoder = nn.Linear(dim_hid, dim_out, bias)

        with torch.no_grad():
            # Initialize with normal distributions (uniform has smaller variance!)
            if gaussian_init:
                for key, par in self.named_parameters():
                    if 'weight' in key:
                        n_l = torch.nn.init._calculate_fan_in_and_fan_out(par)[0]
                        std = 1/math.sqrt(n_l)
                        if key.endswith("ih_l0") and not scale_input_weights:
                            std = 1.
                    elif 'bias' in key:
                        std = 0.
                    if 'weight_hh' in key and g is not None:
                        std *= g
                    if std > 0:
                        nn.init.normal_(par, std=std)
                    else:
                        nn.init.zeros_(par)
            # Rescale output layer
            if out_scale == 'small':
                layer_L = self.decoder
                n_L = layer_L.weight.shape[-1]
                layer_L.weight *= 1 / math.sqrt(n_L)
                if bias:
                    layer_L.bias *= 1 / math.sqrt(n_L)
            
        # Set input and output weights non-trainable?
        if train_layers is None:
            train_layers = [True, True, True]
        train_in, train_hid, train_out = train_layers
        if not train_in:
            self.rnn.weight_ih_l0.requires_grad = False
            if bias:
                self.rnn.bias_ih_l0.requires_grad = False
        if not train_hid:
            self.rnn.weight_hh_l0.requires_grad = False
            if bias:
                self.rnn.bias_hh_l0.requires_grad = False
        if not train_out:
            self.decoder.weight.requires_grad = False
            if bias:
                self.decoder.bias.requires_grad = False
                
    def rnn_forward(self, input, h_0=None, noise_std=0.):
        
        if type(noise_std) in [np.ndarray, torch.Tensor]:
            has_noise = True
        else:
            has_noise = noise_std > 0.
        if self.dt == 1 and not has_noise and self.nonlin_str in ["tanh", "relu"]:
            x, hid = self.rnn(input, h_0)
            if self.rnn.batch_first:
                x = x[:, ::self.rec_step_dt]
            else:
                x = x[::self.rec_step_dt]
            return x, hid
        else:
            # Input: set to [seq_len, batch_size, dim_in]
            if self.rnn.batch_first:
                input = input.transpose(1, 0)
            n_t, batch_size, dim_in = input.shape
            # Initial hidden states
            if h_0 is None:
                h_0 = torch.zeros((self.rnn.num_layers, batch_size, self.rnn.hidden_size), 
                                  device=input.device)
            # Noise
            if has_noise:
                if type(noise_std) in [np.ndarray, torch.Tensor]:
                    # Given noise
                    noise = noise_std
                else:
                    noise = math.sqrt(self.dt) * noise_std * torch.randn(
                        batch_size, n_t, self.rnn.hidden_size, device=input.device)
            # Output = the last hidden layer. 
            n_rec = max(int(np.ceil(n_t / self.rec_step_dt)), 1)
            last_hidden_layer = torch.zeros((n_rec, batch_size, self.rnn.hidden_size), device=input.device)
            # Iterate
            h = h_0.clone()
            h_t = h_0[-1] # last layer for saving
            for t, x_t in enumerate(input):
                # Save the last layer for every time step
                if t % self.rec_step_dt == 0:
                    k = t // self.rec_step_dt
                    last_hidden_layer[k] = h_t
                for i_l in range(self.rnn.num_layers):
                    h_t = h[i_l]
                    weight_ih = getattr(self.rnn, "weight_ih_l%d"%i_l)
                    weight_hh = getattr(self.rnn, "weight_hh_l%d"%i_l)
                    f_t = self.nonlin(h_t) @ weight_hh.T + x_t @ weight_ih.T
                    if self.rnn.bias:
                        bias_ih = getattr(self.rnn, "bias_ih_l%d"%i_l)
                        bias_hh = getattr(self.rnn, "bias_hh_l%d"%i_l)
                        add_bias = bias_ih + bias_hh
                        f_t = f_t + add_bias
                    # Update: hidden state in this layer; 
                    h_t = (1 - self.dt) * h_t + self.dt * f_t
                    if has_noise:
                        h_t = h_t + noise[:, t, :]
                    h[i_l] = h_t
                    # Input to the next layer is the **updated** state of this layer
                    x_t = h_t
            if self.rnn.batch_first:
                input = input.transpose(1, 0)
                last_hidden_layer = last_hidden_layer.transpose(1, 0)
            return last_hidden_layer, h
    
    def forward(self, input, h_0=None, noise_std=0.):
        x, _ = self.rnn_forward(input, h_0, noise_std)
        x = self.decoder(x)
        return x

    def forward_hid(self, input, h_0=None, noise_std=0., last_time=False):
        if last_time:
            # Save last time step of input, all hidden layers, and output. 
            # This is needed to compute the scaling dy.
            hid = []
            hid.append(input[:, -1].detach().clone())
        x, last_hidden = self.rnn_forward(input, h_0, noise_std)
        if last_time:
            for hid_i in last_hidden:
                hid.append(hid_i.detach().clone())
        else:
            assert self.rnn.num_layers == 1, "Only the last hidden layer is saved! Deactivate this line in case you nonetheless want to proceed."
            # Only save the (last) hidden layer, but all points in time.
            # Add a dimension for the layer to be consistent with h_0 above.
            hid = x.detach().clone()[None, :]
        x = self.decoder(x)
        if last_time:
            hid.append(x[:, -1].detach().clone())
        return x, hid

def find_fp(net, hidden_guess, input, n_steps=1000, lr=1e-2, verbose=False, noise_std=0):
    """ Find a fixed point of a network. 
    `net` needs to have a hidden layer called net.rnn. Adapt this if necessary...
    """
    # Move to GPU?
    net_device = list(net.parameters())[0].device
    input_device = input.device
    hid_device = hidden_guess.device
    hidden_guess = hidden_guess.to(net_device)
    input = input.to(net_device)
    
    # Check input and guess sizes
    batch_size, n_t, _ = input.shape
    n_layers_hid, batch_size_hid, dim_hid = hidden_guess.shape
    assert n_t == 1, "We only want a single time step! FP finding only makes sense for constant input."
    assert net.rnn.num_layers == n_layers_hid, "Hidden guess must have same number of layers as rnn!"
    assert batch_size == batch_size_hid, "Batch size of input and hidden_guess must agree!"
    assert net.rnn.num_layers == 1, "Only implemented for 1 hidden layer. "
    
    # Set all weights non-trainable, but save their status for reset at the end
    req_grad = OrderedDict()
    for key, par in net.named_parameters():
        req_grad[key] = par.requires_grad
        par.requires_grad = False

    energy_all = torch.zeros((batch_size, n_steps))
    i_choose_all = torch.zeros((batch_size), dtype=int)
    fp = torch.zeros((batch_size, n_layers_hid, dim_hid))
    time0 = time.time()
    for i_b in range(batch_size):
        # Make the hidden state a trainable parameter
        hidden_init = torch.nn.Parameter(hidden_guess[:, i_b:i_b+1])
        opt_energy = torch.optim.Adam([hidden_init], lr=lr)
        loss_crit_energy = torch.nn.MSELoss()

        for i in range(n_steps):
            opt_energy.zero_grad()

            _, hidden = net.rnn_forward(input[i_b:i_b+1], hidden_init, noise_std)
            energy_i = loss_crit_energy(hidden, hidden_init)
            energy_i.backward()
            opt_energy.step()
            energy_i = energy_i.item()

            with torch.no_grad():
                # Save best state
                if i < 2:
                    cond = True
                else:
                    cond_min = energy_i < energy_min 
                    # Only save it if there has been mostly progress in the past
                    # Measure relative slope
                    de = torch.diff(energy_all[i_b, :i]) / energy_all[i_b, :i-1]
                    cond_de = de.max() < 2.
                    cond = cond_min and cond_de
                if cond:
                    energy_min = energy_i
                    i_choose = i
                    fp_i = hidden_init.detach().clone().cpu()

            # Save energy
            energy_all[i_b, i] = energy_i
            i_choose_all[i_b] = i_choose
        fp[i_b] = fp_i
    if verbose: print("Took %.1f sec" % (time.time() - time0))
    
    # Reset network
    for key, par in net.named_parameters():
        par.requires_grad = req_grad[key]
    input = input.to(input_device)
    hidden_guess = hidden_guess.to(hid_device)
        
    return fp, energy_all, i_choose_all



# ############################################################################################
# # Test network model
# dim_in = 5
# dim_out = 7
# dim_hid = 512
# n_layers = 2
# nonlin = torch.nn.Tanh()
# bias = False
# out_scale = 'small'
# g = 2.

# net = RNN_Net(dim_in, dim_hid, dim_out, n_layers, nonlin, bias, out_scale, g)

# init_stds = OrderedDict()
# with torch.no_grad():
#     for key, par in net.named_parameters():
#         if par.requires_grad:
#             init_stds[key] = torch.sqrt((par**2).mean())
# print(init_stds.values())

# batch_size = 32
# rec_step_dt = int(1/dt)
# n_t = 10 * int(1/dt)
# with torch.no_grad():
#     input = torch.randn(batch_size, n_t, dim_in)
#     output = net(input)
# #     output, hid = net.forward_hid(input)
# #     output, hid = net.forward_hid(input)
# #     output, hid = net.forward_hid(input)
    
# # print(input.shape, output.shape, [hi.shape for hi in hid])
# # output = net(input, rec_step_dt=rec_step_dt)
# # # output.shape

# # Compute fixed point of the network
# # Initial guess for hidden state
# n_hidden_layers = net.rnn.num_layers
# hidden_guess = 0.1 * torch.randn((n_hidden_layers, batch_size, dim_hid))
# # Input: only the first time step
# x = input[:, :1]
# fp, energy_all = find_fp(net, hidden_guess, x)

# # Plot energy over optimization
# plt.plot(energy_all)
# plt.axhline(0, c=c_leg)