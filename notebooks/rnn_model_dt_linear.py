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
        
        # RNN class only allows for tanh or relu. Define the nonlinearity by hand
        self.rnn = nn.RNN(dim_in, int(dim_hid), n_layers - 1, 
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
        input = input.type(torch.float32)
        
        if type(noise_std) in [np.ndarray, torch.Tensor]:
            has_noise = True
        else:
            has_noise = noise_std > 0.
            
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
        
        ### Iterating over multiple layers does not work any more.
        assert self.rnn.num_layers == 1
        # Iterate
        # h = h_0.detach().clone()
        h_t = h_0.detach().clone()[0] 
        i_l = 0
        weight_ih = getattr(self.rnn, "weight_ih_l%d"%i_l)
        weight_hh = getattr(self.rnn, "weight_hh_l%d"%i_l)
        if self.rnn.bias:
            bias_ih = getattr(self.rnn, "bias_ih_l%d"%i_l)
            bias_hh = getattr(self.rnn, "bias_hh_l%d"%i_l)
            
        for t, x_t in enumerate(input):
            # Save the last layer for every time step
            if t % self.rec_step_dt == 0:
                k = t // self.rec_step_dt
                last_hidden_layer[k] = h_t
            f_t = h_t @ weight_hh.T + x_t @ weight_ih.T
            if self.rnn.bias:
                add_bias = bias_ih + bias_hh
                f_t = f_t + add_bias
            # Update: hidden state in this layer; 
            h_t = (1 - self.dt) * h_t + self.dt * f_t
            if has_noise:
                h_t = h_t + noise[:, t, :]
            # Input to the next layer is the **updated** state of this layer
            x_t = h_t
        
#         # Iterate
#         h = h_0.detach().clone()
#         h_t = h_0.detach().clone()[-1] # last layer for saving
#         for t, x_t in enumerate(input):
#             # Save the last layer for every time step
#             if t % self.rec_step_dt == 0:
#                 k = t // self.rec_step_dt
#                 last_hidden_layer[k] = h_t
#             for i_l in range(self.rnn.num_layers):
#                 h_t = h[i_l]
#                 weight_ih = getattr(self.rnn, "weight_ih_l%d"%i_l)
#                 weight_hh = getattr(self.rnn, "weight_hh_l%d"%i_l)
#                 f_t = h_t @ weight_hh.T + x_t @ weight_ih.T
#                 if self.rnn.bias:
#                     bias_ih = getattr(self.rnn, "bias_ih_l%d"%i_l)
#                     bias_hh = getattr(self.rnn, "bias_hh_l%d"%i_l)
#                     add_bias = bias_ih + bias_hh
#                     f_t = f_t + add_bias
#                 # Update: hidden state in this layer; 
#                 h_t = (1 - self.dt) * h_t + self.dt * f_t
#                 if has_noise:
#                     h_t = h_t + noise[:, t, :]
#                 h[i_l] = h_t
#                 # Input to the next layer is the **updated** state of this layer
#                 x_t = h_t
                
        if self.rnn.batch_first:
            input = input.transpose(1, 0)
            last_hidden_layer = last_hidden_layer.transpose(1, 0)
        return last_hidden_layer, h_t
    
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
            for hid_i in last_hidden[None]:
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
