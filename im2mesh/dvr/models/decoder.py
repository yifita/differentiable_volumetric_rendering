import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import ResnetBlockFC


# class Decoder(nn.Module):
#     ''' Decoder class.

#     As discussed in the paper, we implement the OccupancyNetwork
#     f and TextureField t in a single network. It consists of 5
#     fully-connected ResNet blocks with ReLU activation.

#     Args:
#         dim (int): input dimension
#         z_dim (int): dimension of latent code z
#         c_dim (int): dimension of latent conditioned code c
#         hidden_size (int): hidden size of Decoder network
#         leaky (bool): whether to use leaky ReLUs
#         n_blocks (int): number of ResNet blocks
#         out_dim (int): output dimension (e.g. 1 for only
#             occupancy prediction or 4 for occupancy and
#             RGB prediction)
#     '''

#     def __init__(self, dim=3, c_dim=128,
#                  hidden_size=512, leaky=False, n_blocks=5, out_dim=4):
#         super().__init__()
#         self.c_dim = c_dim
#         self.n_blocks = n_blocks
#         self.out_dim = out_dim

#         # Submodules
#         self.fc_p = nn.Linear(dim, hidden_size)
#         self.fc_out = nn.Linear(hidden_size, out_dim)

#         if c_dim != 0:
#             self.fc_c = nn.ModuleList([
#                 nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
#             ])

#         self.blocks = nn.ModuleList([
#             ResnetBlockFC(hidden_size) for i in range(n_blocks)
#         ])

#         if not leaky:
#             self.actvn = F.relu
#         else:
#             self.actvn = lambda x: F.leaky_relu(x, 0.2)

#     def forward(self, p, c=None, batchwise=True, only_occupancy=False,
#                 only_texture=False, **kwargs):

#         assert((len(p.shape) == 3) or (len(p.shape) == 2))

#         net = self.fc_p(p)
#         for n in range(self.n_blocks):
#             if self.c_dim != 0 and c is not None:
#                 net_c = self.fc_c[n](c)
#                 if batchwise:
#                     net_c = net_c.unsqueeze(1)
#                 # use plus is quite unconventional
#                 # conditonal batch normalization like spade?
#                 net = net + net_c

#             net = self.blocks[n](net)

#         out = self.fc_out(self.actvn(net))

#         if only_occupancy:
#             if len(p.shape) == 3:
#                 out = out[:, :, 0]
#             elif len(p.shape) == 2:
#                 out = out[:, 0]
#         elif only_texture:
#             if len(p.shape) == 3:
#                 out = out[:, :, 1:4]
#             elif len(p.shape) == 2:
#                 out = out[:, 1:4]

#         out = out.squeeze(-1)
#         return out


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, dim, out_dim, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.dim = dim
        self.linear = nn.Linear(dim, out_dim, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.dim,
                                            1 / self.dim)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.dim) / self.omega_0,
                                            np.sqrt(6 / self.dim) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Decoder(nn.Module):
    def __init__(self, dim: int, hidden_size: int = 256,
                 n_layers: int = 3, out_dim: int = 4,
                 outermost_linear: bool = True, c_dim: int = 256,
                 first_omega_0: float = 30,
                 hidden_omega_0: float = 30.,
                 activation: str = None,
                 **kwargs,
                 ):
        """
        Args:
            dim: first input dimension
            hidden_size: intermediate feature dimension
            n_layers: number of hidden layers (total number of layers = n_layers + 2)
            out_dim: last output dimension
            outermost_linear: use linear layer as the last layer instead of sine layer
            activation: for the sdf value
        """
        super().__init__()
        self.out_dim = out_dim
        self.dim = dim
        self.c_dim = c_dim

        self.net = []
        self.net.append(SineLayer(dim + c_dim, hidden_size,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(n_layers):
            self.net.append(SineLayer(hidden_size, hidden_size,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_size, self.out_dim)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_size) / hidden_omega_0,
                                             np.sqrt(6 / hidden_size) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_size, self.out_dim,
                                      is_first=False, omega_0=hidden_omega_0))

        self.use_activation = activation is not None

        if self.use_activation:
            self.last_activation = get_class_from_string(activation)()
            self.net.append(self.last_activation)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords, c=None, only_occupancy=False, only_texture=False, **kwargs):
        """
        Args:
            coords: input coordinates (N, *, dim)
            c (tensor): code (N, 1, c_dim)
        Returns:
            (N, *, out_dim)
        """
        # coords = coords.clone().detach().requires_grad_(
        #     True)  # allows to take derivative w.r.t. input
        if c is not None and c.numel() > 0:
            assert(coords.ndim == c.ndim)
            coords = torch.cat([c, coords], dim=-1)

        output = self.net(coords)
        output[..., 1:4] = (output[..., 1:4] + 1)/2.0
        if only_occupancy:
            output = output[..., 0]
        elif only_texture:
            output = output[..., 1:4]
        return output