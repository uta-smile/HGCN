"Disentangle function"
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros


# linearly disentangle node representations
class linearDisentangle(torch.nn.Module):
    def __init__(self, in_dims, out_dims):
        super(linearDisentangle, self).__init__()
        self.weight = Parameter(torch.Tensor(in_dims, out_dims)) 
        self.bias = Parameter(torch.Tensor(out_dims))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias