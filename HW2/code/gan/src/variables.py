import torch

from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim=2, output_dim=2, hidden_dim=100):
        super().__init__()
        # TODO
        raise NotImplementedError("Implement a single hidden layer MLP with ReLU below. See torch.nn.Sequential docs for a start")
        self.inner = NotImplemented

    def forward(self, z):
        """
        Evaluate on a sample. The variable z contains one sample per row
        """
        return self.inner(z)


class DualVariable(nn.Module):
    def __init__(self, input_dim=2,hidden_dim=100, c=1e-2):
        super().__init__()
        self.c=c
        # TODO
        raise NotImplementedError("Implement a single hidden layer MLP with ReLU below. See torch.nn.Sequential docs for a start")
        self.inner=NotImplemented

    def forward(self, x):
        """
        Evaluate on a sample. The variable x contains one sample per row
        """
        return self.inner(x)

    def enforce_lipschitz(self):
        """Enforce the 1-Lipschitz condition of the function by doing weight clipping or spectral normalization"""
        self.spectral_normalisation() # <= you have to implement this one
        #self.weight_clipping() <= this one is for another year/only for you as a bonus if you want to compare
    def spectral_normalisation(self):
        """
        Perform spectral normalisation, forcing the singular value of the weights to be upper bounded by 1.
        """
        raise NotImplementedError("Your code here")

    def weight_clipping(self):
        """
        Clip the parameters to $-c,c$. You can access a modules parameters via self.parameters().
        Remember to access the parameters  in-place and outside of the autograd with Tensor.data.
        """
        with torch.no_grad():
            for p in self.parameters():
                # TODO
                raise NotImplementedError("Clip things here")

