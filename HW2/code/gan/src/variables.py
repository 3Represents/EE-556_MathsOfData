import torch
from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, noise_dim=2, output_dim=2, hidden_dim=100):
        super().__init__()
        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        """
        Evaluate on a sample. The variable z contains one sample per row
        """
        z = F.relu(self.fc1(z))
        return self.fc2(z)


class DualVariable(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=100, c=1e-2): # Add output_dim
        super().__init__()
        self.c = c
        self.u = [torch.rand(hidden_dim, 1), torch.rand(output_dim, 1)]
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Evaluate on a sample. The variable x contains one sample per row
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def enforce_lipschitz(self):
        """Enforce the 1-Lipschitz condition of the function by doing weight clipping or spectral normalization"""
        self.spectral_normalisation() # <= you have to implement this one
        #self.weight_clipping() <= this one is for another year/only for you as a bonus if you want to compare

    def spectral_normalisation(self):
        """
        Perform spectral normalisation, forcing the singular value of the weights to be upper bounded by 1.
        """
        Y1, Y2 = self.fc1.weight.detach(), self.fc2.weight.detach()

        v1, v2 = Y1.t() @ self.u[0], Y2.t() @ self.u[1]
        v1 /= v1.norm()
        v2 /= v2.norm()
        self.u[0], self.u[1] = Y1 @ v1, Y2 @ v2
        for u in self.u:
            u /= u.norm()

        Y1 /= self.u[0].t() @ Y1 @ v1
        Y2 /= self.u[1].t() @ Y2 @ v2
        self.fc1.weight.data, self.fc2.weight.data = Y1, Y2

    def weight_clipping(self):
        """
        Clip the parameters to $-c,c$. You can access a modules parameters via self.parameters().
        Remember to access the parameters in-place and outside of the autograd with Tensor.data.
        """
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp(-self.c, self.c)

