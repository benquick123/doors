import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DIAYNDiscriminator(nn.Module):
    
    def __init__(self, in_dim, out_dim, hidden_dim, n_layers):
        super().__init__()
        
        network = []
        for layer_idx in range(n_layers):
            _in = _out = hidden_dim
            if layer_idx == 0:
                _in = in_dim
            if layer_idx == n_layers - 1:
                _out = out_dim
                
            network.append(nn.Linear(_in, _out))
            network.append(nn.ReLU())
        network = network[:-1]
        
        self.network = nn.Sequential(*network)
        
    def forward(self, x):
        # returns logits with shape (batch_size, num_skills)
        return self.network(x)
    
    def get_entropy(self, x):
        """
        Compute the entropy of the input tensor.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Entropy of the input tensor.
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * th.log(probs + 1e-10)).sum(dim=1)
        return entropy
    
    def loss(self, x, z):
        """
        Compute the loss for the discriminator.
        Args:
            x (torch.Tensor): Input tensor. Shape (batch_size, obs_dim)
            z (torch.Tensor): Target tensor. Shape (batch_size, num_skills)
        Returns:
            torch.Tensor: Loss value.
        """
        return F.cross_entropy(self.forward(x), z)
    