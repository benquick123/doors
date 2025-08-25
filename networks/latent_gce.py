import torch as th
from torch import nn


class LatentGCE(nn.Module):
    
    def __init__(self,
                 obs_dim,
                 action_dim,
                 A_hidden_dim=512, bias_hidden_dim=256,
                 A_n_layers=3, bias_n_layers=3,
                 action_encoder=False, action_encoder_output_dim=32):
        super().__init__()
        
        if action_encoder:
            A_network_output_dim = action_encoder_output_dim
            # we reuse network hyperparams from A_network
            self.action_encoder = ActionEncoder(action_dim, A_hidden_dim, A_n_layers, action_encoder_output_dim)
        else:
            A_network_output_dim = action_dim
            self.action_encoder = lambda x : x
        
        self.A_network = ANetwork(obs_dim, A_hidden_dim, A_n_layers, A_network_output_dim * obs_dim)
        self.bias_network = BiasNetwork(obs_dim, bias_hidden_dim, bias_n_layers)
        
        self.loss_fn = nn.MSELoss()
        
    def forward(self, obs, actions):
        A = self.A_network(obs)
        bias = self.bias_network(obs)
        
        # actions.shape = (batch_size, action_dim) -> (batch_size, action_dim, 1)
        # maybe actions.shape = (batch_size, action_dim) -> (batch_size, action_encoer_output_dim)
        actions = self.action_encoder(actions)
        actions = actions.unsqueeze(-1)
        
        # A.shape = (batch_size, obs_dim, action_dim)
        # th.bmm().shape = (batch_size, obs_dim, 1) -> (batch_size, obs_dim)
        # bias.shape = (batch_size, obs_dim)
        return th.bmm(A, actions).squeeze(-1) + bias
    
    def forward_svd(self, obs, use_cpu=True):
        A = self.A_network(obs)
        
        if use_cpu:
            A = A.to("cpu")
        
        _, singular_values, _ = th.svd(A, some=True, compute_uv=False) # returns U, singular_values, V
        
        return singular_values
    
    def loss(self, obs, actions, target):
        return self.loss_fn(self.forward(obs, actions), target)
    

class ActionEncoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super().__init__()
        
        network = []
        for layer_idx in range(n_layers):
            _in = _out = hidden_dim
            if layer_idx == 0:
                _in = input_dim
            if layer_idx == n_layers - 1:
                _out = output_dim
                
            network.append(nn.Linear(_in, _out))
            network.append(nn.ReLU())
        network = network[:-1]
        
        self.network = nn.Sequential(*network)
        
    def forward(self, x):
        return self.network(x)
    
    
class ANetwork(nn.Module):
    
    def __init__(self, obs_dim, hidden_dim, n_layers, out_dim):
        super().__init__()
        
        network = []
        for layer_idx in range(n_layers):
            _in = _out = hidden_dim
            if layer_idx == 0:
                _in = obs_dim
            if layer_idx == n_layers - 1:
                _out = out_dim
                
            network.append(nn.Linear(_in, _out))
            network.append(nn.ReLU())
        network = network[:-1]
        
        self.network = nn.Sequential(*network)
        
    def forward(self, x):
        assert len(x.shape) == 2, f"Input shape must be (batch_size, obs_dim); got {x.shape}."
        y = self.network(x)
        return y.reshape(*x.shape, -1) # last dimension should be action dim
    

class BiasNetwork(nn.Module):
    
    def __init__(self, obs_dim, hidden_dim, n_layers):
        super().__init__()
        
        network = []
        for layer_idx in range(n_layers):
            _in = _out = hidden_dim
            if layer_idx == 0:
                _in = obs_dim
            if layer_idx == n_layers - 1:
                _out = obs_dim
                
            network.append(nn.Linear(_in, _out))
            network.append(nn.ReLU())
        network = network[:-1]
        
        self.network = nn.Sequential(*network)
        
    def forward(self, x):
        return self.network(x)