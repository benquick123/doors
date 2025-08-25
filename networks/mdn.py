import torch as th
import torch.nn as nn
from torch.distributions import MultivariateNormal, OneHotCategorical


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network compatible with full covariance.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    full_cov: bool; whether to use full or diagonal covariance matrix
    """
    
    def __init__(self, 
                 dim_in, 
                 dim_out, 
                 n_components, 
                 hidden_dim, 
                 pi_layers=3,
                 normal_layers=3, 
                 full_cov=True, 
                 mu_bias_init=None,
                 tau_pi=1.0, 
                 lambda_pi=0.0, 
                 lambda_sigma=0.0,
                 lambda_mu=0.0):
        super().__init__()
        
        self.pi_network = CategoricalNetwork(dim_in, 
                                            n_components, 
                                            hidden_dim, 
                                            n_layers=pi_layers, 
                                            tau=tau_pi)
    
        self.normal_network = NormalNetwork(dim_in, 
                                            dim_out, 
                                            hidden_dim, 
                                            n_components, 
                                            n_layers=normal_layers, 
                                            full_cov=full_cov, 
                                            mu_bias_init=mu_bias_init)
        
        self.n_components = n_components
        self.printed_entropy_warning = False
        self.lambda_pi = lambda_pi
        self.lambda_sigma = lambda_sigma
        self.lambda_mu = lambda_mu

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def loss(self, x, y):
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loss = -th.logsumexp(th.log(pi.probs) + loglik, dim=1)
        
        if self.lambda_pi > 0:
            params = th.cat([x.view(-1) for x in self.pi_network.network.parameters()])
            loss += self.lambda_pi * th.linalg.norm(params, ord=2)
        if self.lambda_sigma > 0:
            params = th.cat([x.view(-1) for x in self.normal_network.tril_net.parameters()])
            loss += self.lambda_sigma * th.linalg.norm(params, ord=2)
        if self.lambda_mu > 0:
            params = th.cat([x.view(-1) for x in self.normal_network.mean_net.parameters()])
            loss += self.lambda_mu * th.linalg.norm(params, ord=2)
        
        return loss

    def entropy(self, x):
        pi, normal = self.forward(x)
        if self.n_components == 1:
            return normal.entropy().sum(dim=1)
        else:
            if not self.printed_entropy_warning:
                print("Using entropy ~approaximation~ when n_components > 1.")
                self.printed_entropy_warning = True
            return (pi.probs * normal.entropy()).sum(dim=1)


class NormalNetwork(nn.Module):
    
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 hidden_dim, 
                 n_components, 
                 n_layers, 
                 full_cov=True, 
                 mu_bias_init=None):
        
        super().__init__()
        self.n_components = n_components
        self.out_dim = out_dim
        self.full_cov = full_cov
        
        self.tril_indices = th.tril_indices(row=out_dim, col=out_dim, offset=0)
        self.elu = nn.ELU()
        
        mean_net = []
        for layer_idx in range(n_layers):
            _in = _out = hidden_dim
            if layer_idx == 0:
                _in = in_dim
                
            if layer_idx == n_layers - 2:
                _out = hidden_dim * n_components
                
            if layer_idx == n_layers - 1:
                _in = hidden_dim * n_components
                _out = out_dim * n_components
                
            mean_net.append(nn.Linear(_in, _out))
            mean_net.append(nn.ReLU())
        mean_net = mean_net[:-1]
        
        if mu_bias_init is not None:
            assert mu_bias_init.shape[0] == out_dim * n_components, f"mu_bias_init must have length equal to out_dim; {mu_bias_init.shape[0]} != {out_dim * n_components}."
            assert isinstance(mu_bias_init, th.Tensor), "mu_bias_init must be a torch.Tensor."
            mean_net[-1].bias.data = mu_bias_init.float()
            
        self.mean_net = nn.Sequential(*mean_net)
        
        tril_net = []
        for layer_idx in range(n_layers):
            _in = _out = hidden_dim
            if layer_idx == 0:
                _in = in_dim
            if layer_idx == n_layers - 1:
                _out = int(out_dim * (out_dim + 1) / 2 * n_components) if full_cov else out_dim * n_components
            tril_net.append(nn.Linear(_in, _out))
            tril_net.append(nn.ReLU())
        tril_net = tril_net[:-1]
        self.tril_net = nn.Sequential(*tril_net)

    def forward(self, x):
        mean = self.mean_net(x).reshape(-1, self.n_components, self.out_dim)
        tril_values = self.tril_net(x).reshape(mean.shape[0], self.n_components, -1)
        
        if self.full_cov:
            tril = th.zeros(mean.shape[0], mean.shape[1], mean.shape[2], mean.shape[2], dtype=x.dtype, device=x.device)
            tril[:, :, self.tril_indices[0], self.tril_indices[1]] = tril_values
            # diagonal element must be strictly positive
            # use diag = elu(diag) + 1 to ensure positivity
            tril = tril - th.diag_embed(th.diagonal(tril, dim1=-2, dim2=-1)) +\
                          th.diag_embed(self.elu(th.diagonal(tril, dim1=-2, dim2=-1)) + 1 + 1e-2)
        else:
            tril = th.diag_embed(self.elu(tril_values) + 1 + 1e-2)
            
        return MultivariateNormal(mean, scale_tril=tril)


class CategoricalNetwork(nn.Module):

    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 hidden_dim, 
                 n_layers,
                 tau=1.0):
        
        super().__init__()
        
        if out_dim > 1:
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
        
        self.n_components = out_dim
        
        # tau<1.0: more deterministic, tau>1.0: more stochastic
        self.tau = tau

    def forward(self, x):
        if self.n_components == 1:
            return OneHotCategorical(probs=th.ones(x.shape[0], 1, device=x.device, dtype=x.dtype))
        else:
            params = self.network(x) / self.tau
            return OneHotCategorical(logits=params)
