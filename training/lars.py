"""
LARS (Layer-wise Adaptive Rate Scaling) optimizer wrapper.
Based on YugeTen's ADIOS implementation.

Reference: https://arxiv.org/abs/1708.03888
"""

import torch


class LARSWrapper:
    """
    LARS optimizer wrapper for SGD.
    
    Scales learning rate per layer based on the ratio of 
    parameter norm to gradient norm.
    
    Args:
        optimizer: Base optimizer (typically SGD)
        eta: LARS coefficient (default: 0.02)
        clip: Whether to clip the LARS ratio (default: True)
        eps: Small constant for numerical stability (default: 1e-8)
        exclude_bias_n_norm: Exclude bias and normalization params (default: True)
    """
    
    def __init__(
        self, 
        optimizer, 
        eta=0.02, 
        clip=True, 
        eps=1e-8, 
        exclude_bias_n_norm=True
    ):
        self.optim = optimizer
        self.eta = eta
        self.eps = eps
        self.clip = clip
        self.exclude_bias_n_norm = exclude_bias_n_norm
        
        # Transfer optimizer methods
        self.state_dict = self.optim.state_dict
        self.load_state_dict = self.optim.load_state_dict
        self.zero_grad = self.optim.zero_grad
        self.add_param_group = self.optim.add_param_group

    @property
    def defaults(self):
        return self.optim.defaults

    @property
    def state(self):
        return self.optim.state

    @property
    def param_groups(self):
        return self.optim.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step with LARS scaling.
        """
        weight_decays = []
        
        for group in self.optim.param_groups:
            # Store and temporarily remove weight decay
            weight_decay = group.get("weight_decay", 0)
            weight_decays.append(weight_decay)
            group["weight_decay"] = 0
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                # Skip bias and norm parameters if configured
                if p.ndim == 1 and self.exclude_bias_n_norm:
                    continue
                
                # Compute LARS scaling factor
                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(p.grad.data)
                
                if param_norm > 0 and grad_norm > 0:
                    # LARS ratio
                    lars_ratio = self.eta * param_norm / (
                        grad_norm + weight_decay * param_norm + self.eps
                    )
                    
                    # Optionally clip the ratio
                    if self.clip:
                        lars_ratio = min(lars_ratio, 10.0)
                    
                    # Scale the gradient
                    p.grad.data.mul_(lars_ratio)
                    
                    # Apply weight decay manually (decoupled)
                    if weight_decay > 0:
                        p.grad.data.add_(p.data, alpha=weight_decay * lars_ratio)
        
        # Perform the actual optimization step
        loss = self.optim.step(closure)
        
        # Restore weight decay values
        for group, wd in zip(self.optim.param_groups, weight_decays):
            group["weight_decay"] = wd
            
        return loss