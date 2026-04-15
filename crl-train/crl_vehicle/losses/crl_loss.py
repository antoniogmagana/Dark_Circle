import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(x_hat, x):
    """
    MSE reconstruction loss over filterbank log-envelopes.
    
    x_hat, x: (B, K, T')
    Returns scalar mean loss.
    """
    return F.mse_loss(x_hat, x, reduction="mean")

def kl_divergence(mu, log_var, beta=1.0):
    """
    KL(q(z|x) || N(0, I)) per dimension, averaged over batch.
    
    mu, log_var: (B, d_z)
    Returns scalar.
    """
    return beta * 0.5 * (torch.exp(log_var) + mu**2 - 1 - log_var).sum(dim=-1).mean()

def intervention_matching_loss(interv_logits, interv_targets):
    """
    standard cross-entropy betwen intervention logits and targets
    
    interv_logits: (B, n_targets)
    interv_targets: (B, )
    Returns scalar.
    """
    return F.cross_entropy(interv_logits, interv_targets)

