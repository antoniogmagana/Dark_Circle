"""
Intervention modules.

KnownInterventionHandler  — builds binary masks from ground-truth interv_idx.
UnknownInterventionClassifier — CITRIS-style: infers which z_i changed
                                between two consecutive latents (z_t, z_t1).
"""

import torch
import torch.nn as nn

from crl_vehicle.data.transforms import N_INTERVENTIONS


class KnownInterventionHandler(nn.Module):
    """
    Converts integer intervention indices into a (B, d_z) binary mask
    for use in the SCM forward pass and KL computation.

    interv_idx semantics (from transforms.py):
        0           = no intervention
        1..N_INTERVENTIONS = noise type index
    The mask maps noise type k → z dimension (k-1) in the noise block.
    Dimensions outside the noise block are never masked here; the SCM
    and KL use the mask to skip the mechanism / KL for that dimension.
    """

    def __init__(self, d_z: int, noise_start_idx: int):
        """
        d_z             : total latent dimension
        noise_start_idx : index of the first noise dimension in z
                          (= d_z_presence + d_z_type + d_z_proximity)
        """
        super().__init__()
        self.d_z = d_z
        self.noise_start = noise_start_idx

    def make_mask(
        self, interv_idx: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        interv_idx : (B,) long tensor, values in {0, 1..N_INTERVENTIONS}
        Returns    : (B, d_z) float binary mask
        """
        B = interv_idx.shape[0]
        mask = torch.zeros(B, self.d_z, device=device)
        active = interv_idx > 0
        if active.any():
            # Map intervention type 1..N_INTERVENTIONS to a noise dim.
            # Cycle within the noise block if N_INTERVENTIONS > d_z_noise.
            noise_dim = (interv_idx[active] - 1) % (self.d_z - self.noise_start)
            mask[active, self.noise_start + noise_dim] = 1.0
        return mask

    def forward(
        self, interv_idx: torch.Tensor
    ) -> torch.Tensor:
        return self.make_mask(interv_idx, interv_idx.device)


class UnknownInterventionClassifier(nn.Module):
    """
    CITRIS-style classifier: given (z_t, z_t1), predict which latent
    variable was intervened on between t and t+1.

    Input : concatenation of [z_t, z_t1 - z_t]  →  (B, 2*d_z)
    Output: logits over (d_z + 1) targets
            — d_z variable indices plus class 0 = "no intervention"
    """

    def __init__(self, d_z: int):
        super().__init__()
        n_targets = d_z + 1   # one per variable + "no intervention"
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_z, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_targets),
        )

    def forward(
        self, z_t: torch.Tensor, z_t1: torch.Tensor
    ) -> torch.Tensor:
        """
        z_t, z_t1 : (B, d_z)
        Returns   : (B, d_z + 1) logits
        """
        delta = torch.cat([z_t, z_t1 - z_t], dim=-1)   # (B, 2*d_z)
        return self.classifier(delta)
