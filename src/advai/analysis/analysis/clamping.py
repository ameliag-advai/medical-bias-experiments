"""


Manual PyTorch clamping utilities for SAE features representing demographic info (sex, age).
Allows user to clamp features for 'male', 'female', 'old', 'young' to a specified extent (e.g., 5x, 10x).

Features were identified in previous analysis; update the indices below if new analysis is run.
"""
import torch
from typing import List, Literal

# Indices of features most predictive for each demographic
MALE_FEATURES = [198, 845, 678]       # Positive coef for sex (male)
FEMALE_FEATURES = [1577, 1343, 1699, 856, 382, 184, 507]  # Negative coef for sex (female)
OLD_FEATURES = [184, 394, 137, 674, 110, 1577, 1667]      # Positive coef for age
YOUNG_FEATURES = [478, 1533, 1403]    # Negative coef for age

DemographicType = Literal['male', 'female', 'old', 'young']

def clamp_sae_features(
    sae_out: torch.Tensor,
    demographic: DemographicType,
    extent: float = 5.0,
    inplace: bool = False
) -> torch.Tensor:
    """
    Clamp SAE features for a given demographic by multiplying their value by `extent`.
    Args:
        sae_out: SAE output tensor (1, num_features) or (num_features,)
        demographic: 'male', 'female', 'old', or 'young'
        extent: Factor to multiply the feature by (e.g., 5.0 or 10.0)
        inplace: If True, modify sae_out in-place. If False, return a new tensor.
    Returns:
        Tensor with clamped features.
    """
    feature_map = {
        'male': MALE_FEATURES,
        'female': FEMALE_FEATURES,
        'old': OLD_FEATURES,
        'young': YOUNG_FEATURES,
    }
    features = feature_map[demographic]
    if not inplace:
        sae_out = sae_out.clone()
    for f in features:
        sae_out[..., f] *= extent
    return sae_out

# Example usage:
# clamped = clamp_sae_features(sae_out, demographic='male', extent=5.0)
