"""
Clamping functions for SAE features.
"""

import torch
from typing import List, Union
from .constants import MALE_FEATURES, FEMALE_FEATURES, OLD_FEATURES, YOUNG_FEATURES

DemographicType = Union[str, List[str]]


def clamp_sae_features(
    sae_out: torch.Tensor,
    demographics: DemographicType,
    extent: float = 5.0,
    inplace: bool = False
) -> torch.Tensor:
    """
    Clamp SAE features associated with demographics.
    
    Args:
        sae_out: SAE output tensor [..., num_features]
        demographics: Single demographic or list of demographics
        extent: Multiplier for feature activations
        inplace: Whether to modify tensor in place
        
    Returns:
        Tensor with clamped features
    """
    if isinstance(demographics, str):
        demographics = [demographics]
    
    feature_map = {
        'male': MALE_FEATURES,
        'female': FEMALE_FEATURES,
        'old': OLD_FEATURES,
        'young': YOUNG_FEATURES,
    }
    
    features = []
    for demographic in demographics:
        if demographic in feature_map:
            features.extend(feature_map[demographic])
    
    if not inplace:
        sae_out = sae_out.clone()
    
    for f in features:
        sae_out[..., f] *= extent
    
    return sae_out
