"""
Clamping functions for SAE features.
"""

import torch
from typing import List, Union
from .constants_v2 import (
    MALE_FEATURES,
    FEMALE_FEATURES,
    OLD_FEATURES,
    YOUNG_FEATURES,
    MALE_FEATURES_WITH_DIRECTIONS,
    FEMALE_FEATURES_WITH_DIRECTIONS,
    OLD_FEATURES_WITH_DIRECTIONS,
    YOUNG_FEATURES_WITH_DIRECTIONS,
    PEDIATRIC_FEATURES_WITH_DIRECTIONS,
    ADOLESCENT_FEATURES_WITH_DIRECTIONS,
    YOUNG_ADULT_FEATURES_WITH_DIRECTIONS,
    MIDDLE_AGE_FEATURES_WITH_DIRECTIONS,
    SENIOR_FEATURES_WITH_DIRECTIONS
)

DemographicType = Union[str, List[str]]


def clamp_sae_features(
    sae_out: torch.Tensor,
    clamp_features: Union[DemographicType, dict],
    clamp_value: float = 5.0,
    inplace: bool = False
) -> torch.Tensor:
    """
    Clamp SAE features using additive method with demographic differences.
    
    Args:
        sae_out: SAE output tensor [..., num_features]
        clamp_features: Either demographic name(s) or dict of {feature_idx: value}
        clamp_value: Extent/intensity for demographic differences (when using demographic names)
        inplace: Whether to modify tensor in place
        
    Returns:
        Tensor with clamped features
    """
    if not inplace:
        sae_out = sae_out.clone()
    
    # Handle dictionary of specific feature values using additive method
    if isinstance(clamp_features, dict):
        for feature_idx, demographic_difference in clamp_features.items():
            # Use additive method: new_activation = original + (demographic_difference × extent)
            added_amount = demographic_difference * clamp_value
            sae_out[..., feature_idx] += added_amount
        return sae_out
    
    # Handle demographic names using additive method
    if isinstance(clamp_features, str):
        clamp_features = [clamp_features]
    
    # Map demographics to their directional features
    directional_feature_map = {
        'male': MALE_FEATURES_WITH_DIRECTIONS,
        'female': FEMALE_FEATURES_WITH_DIRECTIONS, 
        'old': OLD_FEATURES_WITH_DIRECTIONS,
        'young': YOUNG_FEATURES_WITH_DIRECTIONS,
        'pediatric': PEDIATRIC_FEATURES_WITH_DIRECTIONS,
        'adolescent': ADOLESCENT_FEATURES_WITH_DIRECTIONS,
        'young_adult': YOUNG_ADULT_FEATURES_WITH_DIRECTIONS,
        'middle_age': MIDDLE_AGE_FEATURES_WITH_DIRECTIONS,
        'senior': SENIOR_FEATURES_WITH_DIRECTIONS,
    }
    
    for demographic in clamp_features:
        if demographic not in directional_feature_map:
            print(f"Warning: No directional features defined for {demographic}")
            continue
            
        features_with_directions = directional_feature_map[demographic]
        
        for feature_idx, target_value in features_with_directions.items():
            # For the additive method, we add the demographic difference scaled by extent
            # new_activation = original + (demographic_difference × extent)
            # Since we have the target value directly, we use it as the demographic difference
            demographic_difference = target_value
            added_amount = demographic_difference * clamp_value
            sae_out[..., feature_idx] += added_amount
    
    return sae_out
