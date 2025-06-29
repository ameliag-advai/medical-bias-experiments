"""
Multi-demographic clamping utility for simultaneous female + old clamping.
"""
import torch
from typing import List, Literal, Optional
from src.advai.analysis.clamping_v2 import clamp_sae_features, get_feature_directions

DemographicType = Literal['male', 'female', 'old', 'young']
ClampingMethod = Literal['multiply', 'add', 'set']


def clamp_multiple_demographics(
    sae_out: torch.Tensor,
    demographics: List[DemographicType],
    extent: float = 5.0,
    method: ClampingMethod = 'add',
    inplace: bool = False
) -> torch.Tensor:
    """
    Clamp SAE features for multiple demographics simultaneously.
    
    Args:
        sae_out: SAE output tensor (1, num_features) or (num_features,)
        demographics: List of demographics to clamp (e.g., ['female', 'old'])
        extent: Factor for clamping intensity
        method: Clamping method - 'multiply', 'add', or 'set'
        inplace: If True, modify sae_out in-place
    
    Returns:
        Tensor with features clamped for all specified demographics.
    """
    if not inplace:
        result = sae_out.clone()
    else:
        result = sae_out
    
    # Apply clamping for each demographic sequentially
    for demographic in demographics:
        result = clamp_sae_features(
            result, 
            demographic=demographic, 
            extent=extent, 
            method=method, 
            inplace=True
        )
    
    return result


def clamp_female_old(
    sae_out: torch.Tensor,
    extent: float = 5.0,
    method: ClampingMethod = 'add',
    inplace: bool = False
) -> torch.Tensor:
    """
    Convenience function to clamp for female + old simultaneously.
    
    Args:
        sae_out: SAE output tensor
        extent: Clamping intensity
        method: Clamping method
        inplace: Whether to modify in-place
    
    Returns:
        Tensor clamped for both female and old characteristics.
    """
    return clamp_multiple_demographics(
        sae_out, 
        demographics=['female', 'old'], 
        extent=extent, 
        method=method, 
        inplace=inplace
    )


def analyze_multi_demographic_effect(
    original: torch.Tensor,
    clamped: torch.Tensor,
    demographics: List[DemographicType],
    top_k: int = 5
) -> dict:
    """
    Analyze the combined effect of multi-demographic clamping.
    
    Returns:
        Dictionary with analysis of changes for each demographic's features.
    """
    analysis = {}
    
    for demographic in demographics:
        feature_directions = get_feature_directions(demographic)
        
        for feature_idx, directions in list(feature_directions.items())[:top_k]:
            if feature_idx not in analysis:
                analysis[feature_idx] = {
                    'original_activation': original[0, feature_idx].item() if original.dim() > 1 else original[feature_idx].item(),
                    'clamped_activation': clamped[0, feature_idx].item() if clamped.dim() > 1 else clamped[feature_idx].item(),
                    'demographics': [],
                    'total_change': 0
                }
            
            change = analysis[feature_idx]['clamped_activation'] - analysis[feature_idx]['original_activation']
            analysis[feature_idx]['demographics'].append({
                'demographic': demographic,
                'expected_directions': directions,
                'change_contribution': change
            })
            analysis[feature_idx]['total_change'] = change
    
    return analysis


def test_female_old_clamping():
    """Test the female + old clamping functionality."""
    print("=== Testing Female + Old Multi-Demographic Clamping ===\n")
    
    # Create test activations
    test_acts = torch.zeros(1, 5000)
    
    # Set some baseline values for female features
    female_features = {696: 0.1396, 861: 0.1631, 610: 0.1201}
    for feat, val in female_features.items():
        test_acts[0, feat] = val
    
    # Set some baseline values for old features (placeholder - update when available)
    # old_features = {123: 0.1, 456: -0.2}  # Update with real old features
    # for feat, val in old_features.items():
    #     test_acts[0, feat] = val
    
    print("Original activations:")
    for feat, val in female_features.items():
        print(f"  Female feature {feat}: {test_acts[0, feat].item():.4f}")
    
    # Test sequential clamping
    print("\n=== Method 1: Sequential Clamping ===")
    female_clamped = clamp_sae_features(test_acts.clone(), 'female', extent=3.0, method='add')
    # Note: Old features not available yet, so this will only show female effect
    female_old_clamped = clamp_sae_features(female_clamped, 'old', extent=3.0, method='add')
    
    print("After female + old clamping:")
    for feat, val in female_features.items():
        original = test_acts[0, feat].item()
        final = female_old_clamped[0, feat].item()
        change = final - original
        print(f"  Female feature {feat}: {original:.4f} → {final:.4f} (Δ{change:+.4f})")
    
    # Test convenience function
    print("\n=== Method 2: Convenience Function ===")
    convenience_result = clamp_female_old(test_acts.clone(), extent=3.0, method='add')
    
    print("Using clamp_female_old():")
    for feat, val in female_features.items():
        original = test_acts[0, feat].item()
        final = convenience_result[0, feat].item()
        change = final - original
        print(f"  Female feature {feat}: {original:.4f} → {final:.4f} (Δ{change:+.4f})")
    
    print("\n✅ Multi-demographic clamping ready!")
    print("💡 Note: Old features will be more effective once OLD_FEATURES_WITH_DIRECTIONS is populated")


if __name__ == "__main__":
    test_female_old_clamping()
