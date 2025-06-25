"""Enhanced clamping utilities that handle directional feature activations.

This version properly handles the fact that demographic features can have both
positive and negative activations, and what matters is the relative difference
between demographics, not just the absolute activation value.
"""
import torch
from typing import List, Literal, Optional

from src.advai.analysis.constants_v2 import (
    MALE_FEATURES_WITH_DIRECTIONS,
    FEMALE_FEATURES_WITH_DIRECTIONS,
    OLD_FEATURES_WITH_DIRECTIONS,
    YOUNG_FEATURES_WITH_DIRECTIONS,
    MALE_FEATURES,
    FEMALE_FEATURES,
    OLD_FEATURES,
    YOUNG_FEATURES
)

DemographicType = Literal['male', 'female', 'old', 'young']
ClampingMethod = Literal['multiply', 'add', 'set']


def clamp_sae_features(
    sae_out: torch.Tensor,
    demographic: DemographicType,
    extent: float = 5.0,
    inplace: bool = False,
    method: ClampingMethod = 'multiply'
) -> torch.Tensor:
    """
    Clamp SAE features for a given demographic using directional information.

    Args:
        sae_out: SAE output tensor (1, num_features) or (num_features,)
        demographic: 'male', 'female', 'old', or 'young'
        extent: Factor for clamping intensity
        inplace: If True, modify sae_out in-place
        method: Clamping method - 'multiply', 'add', or 'set'
    
    Returns:
        Tensor with clamped features.
    """
    if method == 'multiply':
        return clamp_sae_features_simple(sae_out, demographic, extent, inplace)
    else:
        return clamp_sae_features_advanced(sae_out, demographic, extent, method, inplace)


def clamp_sae_features_simple(
    sae_out: torch.Tensor,
    demographic: DemographicType,
    extent: float = 5.0,
    inplace: bool = False
) -> torch.Tensor:
    """
    Simple clamping: multiply features by extent (backward compatibility).
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


def clamp_sae_features_advanced(
    sae_out: torch.Tensor,
    demographic: DemographicType,
    extent: float = 5.0,
    method: ClampingMethod = 'add',
    inplace: bool = False
) -> torch.Tensor:
    """
    Advanced clamping that respects the directional nature of features.
    
    Args:
        method: 
            - 'add': Add the typical demographic difference scaled by extent
            - 'set': Set features to typical demographic values scaled by extent
    """
    feature_map = {
        'male': MALE_FEATURES_WITH_DIRECTIONS,
        'female': FEMALE_FEATURES_WITH_DIRECTIONS,
        'old': OLD_FEATURES_WITH_DIRECTIONS,
        'young': YOUNG_FEATURES_WITH_DIRECTIONS,
    }

    features_with_directions = feature_map[demographic]
    
    if not features_with_directions:
        print(f"Warning: No directional features defined for {demographic}")
        return sae_out if inplace else sae_out.clone()
    
    if not inplace:
        sae_out = sae_out.clone()
    
    for feature_idx, directions in features_with_directions.items():
        if demographic in ['male', 'female']:
            target_value = directions[f'{demographic}_mean']
            other_value = directions['female_mean' if demographic == 'male' else 'male_mean']
        else:  # age demographics
            target_value = directions[f'{demographic}_mean']
            other_value = directions['young_mean' if demographic == 'old' else 'old_mean']
        
        difference = target_value - other_value
        
        if method == 'add':
            # Add the demographic difference scaled by extent
            sae_out[..., feature_idx] += difference * extent
        elif method == 'set':
            # Set to the target demographic value scaled by extent
            # Use the neutral point (average) as baseline and move toward target
            neutral_value = (target_value + other_value) / 2
            sae_out[..., feature_idx] = neutral_value + (target_value - neutral_value) * extent
    
    return sae_out


def get_feature_directions(demographic: DemographicType) -> dict:
    """
    Get the directional information for features of a given demographic.
    
    Returns:
        Dictionary with feature indices as keys and direction info as values
    """
    feature_map = {
        'male': MALE_FEATURES_WITH_DIRECTIONS,
        'female': FEMALE_FEATURES_WITH_DIRECTIONS,
        'old': OLD_FEATURES_WITH_DIRECTIONS,
        'young': YOUNG_FEATURES_WITH_DIRECTIONS,
    }
    
    return feature_map[demographic]


def analyze_clamping_effect(
    original: torch.Tensor,
    clamped: torch.Tensor,
    demographic: DemographicType,
    top_k: int = 5
) -> dict:
    """
    Analyze the effect of clamping on feature activations.
    
    Returns:
        Dictionary with analysis results
    """
    features_with_directions = get_feature_directions(demographic)
    
    if not features_with_directions:
        return {"error": f"No directional features defined for {demographic}"}
    
    results = {}
    feature_indices = list(features_with_directions.keys())[:top_k]
    
    for feature_idx in feature_indices:
        orig_val = original[..., feature_idx].item()
        clamp_val = clamped[..., feature_idx].item()
        change = clamp_val - orig_val
        
        directions = features_with_directions[feature_idx]
        
        results[feature_idx] = {
            'original_activation': orig_val,
            'clamped_activation': clamp_val,
            'change': change,
            'expected_directions': directions
        }
    
    return results


def test_clamping_methods(extent: float = 5.0, demographic: DemographicType = 'male'):
    """
    Test and compare all three clamping methods with example data.
    
    Args:
        extent: Clamping intensity factor
        demographic: Which demographic to test
    """
    print(f"=== Testing Clamping Methods for {demographic.upper()} (extent={extent}) ===\n")
    
    # Create test activations with neutral values
    test_acts = torch.zeros(1, 5000)
    
    if demographic == 'male':
        # Set to neutral values (halfway between male/female means)
        test_features = {
            1476: (-0.0912 + -0.0972) / 2,  # -0.0942
            1997: (0.0369 + 0.0153) / 2,    # 0.0261
            793: (-0.0787 + -0.0796) / 2,   # -0.0792
        }
    elif demographic == 'female':
        test_features = {
            953: (-0.1137 + -0.1029) / 2,   # -0.1083
            696: (0.1301 + 0.1490) / 2,     # 0.1396
            610: (0.1145 + 0.1256) / 2,     # 0.1201
        }
    else:
        print(f"No test data for {demographic} yet - run age analysis first!")
        return
    
    # Set test activations
    for feat, val in test_features.items():
        test_acts[0, feat] = val
    
    print("Original activations:")
    for feat, val in test_features.items():
        print(f"  Feature {feat}: {val:.4f}")
    print()
    
    # Test each method
    methods = ['multiply', 'add', 'set']
    results = {}
    
    for method in methods:
        clamped = clamp_sae_features(test_acts.clone(), demographic, extent=extent, method=method)
        results[method] = clamped
        
        print(f"After {method.upper()} method:")
        for feat in test_features.keys():
            orig = test_acts[0, feat].item()
            new = clamped[0, feat].item()
            change = new - orig
            print(f"  Feature {feat}: {orig:.4f} → {new:.4f} (change: {change:+.4f})")
        print()
    
    # Analysis
    print("=== Method Comparison ===")
    directions = get_feature_directions(demographic)
    
    for feat in test_features.keys():
        if feat in directions:
            dir_info = directions[feat]
            if demographic in ['male', 'female']:
                target = dir_info[f'{demographic}_mean']
                other = dir_info['female_mean' if demographic == 'male' else 'male_mean']
            else:
                target = dir_info[f'{demographic}_mean']
                other = dir_info['young_mean' if demographic == 'old' else 'old_mean']
            
            print(f"\nFeature {feat} (target: {target:.4f}, other: {other:.4f}):")
            for method in methods:
                val = results[method][0, feat].item()
                distance_to_target = abs(val - target)
                print(f"  {method:8}: {val:.4f} (distance to target: {distance_to_target:.4f})")


def demonstrate_clamping_concepts():
    """
    Demonstrate the key concepts behind directional clamping.
    """
    print("=== CLAMPING CONCEPTS EXPLAINED ===\n")
    
    print("🔍 Understanding Feature Directions:")
    print("  • POSITIVE male feature: males have MORE positive activation than females")
    print("  • NEGATIVE male feature: males have LESS negative activation than females")
    print("  • What matters is the RELATIVE difference, not absolute sign\n")
    
    print("📊 Example with Feature 1997 (positive male feature):")
    print("  • Male typical: +0.0369")
    print("  • Female typical: +0.0153") 
    print("  • Difference: +0.0216 (males more positive)\n")
    
    print("📊 Example with Feature 1476 (negative male feature):")
    print("  • Male typical: -0.0912")
    print("  • Female typical: -0.0972")
    print("  • Difference: +0.0060 (males less negative)\n")
    
    print("⚠️  Why MULTIPLY method fails:")
    print("  • Positive feature: 0.026 × 5 = 0.130 ✅ (more positive = good)")
    print("  • Negative feature: -0.094 × 5 = -0.471 ❌ (more negative = wrong!)\n")
    
    print("✅ Why ADD method works:")
    print("  • Always adds the demographic difference")
    print("  • Positive: +0.026 + (0.022×5) = +0.134 ✅")
    print("  • Negative: -0.094 + (0.006×5) = -0.064 ✅ (less negative)\n")
    
    print("🎯 Recommendations:")
    print("  • USE: ADD method with extent 2-5")
    print("  • AVOID: MULTIPLY method for mixed pos/neg features")
    print("  • CONSIDER: SET method for strong interventions")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_clamping_concepts()
    print("\n" + "="*60 + "\n")
    test_clamping_methods(extent=3.0, demographic='male')
    print("\n" + "="*60 + "\n")
    test_clamping_methods(extent=3.0, demographic='female')
