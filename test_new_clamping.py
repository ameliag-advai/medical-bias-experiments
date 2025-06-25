"""Test the new directional clamping approach"""
import torch
import numpy as np

def create_test_activations():
    """Create test activations that match the statistics from analysis"""
    activations = torch.zeros(1, 5000)  # Assuming 5000 features
    
    # Set male feature activations to their typical "neutral" values
    # (halfway between male and female means)
    male_features = {
        1476: (-0.0912 + -0.0972) / 2,  # -0.0942
        1997: (0.0369 + 0.0153) / 2,    # 0.0261
        793: (-0.0787 + -0.0796) / 2,   # -0.0792
        728: (0.2034 + 0.1959) / 2,     # 0.1997
        126: (-0.1319 + -0.1347) / 2,   # -0.1333
        238: (-0.1698 + -0.1822) / 2,   # -0.1760
        1202: (-0.3500 + -0.3666) / 2,  # -0.3583
        1738: (-0.1565 + -0.1678) / 2,  # -0.1622
        317: (-0.0045 + -0.0106) / 2,   # -0.0076
        356: (-0.4624 + -0.4880) / 2,   # -0.4752
    }
    
    female_features = {
        953: (-0.1137 + -0.1029) / 2,   # -0.1083
        694: (-0.3857 + -0.3635) / 2,   # -0.3746
        696: (0.1301 + 0.1490) / 2,     # 0.1396
        346: (-0.2348 + -0.2233) / 2,   # -0.2291
        861: (0.1616 + 0.1646) / 2,     # 0.1631
        1989: (-0.1159 + -0.1047) / 2,  # -0.1103
        610: (0.1145 + 0.1256) / 2,     # 0.1201
        486: (0.0266 + 0.0440) / 2,     # 0.0353
        1899: (0.1243 + 0.1445) / 2,    # 0.1344
        311: (-0.0378 + -0.0330) / 2,   # -0.0354
    }
    
    # Set the activations
    for feat, val in male_features.items():
        activations[0, feat] = val
    for feat, val in female_features.items():
        activations[0, feat] = val
    
    return activations

def main():
    # Test different clamping approaches
    print("Testing new directional clamping approaches\n")

    # Create test activations
    test_acts = create_test_activations()

    # Import the clamping functions
    from src.advai.analysis.clamping_v2 import clamp_sae_features, analyze_clamping_effect

    print("Original activations for key features:")
    print("Male feature 1997 (positive):", test_acts[0, 1997].item())
    print("Male feature 1476 (negative):", test_acts[0, 1476].item())
    print("Female feature 696 (positive):", test_acts[0, 696].item())
    print("Female feature 953 (negative):", test_acts[0, 953].item())

    # Test 1: Simple multiplication (current approach)
    print("\n=== Test 1: Simple multiplication (5x) ===")
    clamped_male_mult = clamp_sae_features(test_acts.clone(), 'male', extent=5.0, method='multiply')
    print("Male feature 1997 after 5x:", clamped_male_mult[0, 1997].item())
    print("Male feature 1476 after 5x:", clamped_male_mult[0, 1476].item())

    # Test 2: Advanced clamping with 'add' method
    print("\n=== Test 2: Advanced 'add' method (5x) ===")
    clamped_male_add = clamp_sae_features(test_acts.clone(), 'male', extent=5.0, method='add')
    print("Male feature 1997 after add:", clamped_male_add[0, 1997].item())
    print("Male feature 1476 after add:", clamped_male_add[0, 1476].item())

    # Test 3: Advanced clamping with 'set' method
    print("\n=== Test 3: Advanced 'set' method (2x) ===")
    clamped_male_set = clamp_sae_features(test_acts.clone(), 'male', extent=2.0, method='set')
    print("Male feature 1997 after set:", clamped_male_set[0, 1997].item())
    print("Male feature 1476 after set:", clamped_male_set[0, 1476].item())

    # Test female clamping
    print("\n=== Test 4: Female clamping comparison ===")
    clamped_female = clamp_sae_features(test_acts.clone(), 'female', extent=5.0, method='add')
    print("Female feature 696 after add:", clamped_female[0, 696].item())
    print("Female feature 953 after add:", clamped_female[0, 953].item())

    # Analyze the effects
    print("\n=== Detailed Analysis ===")
    analysis = analyze_clamping_effect(test_acts, clamped_male_add, 'male', top_k=3)
    
    for feature_idx, info in analysis.items():
        print(f"\nFeature {feature_idx}:")
        print(f"  Original: {info['original_activation']:.4f}")
        print(f"  Clamped: {info['clamped_activation']:.4f}")
        print(f"  Change: {info['change']:.4f}")
        directions = info['expected_directions']
        print(f"  Expected male: {directions['male_mean']:.4f}, female: {directions['female_mean']:.4f}")
        print(f"  Difference: {directions['male_mean'] - directions['female_mean']:.4f}")

    print("\n=== Recommendations ===")
    print("1. 'multiply' method: Simple but may amplify in wrong direction for negative features")
    print("2. 'add' method: Preserves relative relationships, good for subtle interventions")
    print("3. 'set' method: Most aggressive, directly pushes toward demographic-typical values")
    print("\nFor your analysis, 'add' method with extent 2-5 is probably best.")

if __name__ == "__main__":
    main()
