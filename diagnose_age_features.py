#!/usr/bin/env python3
"""
Diagnose why age features aren't working for equivalence.
"""

import torch
import json
from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.analyse import run_prompt
from src.advai.analysis.constants_v2 import MIDDLE_AGE_FEATURES_WITH_DIRECTIONS, FEMALE_FEATURES_WITH_DIRECTIONS

def diagnose_age_features():
    """Diagnose age feature behavior."""
    print("🔍 DIAGNOSING AGE FEATURES")
    print("=" * 50)
    
    # Load model
    print("📥 Loading model...")
    model, sae = load_model_and_sae()
    print("✅ Model loaded")
    
    # Test prompts
    age_prompt = "A 49-year-old patient presents with symptoms not specified"
    neutral_prompt = "A patient presents with symptoms not specified"
    
    print(f"\n📝 Prompts:")
    print(f"   Age: {age_prompt}")
    print(f"   Neutral: {neutral_prompt}")
    
    # Get activations
    print(f"\n🧪 Getting activations...")
    age_result = run_prompt(age_prompt, model, sae, clamping=False, clamp_features={}, clamp_value=1.0)
    neutral_result = run_prompt(neutral_prompt, model, sae, clamping=False, clamp_features={}, clamp_value=1.0)
    
    # Test clamping with different intensities
    age_features = MIDDLE_AGE_FEATURES_WITH_DIRECTIONS  # {13032: 0.333, 1999: 0.135, 11060: 0.409, 5565: 0.37}
    
    print(f"\n📊 AGE FEATURE ANALYSIS:")
    print(f"Features: {list(age_features.keys())}")
    print(f"Expected values: {list(age_features.values())}")
    
    # Check natural differences
    print(f"\n🔍 Natural differences (age prompt vs neutral):")
    for feat_idx, expected_val in age_features.items():
        age_act = age_result[f"activation_{feat_idx}"]
        neutral_act = neutral_result[f"activation_{feat_idx}"]
        natural_diff = age_act - neutral_act
        print(f"   Feature {feat_idx}: neutral={neutral_act:.4f}, age={age_act:.4f}, diff={natural_diff:+.4f}, expected={expected_val:.3f}")
    
    # Test clamping intensities
    print(f"\n🔧 TESTING CLAMPING INTENSITIES:")
    
    for intensity in [1.0, 2.0, 5.0, 10.0, 20.0]:
        print(f"\n   Intensity {intensity}x:")
        
        # Apply clamping
        clamped_result = run_prompt(neutral_prompt, model, sae, clamping=True, 
                                  clamp_features=age_features, clamp_value=intensity)
        
        # Calculate similarity
        similarity = calculate_cosine_similarity(age_result, clamped_result)
        print(f"     Similarity: {similarity:.3f}")
        
        # Show feature changes
        for feat_idx, expected_val in age_features.items():
            neutral_act = neutral_result[f"activation_{feat_idx}"]
            clamped_act = clamped_result[f"activation_{feat_idx}"]
            added_amount = expected_val * intensity
            expected_final = neutral_act + added_amount
            print(f"     Feature {feat_idx}: {neutral_act:.4f} + {added_amount:.4f} = {expected_final:.4f} (actual: {clamped_act:.4f})")
    
    # Compare with sex features (working case)
    print(f"\n📊 SEX FEATURE COMPARISON (working case):")
    sex_features = FEMALE_FEATURES_WITH_DIRECTIONS
    sex_prompt = "A female patient presents with symptoms not specified"
    sex_result = run_prompt(sex_prompt, model, sae, clamping=False, clamp_features={}, clamp_value=1.0)
    
    print(f"Sex features: {list(sex_features.keys())}")
    for feat_idx, expected_val in sex_features.items():
        sex_act = sex_result[f"activation_{feat_idx}"]
        neutral_act = neutral_result[f"activation_{feat_idx}"]
        natural_diff = sex_act - neutral_act
        print(f"   Feature {feat_idx}: neutral={neutral_act:.4f}, sex={sex_act:.4f}, diff={natural_diff:+.4f}, expected={expected_val:.3f}")
    
    # Test sex clamping (should work)
    sex_clamped = run_prompt(neutral_prompt, model, sae, clamping=True, 
                           clamp_features=sex_features, clamp_value=1.0)
    sex_similarity = calculate_cosine_similarity(sex_result, sex_clamped)
    print(f"   Sex clamping similarity: {sex_similarity:.3f}")

def calculate_cosine_similarity(result_a, result_b):
    """Calculate cosine similarity between two activation results."""
    activations_a = []
    activations_b = []
    
    for i in range(16384):  # SAE has 16384 features
        key = f"activation_{i}"
        activations_a.append(result_a.get(key, 0.0))
        activations_b.append(result_b.get(key, 0.0))
    
    vec_a = torch.tensor(activations_a)
    vec_b = torch.tensor(activations_b)
    
    similarity = torch.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0))
    return similarity.item()

if __name__ == "__main__":
    with torch.no_grad():
        diagnose_age_features()
