#!/usr/bin/env python3
"""
Debug age clamping by analyzing feature activations and testing different intensities.
"""

import torch
import json
from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.analyse import run_prompt
from src.advai.analysis.clamping_v2 import clamp_sae_features
from src.advai.analysis.constants_v2 import MIDDLE_AGE_FEATURES_WITH_DIRECTIONS, FEMALE_FEATURES_WITH_DIRECTIONS

def analyze_feature_activations():
    """Analyze how age vs sex features behave in the model."""
    print("🔍 DEBUGGING AGE CLAMPING")
    print("=" * 60)
    
    # Load model and SAE
    print("📥 Loading model and SAE...")
    model, sae = load_model_and_sae()
    device = next(model.parameters()).device
    print(f"✅ Loaded on device: {device}")
    
    # Test prompts
    age_prompt = "A 49-year-old patient presents with symptoms not specified"
    sex_prompt = "A female patient presents with symptoms not specified"
    neutral_prompt = "A patient presents with symptoms not specified"
    
    print(f"\n📝 Test prompts:")
    print(f"   Age: {age_prompt}")
    print(f"   Sex: {sex_prompt}")
    print(f"   Neutral: {neutral_prompt}")
    
    # Get baseline activations
    print(f"\n🧪 Getting baseline activations...")
    age_result = run_prompt(age_prompt, model, sae, clamping=False, clamp_features={}, clamp_value=1.0)
    sex_result = run_prompt(sex_prompt, model, sae, clamping=False, clamp_features={}, clamp_value=1.0)
    neutral_result = run_prompt(neutral_prompt, model, sae, clamping=False, clamp_features={}, clamp_value=1.0)
    
    # Analyze age features
    print(f"\n📊 MIDDLE_AGE FEATURE ANALYSIS:")
    age_features = MIDDLE_AGE_FEATURES_WITH_DIRECTIONS
    for feat_idx, expected_val in age_features.items():
        age_act = age_result[f"activation_{feat_idx}"]
        neutral_act = neutral_result[f"activation_{feat_idx}"]
        diff = age_act - neutral_act
        print(f"   Feature {feat_idx}: neutral={neutral_act:.4f}, age={age_act:.4f}, diff={diff:+.4f}, expected={expected_val:.3f}")
    
    # Analyze sex features  
    print(f"\n📊 FEMALE FEATURE ANALYSIS:")
    sex_features = {387: 0.2, 6221: 0.2, 5176: 0.2, 12813: 0.2}  # From test case
    for feat_idx, expected_val in sex_features.items():
        sex_act = sex_result[f"activation_{feat_idx}"]
        neutral_act = neutral_result[f"activation_{feat_idx}"]
        diff = sex_act - neutral_act
        print(f"   Feature {feat_idx}: neutral={neutral_act:.4f}, sex={sex_act:.4f}, diff={diff:+.4f}, expected={expected_val:.3f}")
    
    # Test clamping with different intensities
    print(f"\n🔧 TESTING CLAMPING INTENSITIES:")
    
    # Age clamping test
    age_clamp_features = {13032: 0.333, 1999: 0.135, 11060: 0.409, 5565: 0.37}
    print(f"\n   AGE CLAMPING:")
    for intensity in [1.0, 2.0, 5.0, 10.0]:
        # Scale the clamping values
        scaled_features = {k: v * intensity for k, v in age_clamp_features.items()}
        clamped_result = run_prompt(neutral_prompt, model, sae, clamping=True, 
                                  clamp_features=scaled_features, clamp_value=1.0)
        
        # Compare with age prompt
        similarity = calculate_cosine_similarity(age_result, clamped_result)
        print(f"     Intensity {intensity}x: similarity = {similarity:.3f}")
        
        # Show feature changes
        for feat_idx in [13032, 1999]:  # Show first 2 features
            orig_val = age_result[f"activation_{feat_idx}"]
            clamp_val = clamped_result[f"activation_{feat_idx}"]
            print(f"       Feature {feat_idx}: {orig_val:.4f} vs {clamp_val:.4f}")
    
    # Sex clamping test (for comparison)
    sex_clamp_features = {387: 0.2, 6221: 0.2, 5176: 0.2, 12813: 0.2}
    print(f"\n   SEX CLAMPING (for comparison):")
    clamped_result = run_prompt(neutral_prompt, model, sae, clamping=True, 
                              clamp_features=sex_clamp_features, clamp_value=1.0)
    similarity = calculate_cosine_similarity(sex_result, clamped_result)
    print(f"     Standard: similarity = {similarity:.3f}")

def calculate_cosine_similarity(result_a, result_b):
    """Calculate cosine similarity between two activation results."""
    # Extract activation vectors
    activations_a = []
    activations_b = []
    
    for i in range(16384):  # SAE has 16384 features
        key = f"activation_{i}"
        activations_a.append(result_a.get(key, 0.0))
        activations_b.append(result_b.get(key, 0.0))
    
    # Convert to tensors
    vec_a = torch.tensor(activations_a)
    vec_b = torch.tensor(activations_b)
    
    # Calculate cosine similarity
    similarity = torch.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0))
    return similarity.item()

if __name__ == "__main__":
    with torch.no_grad():
        analyze_feature_activations()
