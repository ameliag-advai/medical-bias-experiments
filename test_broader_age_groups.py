#!/usr/bin/env python3
"""
Test broader age groupings: young, middle_aged, old
"""

import torch
import json
from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.analyse import run_prompt
from src.advai.analysis.constants_v2 import (
    YOUNG_FEATURES_WITH_DIRECTIONS, 
    MIDDLE_AGE_FEATURES_WITH_DIRECTIONS,
    OLD_FEATURES_WITH_DIRECTIONS,
    FEMALE_FEATURES_WITH_DIRECTIONS
)

def test_broader_age_groups():
    """Test broader age groupings for better feature activation."""
    print("🔍 TESTING BROADER AGE GROUPS")
    print("=" * 60)
    
    # Load model
    print("📥 Loading model...")
    model, sae = load_model_and_sae()
    print("✅ Model loaded")
    
    # Define age group mappings
    age_groups = {
        'young': {
            'prompts': [
                "A 15-year-old patient presents with symptoms not specified",
                "A 22-year-old patient presents with symptoms not specified", 
                "A young patient presents with symptoms not specified",
                "A teenage patient presents with symptoms not specified"
            ],
            'features': YOUNG_FEATURES_WITH_DIRECTIONS
        },
        'middle_aged': {
            'prompts': [
                "A 45-year-old patient presents with symptoms not specified",
                "A 49-year-old patient presents with symptoms not specified",
                "A middle-aged patient presents with symptoms not specified",
                "An adult patient presents with symptoms not specified"
            ],
            'features': MIDDLE_AGE_FEATURES_WITH_DIRECTIONS
        },
        'old': {
            'prompts': [
                "A 75-year-old patient presents with symptoms not specified",
                "An 82-year-old patient presents with symptoms not specified", 
                "An elderly patient presents with symptoms not specified",
                "An old patient presents with symptoms not specified"
            ],
            'features': OLD_FEATURES_WITH_DIRECTIONS
        }
    }
    
    neutral_prompt = "A patient presents with symptoms not specified"
    
    # Get neutral baseline
    print(f"\n🧪 Getting neutral baseline...")
    neutral_result = run_prompt(neutral_prompt, model, sae, clamping=False, clamp_features={}, clamp_value=1.0)
    
    # Test each age group
    for age_group, config in age_groups.items():
        print(f"\n📊 TESTING {age_group.upper()} GROUP:")
        print(f"Features: {list(config['features'].keys())}")
        print(f"Expected values: {list(config['features'].values())}")
        
        # Test each prompt variant
        best_activation = 0
        best_prompt = ""
        
        for i, prompt in enumerate(config['prompts']):
            print(f"\n   Prompt {i+1}: {prompt}")
            
            # Get activations
            result = run_prompt(prompt, model, sae, clamping=False, clamp_features={}, clamp_value=1.0)
            
            # Check feature activations
            total_activation = 0
            for feat_idx, expected_val in config['features'].items():
                activation = result[f"activation_{feat_idx}"]
                neutral_act = neutral_result[f"activation_{feat_idx}"]
                diff = activation - neutral_act
                total_activation += abs(diff)
                print(f"     Feature {feat_idx}: neutral={neutral_act:.4f}, prompt={activation:.4f}, diff={diff:+.4f}, expected={expected_val:.3f}")
            
            print(f"     Total activation: {total_activation:.4f}")
            
            if total_activation > best_activation:
                best_activation = total_activation
                best_prompt = prompt
        
        print(f"\n   🏆 Best prompt for {age_group}: {best_prompt}")
        print(f"   🏆 Best total activation: {best_activation:.4f}")
        
        # Test clamping with best prompt
        if best_activation > 0:
            print(f"\n   🔧 Testing clamping equivalence:")
            best_result = run_prompt(best_prompt, model, sae, clamping=False, clamp_features={}, clamp_value=1.0)
            
            # Test different intensities
            for intensity in [1.0, 2.0, 5.0]:
                clamped_result = run_prompt(neutral_prompt, model, sae, clamping=True, 
                                          clamp_features=config['features'], clamp_value=intensity)
                similarity = calculate_cosine_similarity(best_result, clamped_result)
                print(f"     Intensity {intensity}x: similarity = {similarity:.3f}")
    
    # Compare with sex features (working baseline)
    print(f"\n📊 SEX FEATURE COMPARISON (working baseline):")
    sex_prompt = "A female patient presents with symptoms not specified"
    sex_result = run_prompt(sex_prompt, model, sae, clamping=False, clamp_features={}, clamp_value=1.0)
    
    sex_features = FEMALE_FEATURES_WITH_DIRECTIONS
    total_sex_activation = 0
    for feat_idx, expected_val in sex_features.items():
        activation = sex_result[f"activation_{feat_idx}"]
        neutral_act = neutral_result[f"activation_{feat_idx}"]
        diff = activation - neutral_act
        total_sex_activation += abs(diff)
        print(f"   Feature {feat_idx}: neutral={neutral_act:.4f}, sex={activation:.4f}, diff={diff:+.4f}, expected={expected_val:.3f}")
    
    print(f"   Total sex activation: {total_sex_activation:.4f}")
    
    # Test sex clamping
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
        test_broader_age_groups()
