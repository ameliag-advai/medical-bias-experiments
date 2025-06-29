#!/usr/bin/env python3
"""
Test different clamping intensities to optimize age equivalence.
"""

import torch
import json
from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.analyse import run_prompt
from src.advai.analysis.constants_v2 import (
    MIDDLE_AGE_FEATURES_WITH_DIRECTIONS,
    YOUNG_FEATURES_WITH_DIRECTIONS,
    OLD_FEATURES_WITH_DIRECTIONS,
    FEMALE_FEATURES_WITH_DIRECTIONS
)

def test_intensity_optimization():
    """Test different clamping intensities for age features."""
    print("🔧 TESTING CLAMPING INTENSITY OPTIMIZATION")
    print("=" * 60)
    
    # Load model
    print("📥 Loading model...")
    model, sae = load_model_and_sae()
    print("✅ Model loaded")
    
    # Test cases
    test_cases = [
        {
            'name': 'middle_age',
            'demographic_prompt': "A 49-year-old patient presents with symptoms not specified",
            'features': MIDDLE_AGE_FEATURES_WITH_DIRECTIONS
        },
        {
            'name': 'young',
            'demographic_prompt': "A 15-year-old patient presents with symptoms not specified", 
            'features': YOUNG_FEATURES_WITH_DIRECTIONS
        },
        {
            'name': 'elderly',
            'demographic_prompt': "A 75-year-old patient presents with symptoms not specified",
            'features': OLD_FEATURES_WITH_DIRECTIONS
        },
        {
            'name': 'female_baseline',
            'demographic_prompt': "A female patient presents with symptoms not specified",
            'features': FEMALE_FEATURES_WITH_DIRECTIONS
        }
    ]
    
    neutral_prompt = "A patient presents with symptoms not specified"
    
    # Test different intensities
    intensities = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    for case in test_cases:
        print(f"\n📊 TESTING {case['name'].upper()}:")
        print(f"   Demographic prompt: {case['demographic_prompt']}")
        print(f"   Features: {list(case['features'].keys())}")
        
        # Get target (demographic prompt result)
        target_result = run_prompt(case['demographic_prompt'], model, sae, 
                                 clamping=False, clamp_features={}, clamp_value=1.0)
        
        print(f"\n   🎯 Testing intensities:")
        best_similarity = 0
        best_intensity = 0
        
        for intensity in intensities:
            # Get clamped result
            clamped_result = run_prompt(neutral_prompt, model, sae, 
                                      clamping=True, clamp_features=case['features'], 
                                      clamp_value=intensity)
            
            # Calculate similarity
            similarity = calculate_cosine_similarity(target_result, clamped_result)
            
            # Check if equivalent (>0.8 threshold)
            equivalent = "✅" if similarity > 0.8 else "❌"
            
            print(f"     Intensity {intensity:3.1f}x: similarity = {similarity:.3f} {equivalent}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_intensity = intensity
        
        print(f"   🏆 Best: {best_intensity}x intensity → {best_similarity:.3f} similarity")
        
        # Test if best intensity achieves equivalence
        if best_similarity > 0.8:
            print(f"   🎉 EQUIVALENCE ACHIEVED with {best_intensity}x intensity!")
        else:
            print(f"   ⚠️  Still below 0.8 threshold (need {0.8 - best_similarity:.3f} more)")

def calculate_cosine_similarity(result_a, result_b):
    """Calculate cosine similarity between two activation results."""
    activations_a = []
    activations_b = []
    
    for i in range(16384):
        key = f"activation_{i}"
        activations_a.append(result_a.get(key, 0.0))
        activations_b.append(result_b.get(key, 0.0))
    
    vec_a = torch.tensor(activations_a)
    vec_b = torch.tensor(activations_b)
    
    similarity = torch.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0))
    return similarity.item()

if __name__ == "__main__":
    with torch.no_grad():
        test_intensity_optimization()
