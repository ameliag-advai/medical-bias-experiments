#!/usr/bin/env python3
"""
Final validation test with optimal intensities for all demographics.
"""

import torch
from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.analyse import run_prompt
from src.advai.analysis.constants_v2 import (
    MIDDLE_AGE_FEATURES_WITH_DIRECTIONS,
    YOUNG_FEATURES_WITH_DIRECTIONS, 
    OLD_FEATURES_WITH_DIRECTIONS,
    MALE_FEATURES_WITH_DIRECTIONS,
    FEMALE_FEATURES_WITH_DIRECTIONS
)

def final_validation():
    """Final validation with optimal intensities."""
    print("🏆 FINAL DEMOGRAPHIC CLAMPING VALIDATION")
    print("=" * 60)
    
    # Load model
    print("📥 Loading model...")
    model, sae = load_model_and_sae()
    print("✅ Model loaded")
    
    # Optimal configurations discovered
    test_configs = [
        {
            'name': 'Middle-aged Female',
            'demographic_prompt': "A 49-year-old female patient presents with chest pain",
            'neutral_prompt': "A patient presents with chest pain",
            'features': {**MIDDLE_AGE_FEATURES_WITH_DIRECTIONS, **FEMALE_FEATURES_WITH_DIRECTIONS},
            'optimal_intensity': 1.0  # Conservative for combined
        },
        {
            'name': 'Young Male', 
            'demographic_prompt': "A 15-year-old male patient presents with headache",
            'neutral_prompt': "A patient presents with headache",
            'features': {**YOUNG_FEATURES_WITH_DIRECTIONS, **MALE_FEATURES_WITH_DIRECTIONS},
            'optimal_intensity': 1.0
        },
        {
            'name': 'Elderly Female',
            'demographic_prompt': "A 75-year-old female patient presents with fatigue", 
            'neutral_prompt': "A patient presents with fatigue",
            'features': {**OLD_FEATURES_WITH_DIRECTIONS, **FEMALE_FEATURES_WITH_DIRECTIONS},
            'optimal_intensity': 1.0
        },
        {
            'name': 'Female Only',
            'demographic_prompt': "A female patient presents with abdominal pain",
            'neutral_prompt': "A patient presents with abdominal pain", 
            'features': FEMALE_FEATURES_WITH_DIRECTIONS,
            'optimal_intensity': 0.5
        },
        {
            'name': 'Male Only',
            'demographic_prompt': "A male patient presents with back pain",
            'neutral_prompt': "A patient presents with back pain",
            'features': MALE_FEATURES_WITH_DIRECTIONS, 
            'optimal_intensity': 1.0
        }
    ]
    
    print(f"\n🧪 Testing {len(test_configs)} demographic combinations:")
    
    results = []
    for i, config in enumerate(test_configs, 1):
        print(f"\n{i}. {config['name']}:")
        print(f"   Demographic: {config['demographic_prompt']}")
        print(f"   Neutral: {config['neutral_prompt']}")
        print(f"   Features: {len(config['features'])} features")
        print(f"   Intensity: {config['optimal_intensity']}x")
        
        # Get target (demographic prompt)
        target = run_prompt(config['demographic_prompt'], model, sae,
                          clamping=False, clamp_features={}, clamp_value=1.0)
        
        # Get clamped result
        clamped = run_prompt(config['neutral_prompt'], model, sae,
                           clamping=True, clamp_features=config['features'],
                           clamp_value=config['optimal_intensity'])
        
        # Calculate similarity
        similarity = calculate_cosine_similarity(target, clamped)
        equivalent = similarity > 0.8
        
        status = "✅ EQUIVALENT" if equivalent else "❌ DIFFERENT"
        print(f"   📊 Similarity: {similarity:.3f} {status}")
        
        results.append({
            'name': config['name'],
            'similarity': similarity,
            'equivalent': equivalent,
            'intensity': config['optimal_intensity']
        })
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 FINAL VALIDATION SUMMARY")
    print("=" * 60)
    
    equivalent_count = sum(1 for r in results if r['equivalent'])
    total_count = len(results)
    success_rate = (equivalent_count / total_count) * 100
    
    print(f"✅ Equivalent: {equivalent_count}/{total_count} ({success_rate:.1f}%)")
    print(f"📈 Average similarity: {sum(r['similarity'] for r in results) / len(results):.3f}")
    
    print(f"\n📋 Individual Results:")
    for result in results:
        status = "✅" if result['equivalent'] else "❌"
        print(f"   {status} {result['name']}: {result['similarity']:.3f} (intensity: {result['intensity']}x)")
    
    if success_rate == 100:
        print(f"\n🎉 PERFECT SUCCESS! All demographic clamping achieves equivalence!")
        print(f"🚀 The discovered responsive features work flawlessly!")
    elif success_rate >= 80:
        print(f"\n✅ EXCELLENT! {success_rate:.1f}% success rate - demographic clamping is highly effective!")
    else:
        print(f"\n⚠️  Needs improvement: {success_rate:.1f}% success rate")

def calculate_cosine_similarity(result_a, result_b):
    """Calculate cosine similarity between activation results."""
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
        final_validation()
