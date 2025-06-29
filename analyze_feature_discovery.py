#!/usr/bin/env python3
"""
Discover which features actually activate with demographic prompts.
"""

import torch
import json
from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.analyse import run_prompt

def discover_active_features():
    """Find features that actually activate with demographic prompts."""
    print("🔍 DISCOVERING ACTIVE FEATURES")
    print("=" * 60)
    
    # Load model
    print("📥 Loading model...")
    model, sae = load_model_and_sae()
    print("✅ Model loaded")
    
    # Test prompts
    prompts = {
        'neutral': "A patient presents with symptoms not specified",
        'age_specific': "A 49-year-old patient presents with symptoms not specified",
        'sex_specific': "A female patient presents with symptoms not specified",
        'both': "A 49-year-old female patient presents with symptoms not specified",
        'age_word': "An elderly patient presents with symptoms not specified",
        'sex_pronoun': "She presents with symptoms not specified"
    }
    
    print(f"\n🧪 Getting activations for all prompts...")
    results = {}
    for name, prompt in prompts.items():
        print(f"   {name}: {prompt}")
        results[name] = run_prompt(prompt, model, sae, clamping=False, clamp_features={}, clamp_value=1.0)
    
    # Find features that show differences
    print(f"\n📊 ANALYZING FEATURE DIFFERENCES:")
    
    threshold = 0.001  # Minimum difference to consider
    
    for comparison_name, comparison_prompt in prompts.items():
        if comparison_name == 'neutral':
            continue
            
        print(f"\n🔍 {comparison_name.upper()} vs NEUTRAL:")
        print(f"   Prompt: {comparison_prompt}")
        
        # Find features with significant differences
        significant_features = []
        
        for i in range(16384):  # All SAE features
            neutral_val = results['neutral'][f"activation_{i}"]
            comparison_val = results[comparison_name][f"activation_{i}"]
            diff = comparison_val - neutral_val
            
            if abs(diff) > threshold:
                significant_features.append({
                    'feature': i,
                    'neutral': neutral_val,
                    'comparison': comparison_val,
                    'difference': diff
                })
        
        # Sort by absolute difference
        significant_features.sort(key=lambda x: abs(x['difference']), reverse=True)
        
        print(f"   Found {len(significant_features)} features with |diff| > {threshold}")
        
        # Show top 10 features
        print(f"   Top 10 most different features:")
        for j, feat in enumerate(significant_features[:10]):
            print(f"     {j+1}. Feature {feat['feature']}: {feat['neutral']:.4f} → {feat['comparison']:.4f} (diff: {feat['difference']:+.4f})")
        
        # Test if these discovered features work for clamping
        if len(significant_features) >= 3:
            print(f"\n   🔧 Testing clamping with discovered features:")
            
            # Use top 3 features for clamping test
            test_features = {}
            for feat in significant_features[:3]:
                test_features[feat['feature']] = feat['difference']  # Use the actual difference as the demographic difference
            
            print(f"   Test features: {test_features}")
            
            # Test clamping
            clamped_result = run_prompt(prompts['neutral'], model, sae, clamping=True, 
                                      clamp_features=test_features, clamp_value=1.0)
            
            # Calculate similarity
            similarity = calculate_cosine_similarity(results[comparison_name], clamped_result)
            print(f"   Clamping similarity: {similarity:.3f}")
    
    # Special analysis for sex features (working case)
    print(f"\n📊 DETAILED SEX ANALYSIS:")
    sex_features = [387, 6221, 5176, 12813]  # Known sex features
    
    for feat_idx in sex_features:
        neutral_val = results['neutral'][f"activation_{feat_idx}"]
        sex_val = results['sex_specific'][f"activation_{feat_idx}"]
        pronoun_val = results['sex_pronoun'][f"activation_{feat_idx}"]
        
        print(f"   Feature {feat_idx}:")
        print(f"     Neutral: {neutral_val:.6f}")
        print(f"     'female': {sex_val:.6f} (diff: {sex_val-neutral_val:+.6f})")
        print(f"     'She': {pronoun_val:.6f} (diff: {pronoun_val-neutral_val:+.6f})")

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
        discover_active_features()
