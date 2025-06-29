#!/usr/bin/env python3
"""
Comprehensive analysis of age-responsive features across different age groups.
"""

import torch
import json
from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.analyse import run_prompt

def comprehensive_age_analysis():
    """Analyze age-responsive features across all age groups."""
    print("🔍 COMPREHENSIVE AGE FEATURE ANALYSIS")
    print("=" * 70)
    
    # Load model
    print("📥 Loading model...")
    model, sae = load_model_and_sae()
    print("✅ Model loaded")
    
    # Define comprehensive age prompts
    age_prompts = {
        'neutral': "A patient presents with symptoms not specified",
        
        # Young variations
        'young_numeric_15': "A 15-year-old patient presents with symptoms not specified",
        'young_numeric_22': "A 22-year-old patient presents with symptoms not specified",
        'young_word': "A young patient presents with symptoms not specified",
        'teenage': "A teenage patient presents with symptoms not specified",
        'adolescent': "An adolescent patient presents with symptoms not specified",
        
        # Middle-aged variations
        'middle_numeric_45': "A 45-year-old patient presents with symptoms not specified",
        'middle_numeric_49': "A 49-year-old patient presents with symptoms not specified",
        'middle_word': "A middle-aged patient presents with symptoms not specified",
        'adult': "An adult patient presents with symptoms not specified",
        
        # Elderly variations
        'elderly_numeric_75': "A 75-year-old patient presents with symptoms not specified",
        'elderly_numeric_82': "An 82-year-old patient presents with symptoms not specified",
        'elderly_word': "An elderly patient presents with symptoms not specified",
        'old_word': "An old patient presents with symptoms not specified",
        'senior': "A senior patient presents with symptoms not specified",
        
        # Sex variations for comparison
        'female': "A female patient presents with symptoms not specified",
        'male': "A male patient presents with symptoms not specified",
        'she_pronoun': "She presents with symptoms not specified",
        'he_pronoun': "He presents with symptoms not specified"
    }
    
    print(f"\n🧪 Getting activations for all {len(age_prompts)} prompts...")
    results = {}
    for name, prompt in age_prompts.items():
        print(f"   {name}: {prompt}")
        results[name] = run_prompt(prompt, model, sae, clamping=False, clamp_features={}, clamp_value=1.0)
    
    # Analyze each age category
    age_categories = {
        'young': ['young_numeric_15', 'young_numeric_22', 'young_word', 'teenage', 'adolescent'],
        'middle_aged': ['middle_numeric_45', 'middle_numeric_49', 'middle_word', 'adult'],
        'elderly': ['elderly_numeric_75', 'elderly_numeric_82', 'elderly_word', 'old_word', 'senior'],
        'sex': ['female', 'male', 'she_pronoun', 'he_pronoun']
    }
    
    threshold = 0.001
    category_features = {}
    
    for category, prompt_names in age_categories.items():
        print(f"\n📊 ANALYZING {category.upper()} CATEGORY:")
        
        # Collect all significant features across prompts in this category
        all_significant_features = {}
        
        for prompt_name in prompt_names:
            print(f"\n   🔍 {prompt_name}:")
            print(f"     Prompt: {age_prompts[prompt_name]}")
            
            # Find significant features for this prompt
            significant_features = []
            for i in range(16384):
                neutral_val = results['neutral'][f"activation_{i}"]
                prompt_val = results[prompt_name][f"activation_{i}"]
                diff = prompt_val - neutral_val
                
                if abs(diff) > threshold:
                    significant_features.append({
                        'feature': i,
                        'difference': diff,
                        'neutral': neutral_val,
                        'prompt': prompt_val
                    })
            
            # Sort by absolute difference
            significant_features.sort(key=lambda x: abs(x['difference']), reverse=True)
            print(f"     Found {len(significant_features)} significant features")
            
            # Show top 5
            for j, feat in enumerate(significant_features[:5]):
                print(f"       {j+1}. Feature {feat['feature']}: {feat['difference']:+.4f}")
                
                # Add to category collection
                if feat['feature'] not in all_significant_features:
                    all_significant_features[feat['feature']] = []
                all_significant_features[feat['feature']].append(feat['difference'])
        
        # Find consensus features for this category (appear in multiple prompts)
        consensus_features = {}
        for feature_id, differences in all_significant_features.items():
            if len(differences) >= 2:  # Appears in at least 2 prompts
                avg_diff = sum(differences) / len(differences)
                consensus_features[feature_id] = {
                    'avg_difference': avg_diff,
                    'count': len(differences),
                    'all_differences': differences
                }
        
        # Sort consensus features by average absolute difference
        sorted_consensus = sorted(consensus_features.items(), 
                                key=lambda x: abs(x[1]['avg_difference']), reverse=True)
        
        print(f"\n   🎯 CONSENSUS FEATURES for {category} (appear in ≥2 prompts):")
        category_top_features = {}
        for i, (feature_id, data) in enumerate(sorted_consensus[:10]):
            print(f"     {i+1}. Feature {feature_id}: avg={data['avg_difference']:+.4f}, count={data['count']}, values={[f'{d:+.3f}' for d in data['all_differences']]}")
            if i < 5:  # Take top 5 for testing
                category_top_features[feature_id] = data['avg_difference']
        
        category_features[category] = category_top_features
        
        # Test clamping with consensus features
        if category_top_features:
            print(f"\n   🔧 Testing clamping with consensus features:")
            
            # Test against the first prompt in the category
            test_prompt_name = prompt_names[0]
            target_result = results[test_prompt_name]
            
            clamped_result = run_prompt(age_prompts['neutral'], model, sae, clamping=True, 
                                      clamp_features=category_top_features, clamp_value=1.0)
            
            similarity = calculate_cosine_similarity(target_result, clamped_result)
            print(f"     Clamping similarity vs {test_prompt_name}: {similarity:.3f}")
    
    # Generate updated constants
    print(f"\n🔧 GENERATING UPDATED CONSTANTS:")
    print("=" * 50)
    
    # Create new feature dictionaries
    new_constants = {
        'YOUNG_FEATURES_WITH_DIRECTIONS': category_features.get('young', {}),
        'MIDDLE_AGE_FEATURES_WITH_DIRECTIONS': category_features.get('middle_aged', {}),
        'OLD_FEATURES_WITH_DIRECTIONS': category_features.get('elderly', {}),
        'FEMALE_FEATURES_WITH_DIRECTIONS': {},
        'MALE_FEATURES_WITH_DIRECTIONS': {}
    }
    
    # Extract sex features separately
    if 'sex' in category_features:
        # For sex, we need to separate male vs female features
        # Use female and she_pronoun for female features
        female_features = {}
        male_features = {}
        
        # Analyze female-specific features
        for prompt_name in ['female', 'she_pronoun']:
            for i in range(16384):
                neutral_val = results['neutral'][f"activation_{i}"]
                prompt_val = results[prompt_name][f"activation_{i}"]
                diff = prompt_val - neutral_val
                
                if abs(diff) > threshold:
                    if i not in female_features:
                        female_features[i] = []
                    female_features[i].append(diff)
        
        # Analyze male-specific features  
        for prompt_name in ['male', 'he_pronoun']:
            for i in range(16384):
                neutral_val = results['neutral'][f"activation_{i}"]
                prompt_val = results[prompt_name][f"activation_{i}"]
                diff = prompt_val - neutral_val
                
                if abs(diff) > threshold:
                    if i not in male_features:
                        male_features[i] = []
                    male_features[i].append(diff)
        
        # Get consensus features
        female_consensus = {k: sum(v)/len(v) for k, v in female_features.items() if len(v) >= 1}
        male_consensus = {k: sum(v)/len(v) for k, v in male_features.items() if len(v) >= 1}
        
        # Take top 5 for each
        female_sorted = sorted(female_consensus.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        male_sorted = sorted(male_consensus.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        new_constants['FEMALE_FEATURES_WITH_DIRECTIONS'] = dict(female_sorted)
        new_constants['MALE_FEATURES_WITH_DIRECTIONS'] = dict(male_sorted)
    
    # Print the new constants
    for const_name, features in new_constants.items():
        print(f"\n{const_name} = {{")
        for feature_id, value in features.items():
            print(f"    {feature_id}: {value:.3f},")
        print("}")
    
    return new_constants

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
        new_constants = comprehensive_age_analysis()
        
        # Save to JSON for easy access
        with open('discovered_demographic_features.json', 'w') as f:
            json.dump(new_constants, f, indent=2)
        print(f"\n💾 Saved discovered features to 'discovered_demographic_features.json'")
