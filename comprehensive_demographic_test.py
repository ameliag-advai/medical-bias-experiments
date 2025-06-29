#!/usr/bin/env python3
"""
Comprehensive Demographic Clamping Test
Tests 100 cases with varying intensities for a 50-year-old male patient example.
"""

import torch
import json
import os
from datetime import datetime
from tqdm import tqdm
from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.analyse import run_prompt
from src.advai.analysis.constants_v2 import (
    MIDDLE_AGE_FEATURES_WITH_DIRECTIONS,
    MALE_FEATURES_WITH_DIRECTIONS
)

def generate_test_cases():
    """Generate 100 test cases with varying intensities and combinations."""
    
    # Base patient demographics
    patient_age = 50
    patient_sex = "male"
    
    # Various symptoms to test
    symptoms = [
        "chest pain", "headache", "fatigue", "back pain", "abdominal pain",
        "shortness of breath", "dizziness", "nausea", "joint pain", "muscle weakness"
    ]
    
    # Intensity levels to test
    intensities = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    
    test_cases = []
    case_id = 1
    
    # For each symptom, create 10 test cases (100 total)
    for symptom in symptoms:
        for intensity in intensities:
            
            # Test case structure
            case = {
                'case_id': case_id,
                'patient_demographics': {
                    'age': patient_age,
                    'sex': patient_sex,
                    'age_group': 'middle_age'
                },
                'symptom': symptom,
                'intensity': intensity,
                'prompts': {
                    # Full demographic prompt (target)
                    'full_demo': f"A {patient_age}-year-old {patient_sex} patient presents with {symptom}",
                    
                    # Age-only prompt  
                    'age_only': f"A {patient_age}-year-old patient presents with {symptom}",
                    
                    # Sex-only prompt
                    'sex_only': f"A {patient_sex} patient presents with {symptom}",
                    
                    # Neutral prompt (for clamping)
                    'neutral': f"A patient presents with {symptom}"
                },
                'clamping_scenarios': {
                    # Clamp neutral to male only
                    'male_only': {
                        'prompt': f"A patient presents with {symptom}",
                        'features': MALE_FEATURES_WITH_DIRECTIONS,
                        'intensity': intensity
                    },
                    
                    # Clamp neutral to middle-age only  
                    'age_only': {
                        'prompt': f"A patient presents with {symptom}",
                        'features': MIDDLE_AGE_FEATURES_WITH_DIRECTIONS,
                        'intensity': intensity
                    },
                    
                    # Clamp neutral to both male + middle-age
                    'combined': {
                        'prompt': f"A patient presents with {symptom}",
                        'features': {**MALE_FEATURES_WITH_DIRECTIONS, **MIDDLE_AGE_FEATURES_WITH_DIRECTIONS},
                        'intensity': intensity
                    },
                    
                    # Clamp sex-only prompt to middle-age
                    'sex_plus_age_clamp': {
                        'prompt': f"A {patient_sex} patient presents with {symptom}",
                        'features': MIDDLE_AGE_FEATURES_WITH_DIRECTIONS,
                        'intensity': intensity
                    },
                    
                    # Clamp age-only prompt to male
                    'age_plus_sex_clamp': {
                        'prompt': f"A {patient_age}-year-old patient presents with {symptom}",
                        'features': MALE_FEATURES_WITH_DIRECTIONS,
                        'intensity': intensity
                    }
                }
            }
            
            test_cases.append(case)
            case_id += 1
    
    return test_cases

def run_comprehensive_test():
    """Run the comprehensive demographic clamping test."""
    
    print("🧪 COMPREHENSIVE DEMOGRAPHIC CLAMPING TEST")
    print("=" * 70)
    print("👤 Patient: 50-year-old male")
    print("🔬 Testing 100 cases with varying intensities")
    print("📊 5 clamping scenarios per case")
    print()
    
    # Generate test cases
    print("📋 Generating test cases...")
    test_cases = generate_test_cases()
    print(f"✅ Generated {len(test_cases)} test cases")
    
    # Load model
    print("📥 Loading model and SAE...")
    model, sae = load_model_and_sae()
    print("✅ Model loaded")
    print()
    
    # Run tests
    results = []
    
    print("🚀 Running comprehensive test...")
    for case in tqdm(test_cases, desc="Processing cases"):
        
        case_results = {
            'case_id': case['case_id'],
            'patient_demographics': case['patient_demographics'],
            'symptom': case['symptom'],
            'intensity': case['intensity'],
            'results': {}
        }
        
        # Get baseline results (targets for comparison)
        targets = {}
        for prompt_type, prompt in case['prompts'].items():
            targets[prompt_type] = run_prompt(prompt, model, sae, 
                                            clamping=False, clamp_features={}, clamp_value=1.0)
        
        # Test each clamping scenario
        for scenario_name, scenario in case['clamping_scenarios'].items():
            
            # Run clamped prompt
            clamped_result = run_prompt(
                scenario['prompt'], model, sae,
                clamping=True, 
                clamp_features=scenario['features'],
                clamp_value=scenario['intensity']
            )
            
            # Compare with appropriate target
            if scenario_name == 'male_only':
                target = targets['sex_only']
                comparison_target = 'sex_only'
            elif scenario_name == 'age_only':
                target = targets['age_only'] 
                comparison_target = 'age_only'
            elif scenario_name in ['combined', 'sex_plus_age_clamp', 'age_plus_sex_clamp']:
                target = targets['full_demo']
                comparison_target = 'full_demo'
            else:
                target = targets['full_demo']
                comparison_target = 'full_demo'
            
            # Calculate similarity
            similarity = calculate_cosine_similarity(target, clamped_result)
            equivalent = similarity > 0.8
            
            case_results['results'][scenario_name] = {
                'similarity': similarity,
                'equivalent': equivalent,
                'comparison_target': comparison_target,
                'features_used': list(scenario['features'].keys()),
                'num_features': len(scenario['features'])
            }
        
        results.append(case_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"comprehensive_demographic_test_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'test_type': 'comprehensive_demographic_clamping',
                'patient_demographics': {'age': 50, 'sex': 'male', 'age_group': 'middle_age'},
                'total_cases': len(test_cases),
                'scenarios_per_case': 5,
                'total_tests': len(test_cases) * 5,
                'timestamp': timestamp
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Generate summary
    generate_summary(results)

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

def generate_summary(results):
    """Generate and display summary statistics."""
    
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    
    # Overall statistics
    total_tests = len(results) * 5  # 5 scenarios per case
    all_similarities = []
    all_equivalences = []
    
    scenario_stats = {}
    intensity_stats = {}
    
    for case in results:
        intensity = case['intensity']
        if intensity not in intensity_stats:
            intensity_stats[intensity] = {'similarities': [], 'equivalences': []}
        
        for scenario_name, scenario_result in case['results'].items():
            similarity = scenario_result['similarity']
            equivalent = scenario_result['equivalent']
            
            all_similarities.append(similarity)
            all_equivalences.append(equivalent)
            
            # By scenario
            if scenario_name not in scenario_stats:
                scenario_stats[scenario_name] = {'similarities': [], 'equivalences': []}
            scenario_stats[scenario_name]['similarities'].append(similarity)
            scenario_stats[scenario_name]['equivalences'].append(equivalent)
            
            # By intensity
            intensity_stats[intensity]['similarities'].append(similarity)
            intensity_stats[intensity]['equivalences'].append(equivalent)
    
    # Overall stats
    avg_similarity = sum(all_similarities) / len(all_similarities)
    equivalence_rate = (sum(all_equivalences) / len(all_equivalences)) * 100
    
    print(f"🎯 OVERALL RESULTS:")
    print(f"   Total tests: {total_tests}")
    print(f"   Average similarity: {avg_similarity:.3f}")
    print(f"   Equivalence rate: {equivalence_rate:.1f}%")
    print(f"   Equivalent tests: {sum(all_equivalences)}/{len(all_equivalences)}")
    
    # By scenario
    print(f"\n📋 BY SCENARIO:")
    for scenario, stats in scenario_stats.items():
        avg_sim = sum(stats['similarities']) / len(stats['similarities'])
        equiv_rate = (sum(stats['equivalences']) / len(stats['equivalences'])) * 100
        print(f"   {scenario:20s}: {avg_sim:.3f} similarity, {equiv_rate:5.1f}% equivalent")
    
    # By intensity
    print(f"\n⚡ BY INTENSITY:")
    for intensity in sorted(intensity_stats.keys()):
        stats = intensity_stats[intensity]
        avg_sim = sum(stats['similarities']) / len(stats['similarities'])
        equiv_rate = (sum(stats['equivalences']) / len(stats['equivalences'])) * 100
        print(f"   {intensity:4.2f}x: {avg_sim:.3f} similarity, {equiv_rate:5.1f}% equivalent")
    
    # Best performing
    best_scenario = max(scenario_stats.items(), key=lambda x: sum(x[1]['similarities'])/len(x[1]['similarities']))
    best_intensity = max(intensity_stats.items(), key=lambda x: sum(x[1]['similarities'])/len(x[1]['similarities']))
    
    print(f"\n🏆 BEST PERFORMING:")
    print(f"   Scenario: {best_scenario[0]} ({sum(best_scenario[1]['similarities'])/len(best_scenario[1]['similarities']):.3f} avg similarity)")
    print(f"   Intensity: {best_intensity[0]:.2f}x ({sum(best_intensity[1]['similarities'])/len(best_intensity[1]['similarities']):.3f} avg similarity)")
    
    if equivalence_rate >= 90:
        print(f"\n🎉 EXCELLENT! {equivalence_rate:.1f}% equivalence rate - demographic clamping is highly effective!")
    elif equivalence_rate >= 70:
        print(f"\n✅ GOOD! {equivalence_rate:.1f}% equivalence rate - demographic clamping works well!")
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT: {equivalence_rate:.1f}% equivalence rate")

if __name__ == "__main__":
    with torch.no_grad():
        run_comprehensive_test()
