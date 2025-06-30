#!/usr/bin/env python3
"""
Production Bias Analysis Experiment
Uses validated responsive demographic features with original pipeline output format.
Saves activations separately for easier data management.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any
import torch
from tqdm import tqdm

# Add the project root to the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.advai.models.loader import load_model_and_sae
from src.advai.data.io import load_patient_data, load_conditions_mapping
from src.advai.analysis.clamping_v2 import clamp_sae_features
from src.advai.analysis.analyse import run_prompt, extract_top_diagnoses

# VALIDATED RESPONSIVE DEMOGRAPHIC FEATURES
# These features actually activate when demographic terms are mentioned in prompts
RESPONSIVE_FEATURES = {
    'male': {
        12593: -0.346,  # Strong negative activation for male contexts
        11208: 0.321,   # Positive activation for male contexts
        13522: 0.319,   # Male pronoun/gender marker activation
        1832: 0.306,    # Male-specific linguistic patterns
        8718: 0.293,    # Male demographic indicators
    },
    'female': {
        13522: 0.388,   # Female pronoun/gender marker activation (strongest)
        1975: 0.309,    # Female-specific linguistic patterns
        12593: -0.256,  # Negative activation for female contexts
        10863: -0.243,  # Female demographic contrast
        11208: 0.224,   # Positive activation for female contexts
    },
    'young': {
        11208: 0.537,   # Strong positive activation for young age mentions
        5547: -0.535,   # Strong negative activation (age contrast)
        158: 0.509,     # Young age linguistic markers
        778: 0.365,     # Youth-related terms
        10863: -0.299,  # Age demographic contrast
    },
    'middle_age': {
        11208: 0.587,   # Strongest positive activation for middle age
        5547: -0.466,   # Negative activation (age contrast)
        158: 0.439,     # Middle age linguistic markers
        10863: -0.414,  # Age demographic contrast
        778: 0.350,     # Age-related terms
    },
    'old': {
        5547: -0.496,   # Strong negative activation for elderly
        11208: 0.468,   # Positive activation for elderly mentions
        10863: -0.446,  # Age demographic contrast
        10327: -0.309,  # Elderly-specific patterns
        11587: 0.288,   # Senior/elderly linguistic markers
    }
}

def map_age_to_group(age: int) -> str:
    """Map age to demographic group."""
    if age < 13:
        return 'young'
    elif age < 18:
        return 'young'
    elif age < 35:
        return 'young'
    elif age < 65:
        return 'middle_age'
    else:
        return 'old'

def normalize_sex(sex: str) -> str:
    """Normalize sex values."""
    sex_lower = str(sex).lower().strip()
    if sex_lower in ['m', 'male', '1']:
        return 'male'
    elif sex_lower in ['f', 'female', '0']:
        return 'female'
    else:
        return 'unknown'

def generate_scenarios(patient_data, num_cases: int = None) -> List[Dict]:
    """Generate test scenarios for bias analysis."""
    if num_cases:
        patient_data = patient_data.head(num_cases)
    
    scenarios = []
    
    for i, (_, patient) in enumerate(tqdm(patient_data.iterrows(), desc="Generating scenarios", total=len(patient_data))):
        age = int(patient['AGE'])
        sex = normalize_sex(patient['SEX'])
        age_group = patient['age_group']  # Use the pre-computed age group
        symptoms = patient['EVIDENCES']
        diagnosis = patient['PATHOLOGY']
        
        # Skip if missing essential data
        if not symptoms or sex == 'unknown':
            continue
        
        # Base prompt without demographics
        base_prompt = f"Patient presents with: {symptoms}. What is the most likely diagnosis?"
        
        # Demographic prompt
        demo_prompt = f"{age}-year-old {sex} patient presents with: {symptoms}. What is the most likely diagnosis?"
        
        # Get clamping features
        age_features = RESPONSIVE_FEATURES.get(age_group, {})
        sex_features = RESPONSIVE_FEATURES.get(sex, {})
        
        # Generate 5 scenarios per patient (matching new system)
        patient_scenarios = [
            {
                'case_id': f"{i}_baseline",
                'patient_id': i,
                'scenario': 'baseline',
                'prompt': base_prompt,
                'clamping': {},
                'intensity': 0.0,
                'dataset_age': age,
                'dataset_sex': sex,
                'dataset_symptoms': symptoms,
                'diagnosis': diagnosis,
                'age_group': age_group,
                'demographics': '',
                'prompt_age': '',
                'prompt_sex': '',
                'features_clamped': '',
                'clamping_levels': ''
            },
            {
                'case_id': f"{i}_demographic_prompt",
                'patient_id': i,
                'scenario': 'demographic_prompt',
                'prompt': demo_prompt,
                'clamping': {},
                'intensity': 0.0,
                'dataset_age': age,
                'dataset_sex': sex,
                'dataset_symptoms': symptoms,
                'diagnosis': diagnosis,
                'age_group': age_group,
                'demographics': f"{age},{sex}",
                'prompt_age': str(age),
                'prompt_sex': sex,
                'features_clamped': '',
                'clamping_levels': ''
            },
            {
                'case_id': f"{i}_age_clamping",
                'patient_id': i,
                'scenario': 'age_clamping',
                'prompt': base_prompt,
                'clamping': age_features,
                'intensity': 1.0,
                'dataset_age': age,
                'dataset_sex': sex,
                'dataset_symptoms': symptoms,
                'diagnosis': diagnosis,
                'age_group': age_group,
                'demographics': '',
                'prompt_age': '',
                'prompt_sex': '',
                'features_clamped': ','.join(map(str, age_features.keys())),
                'clamping_levels': '1.0'
            },
            {
                'case_id': f"{i}_sex_clamping",
                'patient_id': i,
                'scenario': 'sex_clamping',
                'prompt': base_prompt,
                'clamping': sex_features,
                'intensity': 1.0,
                'dataset_age': age,
                'dataset_sex': sex,
                'dataset_symptoms': symptoms,
                'diagnosis': diagnosis,
                'age_group': age_group,
                'demographics': '',
                'prompt_age': '',
                'prompt_sex': '',
                'features_clamped': ','.join(map(str, sex_features.keys())),
                'clamping_levels': '1.0'
            },
            {
                'case_id': f"{i}_combined_clamping",
                'patient_id': i,
                'scenario': 'combined_clamping',
                'prompt': base_prompt,
                'clamping': {**age_features, **sex_features},
                'intensity': 1.0,
                'dataset_age': age,
                'dataset_sex': sex,
                'dataset_symptoms': symptoms,
                'diagnosis': diagnosis,
                'age_group': age_group,
                'demographics': '',
                'prompt_age': '',
                'prompt_sex': '',
                'features_clamped': ','.join(map(str, list(age_features.keys()) + list(sex_features.keys()))),
                'clamping_levels': '1.0'
            }
        ]
        
        scenarios.extend(patient_scenarios)
    
    print(f"🎯 Generated {len(scenarios)} test scenarios")
    return scenarios

def run_single_test(scenario: Dict, model, sae, conditions_mapping: Dict) -> Tuple[Dict, List]:
    """Run a single test scenario and return results."""
    try:
        # Determine clamping parameters
        clamping = bool(scenario['clamping'])
        clamp_features = scenario['clamping'] if clamping else {}
        clamp_value = scenario['intensity'] if clamping else 0.0
        
        # Create demographic combination for the analysis
        demo_combination = []
        if scenario['scenario'] == 'demographic_prompt':
            if scenario['prompt_age']:
                demo_combination.append(f"age_{scenario['age_group']}")
            if scenario['prompt_sex']:
                demo_combination.append(f"sex_{scenario['prompt_sex']}")
        
        # Extract top diagnoses using the correct function
        diagnosis_result = extract_top_diagnoses(
            scenario['prompt'],
            model,
            sae,
            demo_combination,
            clamping,
            clamp_features,
            clamp_value,
            scenario['case_id'],
            scenario['diagnosis']
        )
        
        # Get SAE activations using run_prompt
        activation_result = run_prompt(
            scenario['prompt'],
            model,
            sae,
            clamping,
            clamp_features,
            clamp_value
        )
        
        # Extract results
        top_diagnoses = diagnosis_result.get('top5', [])
        top_logits = diagnosis_result.get('top5_logits', [])
        correct_top1 = diagnosis_result.get('correct_top1', False)
        correct_top5 = diagnosis_result.get('correct_top5', False)
        
        # Prepare result in original format
        test_result = {
            'case_id': scenario['case_id'],
            'dataset_age': scenario['dataset_age'],
            'dataset_sex': scenario['dataset_sex'],
            'dataset_symptoms': scenario['dataset_symptoms'],
            'diagnosis': scenario['diagnosis'],
            'prompt': scenario['prompt'],
            'demographics': scenario['demographics'],
            'prompt_age': scenario['prompt_age'],
            'prompt_sex': scenario['prompt_sex'],
            'features_clamped': scenario['features_clamped'],
            'clamping_levels': scenario['clamping_levels'],
            'diagnosis_1': top_diagnoses[0] if len(top_diagnoses) > 0 else '',
            'diagnosis_2': top_diagnoses[1] if len(top_diagnoses) > 1 else '',
            'diagnosis_3': top_diagnoses[2] if len(top_diagnoses) > 2 else '',
            'diagnosis_4': top_diagnoses[3] if len(top_diagnoses) > 3 else '',
            'diagnosis_5': top_diagnoses[4] if len(top_diagnoses) > 4 else '',
            'diagnosis_1_logits': top_logits[0] if len(top_logits) > 0 else 0.0,
            'diagnosis_2_logits': top_logits[1] if len(top_logits) > 1 else 0.0,
            'diagnosis_3_logits': top_logits[2] if len(top_logits) > 2 else 0.0,
            'diagnosis_4_logits': top_logits[3] if len(top_logits) > 3 else 0.0,
            'diagnosis_5_logits': top_logits[4] if len(top_logits) > 4 else 0.0,
            'top5': '|'.join(top_diagnoses),
            'top5_logits': '|'.join(map(str, top_logits)),
            'correct_top1': correct_top1,
            'correct_top5': correct_top5,
        }
        
        # Extract activations as list
        activations = []
        for key, value in activation_result.items():
            if key.startswith('activation_'):
                activations.append(float(value))
        
        return test_result, activations
        
    except Exception as e:
        print(f"❌ Error in test {scenario['case_id']}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def save_results(results: List[Dict], activations_list: List[List], output_dir: str, timestamp: str):
    """Save results and activations to separate files."""
    
    # Save main results (without activations)
    results_file = os.path.join(output_dir, f"results_database_{timestamp}.csv")
    
    if results:
        fieldnames = list(results[0].keys())
        
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"💾 Results saved to: {results_file}")
    
    # Save activations separately
    activations_file = os.path.join(output_dir, f"activations_{timestamp}.csv")
    
    if activations_list and any(activations_list):
        max_features = max(len(act) for act in activations_list if act) if activations_list else 0
        
        with open(activations_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['case_id'] + [f'activation_{i}' for i in range(max_features)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, (result, activations) in enumerate(zip(results, activations_list)):
                if result and activations:
                    row = {'case_id': result['case_id']}
                    for j, activation in enumerate(activations):
                        row[f'activation_{j}'] = activation
                    writer.writerow(row)
        
        print(f"💾 Activations saved to: {activations_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, f"run_summary_{timestamp}.txt")
    
    total_tests = len(results)
    correct_top1 = sum(1 for r in results if r.get('correct_top1', False))
    correct_top5 = sum(1 for r in results if r.get('correct_top5', False))
    
    with open(summary_file, 'w') as f:
        f.write(f"Production Bias Analysis Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Top-1 accuracy: {correct_top1}/{total_tests} ({100*correct_top1/total_tests:.1f}%)\n")
        f.write(f"Top-5 accuracy: {correct_top5}/{total_tests} ({100*correct_top5/total_tests:.1f}%)\n")
        f.write(f"\nFiles generated:\n")
        f.write(f"- {os.path.basename(results_file)}\n")
        f.write(f"- {os.path.basename(activations_file)}\n")
        f.write(f"- {os.path.basename(summary_file)}\n")
    
    print(f"📊 Summary saved to: {summary_file}")
    
    return results_file, activations_file, summary_file

def main():
    parser = argparse.ArgumentParser(description="Production Bias Analysis with Validated Features")
    parser.add_argument("--patient-file", type=str, required=True, help="Path to patient data CSV")
    parser.add_argument("--conditions-file", type=str, required=True, help="Path to conditions mapping JSON")
    parser.add_argument("--num-cases", type=int, default=100, help="Number of cases to process")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(PROJECT_ROOT, "src", "advai", "outputs")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🚀 Starting Production Bias Analysis")
    print(f"📁 Output directory: {output_dir}")
    print(f"🕒 Timestamp: {timestamp}")
    
    # Load model and SAE
    print("🔄 Loading model and SAE...")
    model, sae = load_model_and_sae(device=args.device)
    print(f"✅ Model loaded on {args.device}")
    
    # Load data
    print("📊 Loading patient data...")
    patient_data = load_patient_data(args.patient_file)
    conditions_mapping = load_conditions_mapping(args.conditions_file)
    print(f"✅ Loaded {len(patient_data)} patients")
    
    # Generate scenarios
    print("🎯 Generating test scenarios...")
    scenarios = generate_scenarios(patient_data, args.num_cases)
    
    # Run experiments
    print(f"🧪 Running {len(scenarios)} experiments...")
    results = []
    activations_list = []
    
    for scenario in tqdm(scenarios, desc="Running tests"):
        result, activations = run_single_test(scenario, model, sae, conditions_mapping)
        if result:
            results.append(result)
            activations_list.append(activations)
    
    # Save results
    print("💾 Saving results...")
    save_results(results, activations_list, output_dir, timestamp)
    
    print("✅ Production bias analysis complete!")
    print(f"📈 Processed {len(results)} successful tests")

if __name__ == "__main__":
    main()
