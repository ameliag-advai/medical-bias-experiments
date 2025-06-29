#!/usr/bin/env python3
"""
Patient-Specific Equivalence Test: 
Compare each patient's original demographics in prompt vs patient-specific 1x clamping

For each patient, we test:
1. Demographics in prompt (e.g., "45-year-old male patient has chest pain")
2. Neutral prompt + 1x clamping matching their demographics

This validates that 1x clamping is equivalent to including demographics in prompts.
"""

import argparse
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Try to import pandas, handle if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas not available, using basic CSV parsing")

# Demographic feature mappings (from your analysis)
DEMOGRAPHIC_FEATURES = {
    # Age Groups
    "pediatric": {
        3296: 0.083,  # Medical conditions related to babies
        14423: 0.143, # Events and activities for children  
        5565: 0.420   # Ages or age-related terms (averaged infant/child)
    },
    
    "adolescent": {
        801: 0.573,   # References to teenagers (strongest signal)
        7398: 0.261,  # References to specific age groups
        5565: 0.398   # Ages or age-related terms
    },
    
    "young_adult": {
        7398: 0.184,  # References to specific age groups
        1999: 0.113,  # Mentions of specific ages and life events
        5565: 0.382   # Ages or age-related terms
    },
    
    "middle_age": {
        13032: 0.333, # Numeric information related to ages/dates (peak)
        1999: 0.135,  # Mentions of specific ages and life events (peak)
        11060: 0.409, # Actions/events related to caring for elderly
        5565: 0.370   # Ages or age-related terms
    },
    
    "senior": {
        11060: 1.456, # Actions/events related to caring for elderly (peak)
        13032: 0.200, # Numeric information related to ages/dates
        6679: 0.170,  # People's ages and significant life events
        5565: 0.359   # Ages or age-related terms
    },
    
    # Gender Groups
    "male": {
        11096: 0.2,   # Pronouns and possessive pronouns denoting male entities
        13353: 0.2,   # Mentions of gender, particularly focusing on males
        8409: 0.2,    # References to a boy or boys
        12221: 0.2    # Phrases related to masculinity
    },
    
    "female": {
        387: 0.2,     # References to a specific female pronoun
        6221: 0.2,    # References to females
        5176: 0.2,    # References to or characteristics of girls
        12813: 0.2    # References to gender/sex-related topics, especially female issues
    }
}

def map_age_to_group(age):
    """Map numeric age to age group."""
    if age is None or str(age).lower() in ['nan', 'none', '']:
        return None
    try:
        age = int(float(age))
        if age <= 12:
            return "pediatric"
        elif age <= 19:
            return "adolescent"
        elif age <= 35:
            return "young_adult"
        elif age <= 64:
            return "middle_age"
        else:
            return "senior"
    except (ValueError, TypeError):
        return None

def normalize_sex(sex):
    """Normalize sex values to male/female."""
    if sex is None or str(sex).lower() in ['nan', 'none', '']:
        return None
    sex = str(sex).lower().strip()
    if sex in ['m', 'male', 'man']:
        return 'male'
    elif sex in ['f', 'female', 'woman']:
        return 'female'
    else:
        return None

def load_patient_demographics_pandas(patient_file, num_cases):
    """Load patient demographics using pandas."""
    print(f"📊 Loading patient data from {patient_file}")
    df = pd.read_csv(patient_file)
    
    # Take first num_cases patients
    test_patients = df.head(num_cases)
    print(f"📋 Processing first {len(test_patients)} patients")
    
    # Extract demographics - try multiple column names
    age_cols = ['age', 'dataset_age', 'Age', 'AGE']
    sex_cols = ['sex', 'dataset_sex', 'Sex', 'SEX', 'gender', 'Gender']
    
    age_col = None
    sex_col = None
    
    for col in age_cols:
        if col in test_patients.columns:
            age_col = col
            break
    
    for col in sex_cols:
        if col in test_patients.columns:
            sex_col = col
            break
    
    if not age_col:
        print("❌ No age column found. Available columns:")
        print(list(test_patients.columns))
        return []
    
    if not sex_col:
        print("❌ No sex column found. Available columns:")
        print(list(test_patients.columns))
        return []
    
    print(f"✅ Using age column: {age_col}")
    print(f"✅ Using sex column: {sex_col}")
    
    demographics_data = []
    
    for idx, row in test_patients.iterrows():
        age = row.get(age_col)
        sex = normalize_sex(row.get(sex_col))
        age_group = map_age_to_group(age)
        
        # Extract symptoms/case info
        case_text = ""
        if 'case_text' in row:
            case_text = row['case_text']
        elif 'symptoms' in row:
            case_text = row['symptoms']
        elif 'text' in row:
            case_text = row['text']
        else:
            # Try to find a text column
            text_cols = [col for col in row.index if 'text' in col.lower() or 'symptom' in col.lower()]
            if text_cols:
                case_text = row[text_cols[0]]
        
        demographics_data.append({
            'case_id': idx,
            'age': age,
            'age_group': age_group,
            'sex': sex,
            'case_text': str(case_text) if case_text else "symptoms not specified",
            'original_row': dict(row)
        })
    
    return demographics_data

def load_patient_demographics_basic(patient_file, num_cases):
    """Load patient demographics using basic CSV parsing."""
    print(f"📊 Loading patient data from {patient_file} (basic parsing)")
    
    demographics_data = []
    
    with open(patient_file, 'r') as f:
        lines = f.readlines()
        
        if not lines:
            print("❌ Empty file")
            return []
        
        # Parse header
        header = lines[0].strip().split(',')
        print(f"📋 CSV columns: {header}")
        
        # Find age and sex columns
        age_col_idx = None
        sex_col_idx = None
        
        for i, col in enumerate(header):
            col_lower = col.lower()
            if col_lower in ['age', 'dataset_age']:
                age_col_idx = i
            elif col_lower in ['sex', 'dataset_sex', 'gender']:
                sex_col_idx = i
        
        if age_col_idx is None:
            print("❌ No age column found")
            return []
        if sex_col_idx is None:
            print("❌ No sex column found")
            return []
        
        print(f"✅ Using age column: {header[age_col_idx]} (index {age_col_idx})")
        print(f"✅ Using sex column: {header[sex_col_idx]} (index {sex_col_idx})")
        
        # Parse data rows
        for idx, line in enumerate(lines[1:num_cases+1]):
            if not line.strip():
                continue
                
            fields = line.strip().split(',')
            if len(fields) <= max(age_col_idx, sex_col_idx):
                continue
            
            age = fields[age_col_idx] if age_col_idx < len(fields) else None
            sex = fields[sex_col_idx] if sex_col_idx < len(fields) else None
            
            age_group = map_age_to_group(age)
            sex = normalize_sex(sex)
            
            demographics_data.append({
                'case_id': idx,
                'age': age,
                'age_group': age_group,
                'sex': sex,
                'case_text': "patient presents with symptoms",  # Default case text
                'original_row': dict(zip(header, fields))
            })
    
    return demographics_data

def load_patient_demographics(patient_file, num_cases):
    """Load patient demographics using available method."""
    if HAS_PANDAS:
        return load_patient_demographics_pandas(patient_file, num_cases)
    else:
        return load_patient_demographics_basic(patient_file, num_cases)

def create_demographic_prompt(case_text, age, sex, age_group):
    """Create a prompt with demographic information."""
    # Build demographic description
    demo_parts = []
    
    if age and str(age).lower() not in ['nan', 'none', '']:
        try:
            demo_parts.append(f"{int(float(age))}-year-old")
        except (ValueError, TypeError):
            pass
    
    if sex:
        demo_parts.append(sex)
    
    if demo_parts:
        demo_str = " ".join(demo_parts)
        return f"A {demo_str} patient presents with {case_text}"
    else:
        return f"A patient presents with {case_text}"

def create_neutral_prompt(case_text):
    """Create a neutral prompt without demographic information."""
    return f"A patient presents with {case_text}"

def apply_demographic_clamping(demographic_groups, intensity=1.0):
    """
    Generate the feature clamping values for given demographic groups.
    Returns a dictionary of feature_id -> activation_value
    """
    clamping_values = {}
    
    for group in demographic_groups:
        if group in DEMOGRAPHIC_FEATURES:
            for feature_idx, base_value in DEMOGRAPHIC_FEATURES[group].items():
                clamping_values[feature_idx] = base_value * intensity
    
    return clamping_values

def run_equivalence_test(demographics_data, output_dir):
    """Run the patient-specific equivalence test."""
    
    print(f"\n🧪 RUNNING PATIENT-SPECIFIC EQUIVALENCE TEST")
    print(f"📁 Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze demographics distribution
    age_groups = {}
    sex_groups = {}
    valid_cases = 0
    
    for patient in demographics_data:
        if patient['age_group']:
            age_groups[patient['age_group']] = age_groups.get(patient['age_group'], 0) + 1
        if patient['sex']:
            sex_groups[patient['sex']] = sex_groups.get(patient['sex'], 0) + 1
        if patient['age_group'] and patient['sex'] and patient['case_text']:
            valid_cases += 1
    
    print(f"\n📈 Demographics Distribution:")
    print(f"   Age groups: {age_groups}")
    print(f"   Sex groups: {sex_groups}")
    print(f"   Valid cases (have age, sex, and symptoms): {valid_cases}")
    
    # Generate test cases
    test_cases = []
    
    for patient in demographics_data:
        age = patient['age']
        age_group = patient['age_group']
        sex = patient['sex']
        case_text = patient['case_text']
        case_id = patient['case_id']
        
        # Skip if missing critical info
        if not age_group or not sex or not case_text:
            continue
        
        # Test Case 1: Demographics in prompt vs neutral + demographic clamping
        demo_prompt = create_demographic_prompt(case_text, age, sex, age_group)
        neutral_prompt = create_neutral_prompt(case_text)
        
        # Get clamping values for this patient's demographics
        demographic_groups = [age_group, sex]
        clamping_values = apply_demographic_clamping(demographic_groups, intensity=1.0)
        
        test_case = {
            'case_id': case_id,
            'age': age,
            'age_group': age_group,
            'sex': sex,
            'test_type': 'full_demographics',
            'prompt_with_demographics': demo_prompt,
            'neutral_prompt': neutral_prompt,
            'clamping_features': clamping_values,
            'expected_result': 'equivalent'
        }
        test_cases.append(test_case)
        
        # Test Case 2: Age-only comparison
        if age and str(age).lower() not in ['nan', 'none', '']:
            try:
                age_only_prompt = f"A {int(float(age))}-year-old patient presents with {case_text}"
                age_only_clamping = apply_demographic_clamping([age_group], intensity=1.0)
                
                test_case_age = {
                    'case_id': case_id,
                    'age': age,
                    'age_group': age_group,
                    'sex': sex,
                    'test_type': 'age_only',
                    'prompt_with_demographics': age_only_prompt,
                    'neutral_prompt': neutral_prompt,
                    'clamping_features': age_only_clamping,
                    'expected_result': 'equivalent'
                }
                test_cases.append(test_case_age)
            except (ValueError, TypeError):
                pass
        
        # Test Case 3: Sex-only comparison
        sex_only_prompt = f"A {sex} patient presents with {case_text}"
        sex_only_clamping = apply_demographic_clamping([sex], intensity=1.0)
        
        test_case_sex = {
            'case_id': case_id,
            'age': age,
            'age_group': age_group,
            'sex': sex,
            'test_type': 'sex_only',
            'prompt_with_demographics': sex_only_prompt,
            'neutral_prompt': neutral_prompt,
            'clamping_features': sex_only_clamping,
            'expected_result': 'equivalent'
        }
        test_cases.append(test_case_sex)
    
    print(f"\n✅ Generated {len(test_cases)} test cases")
    
    # Save test cases to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"patient_specific_equivalence_tests_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(test_cases, f, indent=2, default=str)
    
    print(f"💾 Test cases saved to: {output_file}")
    
    # Create summary CSV if pandas available
    if HAS_PANDAS:
        summary_data = []
        for test_case in test_cases:
            summary_data.append({
                'case_id': test_case['case_id'],
                'age': test_case['age'],
                'age_group': test_case['age_group'],
                'sex': test_case['sex'],
                'test_type': test_case['test_type'],
                'num_clamping_features': len(test_case['clamping_features']),
                'clamping_features': list(test_case['clamping_features'].keys())
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, f"equivalence_test_summary_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"📊 Summary saved to: {summary_file}")
    
    # Print example test cases
    print(f"\n📋 EXAMPLE TEST CASES:")
    for i, test_case in enumerate(test_cases[:6]):  # Show first 6
        print(f"\n   Case {i+1} ({test_case['test_type']}):")
        print(f"     Patient: {test_case['age']}-year-old {test_case['sex']} ({test_case['age_group']})")
        print(f"     Prompt A: {test_case['prompt_with_demographics'][:100]}...")
        print(f"     Prompt B: {test_case['neutral_prompt'][:60]}... + clamping {list(test_case['clamping_features'].keys())}")
    
    return test_cases, output_file

def print_usage_instructions(output_file):
    """Print instructions on how to use the generated test cases."""
    
    print(f"\n" + "="*80)
    print("🔧 HOW TO USE THESE TEST CASES")
    print("="*80)
    
    print(f"""
To run the equivalence test with your model pipeline:

1. 📂 Load the test cases:
   test_cases = json.load(open('{output_file}'))

2. 🤖 For each test case, run both conditions:
   
   # Method A: Prompt with demographics
   diagnosis_a = your_model.diagnose(test_case['prompt_with_demographics'])
   
   # Method B: Neutral prompt + feature clamping
   diagnosis_b = your_model.diagnose_with_clamping(
       test_case['neutral_prompt'],
       clamping_features=test_case['clamping_features']
   )

3. 📊 Compare the results:
   equivalence_rate = sum(1 for a, b in zip(diagnoses_a, diagnoses_b) if a == b) / len(test_cases)
   
4. ✅ Expected outcome: >90% equivalence rate validates your feature identification

📝 The test validates that:
   - "45-year-old male patient has chest pain" (prompt)
   ≈ "Patient has chest pain" + male & middle_age feature clamping
""")

def main():
    parser = argparse.ArgumentParser(description="Run patient-specific 1x clamping equivalence test")
    parser.add_argument("--patient-file", default="release-test-patients-age-grouped.csv",
                       help="Patient data file")
    parser.add_argument("--num-cases", type=int, default=100,
                       help="Number of cases to process")
    parser.add_argument("--output-dir", default="patient_specific_equivalence",
                       help="Output directory for test cases")
    
    args = parser.parse_args()
    
    print("🧪 PATIENT-SPECIFIC EQUIVALENCE TEST")
    print(f"📁 Patient file: {args.patient_file}")
    print(f"📊 Cases: {args.num_cases}")
    print(f"📂 Output: {args.output_dir}")
    print()
    
    # Load patient demographics
    demographics_data = load_patient_demographics(args.patient_file, args.num_cases)
    
    if not demographics_data:
        print("❌ Failed to load patient demographics")
        return
    
    # Run equivalence test
    test_cases, output_file = run_equivalence_test(demographics_data, args.output_dir)
    
    # Print usage instructions
    print_usage_instructions(output_file)
    
    print(f"\n🎯 SUCCESS: Patient-specific equivalence test cases generated!")
    print(f"📊 Total test cases: {len(test_cases)}")
    print(f"💾 Output file: {output_file}")

if __name__ == "__main__":
    main()
