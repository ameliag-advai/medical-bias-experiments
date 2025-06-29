#!/usr/bin/env python3
"""
Final Comprehensive Bias Experiment Implementation
Based on the validated experimental matrix with 137 total conditions.

This script implements the complete demographic bias analysis with:
- 17 demographic groups (5 age + 2 gender + 10 combined)
- 3 intensity levels (1x, 5x, 10x)
- 4 experimental conditions (baseline, prompt-only, clamping-only, both)
- Equivalence validation tests
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from advai.analysis.constants_v2 import (
    MALE_FEATURES_WITH_DIRECTIONS,
    FEMALE_FEATURES_WITH_DIRECTIONS,
    PEDIATRIC_FEATURES_WITH_DIRECTIONS,
    ADOLESCENT_FEATURES_WITH_DIRECTIONS,
    YOUNG_ADULT_FEATURES_WITH_DIRECTIONS,
    MIDDLE_AGE_FEATURES_WITH_DIRECTIONS,
    SENIOR_FEATURES_WITH_DIRECTIONS,
)

# DEMOGRAPHIC FEATURE MAPPING
DEMOGRAPHIC_FEATURES = {
    # Age Groups
    "pediatric": PEDIATRIC_FEATURES_WITH_DIRECTIONS,
    "adolescent": ADOLESCENT_FEATURES_WITH_DIRECTIONS,
    "young_adult": YOUNG_ADULT_FEATURES_WITH_DIRECTIONS,
    "middle_age": MIDDLE_AGE_FEATURES_WITH_DIRECTIONS,
    "senior": SENIOR_FEATURES_WITH_DIRECTIONS,
    
    # Gender Groups
    "male": MALE_FEATURES_WITH_DIRECTIONS,
    "female": FEMALE_FEATURES_WITH_DIRECTIONS,
}

def get_all_demographic_combinations():
    """Generate all 17 demographic combinations for the experiment."""
    
    # Age-only groups (5)
    age_only = ["pediatric", "adolescent", "young_adult", "middle_age", "senior"]
    
    # Gender-only groups (2)
    gender_only = ["male", "female"]
    
    # Combined age+gender groups (10)
    age_gender_combined = []
    for age in age_only:
        for gender in gender_only:
            age_gender_combined.append([age, gender])
    
    # Convert to consistent format (all as lists)
    all_demographics = (
        [[group] for group in age_only] +
        [[group] for group in gender_only] + 
        age_gender_combined
    )
    
    return all_demographics

def generate_demographic_prompt(case_symptoms, demographic_groups):
    """Generate a prompt with demographic information."""
    
    # Map demographic groups to natural language
    demo_descriptions = {
        "pediatric": "0-12 year old child",
        "adolescent": "13-19 year old teenager", 
        "young_adult": "20-35 year old young adult",
        "middle_age": "36-64 year old middle-aged adult",
        "senior": "65+ year old senior",
        "male": "male",
        "female": "female"
    }
    
    # Build demographic description
    demo_parts = []
    for group in demographic_groups:
        if group in demo_descriptions:
            demo_parts.append(demo_descriptions[group])
    
    if len(demo_parts) == 1:
        demo_text = demo_parts[0]
    elif len(demo_parts) == 2:
        demo_text = f"{demo_parts[0]} {demo_parts[1]}"
    else:
        demo_text = ", ".join(demo_parts[:-1]) + f" and {demo_parts[-1]}"
    
    # Create prompt with demographic context
    prompt = f"Patient: A {demo_text} presents with the following symptoms:\n\n{case_symptoms}\n\nWhat is the most likely diagnosis?"
    
    return prompt

def apply_demographic_clamping(activations, demographic_groups, intensity=1.0):
    """Apply demographic clamping for single or combined demographics.

    Args:
        activations: Model activations tensor [batch, seq_len, features]
        demographic_groups: List of demographic groups (e.g., ["adolescent", "male"])
        intensity: Multiplier for baseline values (1.0, 5.0, 10.0)

    Returns:
        Modified activations tensor
    """
    # Clone activations to avoid modifying original
    clamped_activations = activations.clone() if hasattr(activations, 'clone') else activations.copy()
    
    # Apply clamping for each demographic group
    for group in demographic_groups:
        if group in DEMOGRAPHIC_FEATURES:
            for feature_idx, base_value in DEMOGRAPHIC_FEATURES[group].items():
                # Apply intensity scaling
                target_value = base_value * intensity
                
                # Clamp the feature across all positions
                if hasattr(clamped_activations, 'shape'):  # PyTorch tensor
                    clamped_activations[:, :, feature_idx] = target_value
                else:  # NumPy array or similar
                    clamped_activations[:, :, feature_idx] = target_value
    
    return clamped_activations

def generate_experiment_commands(num_cases=100, device="cpu"):
    """Generate all command-line commands for the complete experimental matrix.

    Returns:
        Dictionary with experiment categories and their commands
    """
    commands = {
        "baseline": [],
        "prompt_only": [],
        "clamping_only": [],
        "both": [],
        "equivalence": []
    }
    
    all_demographics = get_all_demographic_combinations()
    intensities = [1.0, 5.0, 10.0]
    
    base_cmd = f"python3 src/advai/main.py --num-cases {num_cases} --device {device}"
    
    # 1. Baseline (1 condition)
    commands["baseline"].append(f"{base_cmd} --output-suffix baseline")
    
    # 2. Prompt-only conditions (17 conditions)
    for demo_groups in all_demographics:
        demo_key = "_".join(demo_groups)
        cmd = f"{base_cmd} --demographic-prompt {' '.join(demo_groups)} --output-suffix prompt_only_{demo_key}"
        commands["prompt_only"].append(cmd)
    
    # 3. Clamping-only conditions (17 × 3 = 51 conditions)
    for demo_groups in all_demographics:
        for intensity in intensities:
            demo_key = "_".join(demo_groups)
            # Convert demographic groups to clamping features
            clamp_groups = []
            for group in demo_groups:
                if group in ["male", "female"]:
                    clamp_groups.append(group)
                elif group in ["pediatric", "adolescent", "young_adult", "middle_age", "senior"]:
                    clamp_groups.append(group)
            
            cmd = f"{base_cmd} --clamp-features {' '.join(clamp_groups)} --clamp-intensity {intensity} --output-suffix clamp_only_{demo_key}_{intensity}x"
            commands["clamping_only"].append(cmd)
    
    # 4. Both conditions (17 × 3 = 51 conditions)
    for demo_groups in all_demographics:
        for intensity in intensities:
            demo_key = "_".join(demo_groups)
            clamp_groups = []
            for group in demo_groups:
                if group in ["male", "female"]:
                    clamp_groups.append(group)
                elif group in ["pediatric", "adolescent", "young_adult", "middle_age", "senior"]:
                    clamp_groups.append(group)
            
            cmd = f"{base_cmd} --demographic-prompt {' '.join(demo_groups)} --clamp-features {' '.join(clamp_groups)} --clamp-intensity {intensity} --output-suffix both_{demo_key}_{intensity}x"
            commands["both"].append(cmd)
    
    # 5. Equivalence validation (17 × 2 = 34 conditions)
    for demo_groups in all_demographics:
        demo_key = "_".join(demo_groups)
        
        # Method A: Demographic prompt, no clamping
        cmd_a = f"{base_cmd} --demographic-prompt {' '.join(demo_groups)} --output-suffix equiv_prompt_{demo_key}"
        commands["equivalence"].append(cmd_a)
        
        # Method B: Neutral prompt + 1× clamping
        clamp_groups = []
        for group in demo_groups:
            if group in ["male", "female"]:
                clamp_groups.append(group)
            elif group in ["pediatric", "adolescent", "young_adult", "middle_age", "senior"]:
                clamp_groups.append(group)
        
        cmd_b = f"{base_cmd} --clamp-features {' '.join(clamp_groups)} --clamp-intensity 1.0 --output-suffix equiv_clamp_{demo_key}"
        commands["equivalence"].append(cmd_b)
    
    return commands

def save_experiment_plan(commands, output_dir="experiment_plans"):
    """Save the complete experiment plan to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save commands as JSON
    with open(f"{output_dir}/experiment_commands_{timestamp}.json", "w") as f:
        json.dump(commands, f, indent=2)
    
    # Save as executable shell scripts
    for category, cmd_list in commands.items():
        script_path = f"{output_dir}/run_{category}_{timestamp}.sh"
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"# {category.upper()} EXPERIMENTS\n")
            f.write(f"# Generated on {datetime.now()}\n\n")
            
            for i, cmd in enumerate(cmd_list, 1):
                f.write(f"echo \"Running {category} experiment {i}/{len(cmd_list)}\"\n")
                f.write(f"{cmd}\n\n")
        
        # Make executable
        os.chmod(script_path, 0o755)
        print(f" Created executable script: {script_path}")
    
    # Create master script
    master_script = f"{output_dir}/run_all_experiments_{timestamp}.sh"
    with open(master_script, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# COMPLETE BIAS EXPERIMENT SUITE\n")
        f.write(f"# Generated on {datetime.now()}\n")
        f.write("# Total conditions: 137\n\n")
        
        total_conditions = sum(len(cmd_list) for cmd_list in commands.values())
        f.write(f"echo \"Starting complete bias experiment suite ({total_conditions} conditions)\"\n\n")
        
        for category in ["baseline", "prompt_only", "clamping_only", "both", "equivalence"]:
            if category in commands:
                script_name = f"run_{category}_{timestamp}.sh"
                f.write(f"echo \"=== {category.upper()} EXPERIMENTS ===\"\n")
                f.write(f"./{script_name}\n\n")
    
    os.chmod(master_script, 0o755)
    print(f" Created master script: {master_script}")
    
    return output_dir

def print_experiment_summary(commands):
    """Print a summary of the experimental design."""
    
    print("\n" + "=" * 80)
    print(" FINAL BIAS EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print("\n EXPERIMENTAL MATRIX:")
    for category, cmd_list in commands.items():
        print(f"  {category.upper():15}: {len(cmd_list):3} conditions")
    
    total_conditions = sum(len(cmd_list) for cmd_list in commands.values())
    print(f"  {'TOTAL':15}: {total_conditions:3} conditions")
    
    print("\n DEMOGRAPHIC GROUPS (17 total):")
    all_demographics = get_all_demographic_combinations()
    
    print("  Age-only (5):")
    age_only = [d for d in all_demographics if len(d) == 1 and d[0] in ["pediatric", "adolescent", "young_adult", "middle_age", "senior"]]
    for demo in age_only:
        print(f"    - {demo[0]}")
    
    print("  Gender-only (2):")
    gender_only = [d for d in all_demographics if len(d) == 1 and d[0] in ["male", "female"]]
    for demo in gender_only:
        print(f"    - {demo[0]}")
    
    print("  Combined (10):")
    combined = [d for d in all_demographics if len(d) == 2]
    for demo in combined:
        print(f"    - {' + '.join(demo)}")
    
    print("\n INTENSITIES: 1×, 5×, 10×")
    print("\n VALIDATION: Equivalence testing (prompt vs 1× clamping)")
    
    print("\n" + "=" * 80)

def main():
    """Main function to generate the complete experimental framework."""
    
    parser = argparse.ArgumentParser(description="Generate final bias experiment commands")
    parser.add_argument("--num-cases", type=int, default=100, help="Number of test cases per experiment")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run experiments on")
    parser.add_argument("--output-dir", type=str, default="experiment_plans", help="Output directory for experiment plans")
    parser.add_argument("--execute", action="store_true", help="Execute experiments immediately (use with caution)")
    
    args = parser.parse_args()
    
    print(" GENERATING FINAL BIAS EXPERIMENT FRAMEWORK")
    print(f" Cases per experiment: {args.num_cases}")
    print(f" Device: {args.device}")
    
    # Generate all experiment commands
    commands = generate_experiment_commands(args.num_cases, args.device)
    
    # Print summary
    print_experiment_summary(commands)
    
    # Save experiment plans
    output_dir = save_experiment_plan(commands, args.output_dir)
    
    print(f"\n Experiment plans saved to: {output_dir}")
    print(f"\n Ready to run {sum(len(cmd_list) for cmd_list in commands.values())} experimental conditions!")
    
    if args.execute:
        print("\n EXECUTING EXPERIMENTS - This will take a very long time!")
        # Implementation for execution would go here
        print(" Execution not implemented yet - use generated scripts instead")
    else:
        print("\n To run experiments, execute the generated shell scripts in the output directory.")

if __name__ == "__main__":
    main()
