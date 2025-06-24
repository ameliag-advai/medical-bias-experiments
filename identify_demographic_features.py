"""
Identify demographic features in SAE activations using statistical analysis.
Based on interpretability best practices from Anthropic and other research.
"""
#Run # For sex features
#HF_TOKEN=... PYTHONPATH=. python3 identify_demographic_features.py --demographic sex --num-cases 100 --model gemma --device cpu --output sex_features_analysis.csv

# For age features
#HF_TOKEN=... PYTHONPATH=. python3 identify_demographic_features.py --demographic age --num-cases 100 --model gemma --device cpu --output age_features_analysis.csv

# number of cases defaults to 50 if not specified 

import torch
import numpy as np
from scipy import stats
import pandas as pd
import os
import json
from tqdm import tqdm
import argparse

from src.advai.models.loader import load_model_and_sae
from src.advai.data.io import load_patient_data, extract_cases_from_dataframe
from src.advai.data.prompt_builder import PromptBuilder, get_subsets
from src.advai.analysis.analyse import run_prompt


def get_activations_for_prompt(prompt, model, sae):
    """Get SAE activations for a given prompt."""
    with torch.no_grad():
        tokenised_prompt = model.to_tokens(prompt)
        model_activations = model.run_with_cache(tokenised_prompt, return_type=None)[1][sae.cfg.hook_name]
        vectorised = model_activations[0, -1, :].unsqueeze(0)
        sae_activations = sae(vectorised)[0]
    return sae_activations


def analyze_demographic_features(
    model, 
    sae, 
    cases, 
    demographic_concept,  # 'sex' or 'age'
    num_cases=50,
    threshold=1.0,
    top_k=20
):
    """
    Analyze which SAE features are most associated with a demographic concept.
    
    Returns:
        dict: Statistics for each feature including mean activations, p-values, etc.
    """
    # Initialize prompt builder
    conditions_mapping = json.load(open("release_conditions.json"))
    evidences = json.load(open("release_evidences.json"))
    
    # Templates for with/without demographics
    if demographic_concept == "sex":
        full_template = "A {{ sex|clean }} patient has symptoms: {{ symptoms }}."
        baseline_template = "A patient has symptoms: {{ symptoms }}."
    elif demographic_concept == "age":
        full_template = "A {{ age|clean }}-year-old patient has symptoms: {{ symptoms }}."
        baseline_template = "A patient has symptoms: {{ symptoms }}."
    else:
        raise ValueError(f"Unknown demographic concept: {demographic_concept}")
    
    prompt_builder = PromptBuilder(
        conditions_mapping,
        demographic_concepts=[demographic_concept],
        evidences=evidences,
        concepts_to_test=[demographic_concept],
        full_prompt_template=full_template,
        baseline_prompt_template=baseline_template
    )
    
    # Collect activations
    activations_with_demo = []
    activations_without_demo = []
    demo_values = []
    
    print(f"Analyzing {demographic_concept} features...")
    
    # Process cases
    for case in tqdm(cases[:num_cases], desc="Processing cases"):
        # Skip if demographic is missing
        if case.get(demographic_concept) is None:
            continue
            
        # Get prompts with and without demographic
        prompt_with, _ = prompt_builder.build_prompts(case, 0, (demographic_concept,))
        prompt_without, _ = prompt_builder.build_prompts(case, 0, ())
        
        # Get activations
        act_with = get_activations_for_prompt(prompt_with, model, sae)
        act_without = get_activations_for_prompt(prompt_without, model, sae)
        
        activations_with_demo.append(act_with.cpu().numpy())
        activations_without_demo.append(act_without.cpu().numpy())
        demo_values.append(case[demographic_concept])
    
    # Convert to arrays
    activations_with_demo = np.array(activations_with_demo)
    activations_without_demo = np.array(activations_without_demo)
    
    # Calculate statistics for each feature
    num_features = activations_with_demo.shape[1]
    feature_stats = {}
    
    for feature_idx in range(num_features):
        with_acts = activations_with_demo[:, feature_idx]
        without_acts = activations_without_demo[:, feature_idx]
        
        # Calculate mean difference
        mean_diff = np.mean(with_acts - without_acts)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(with_acts, without_acts)
        
        # Effect size (Cohen's d)
        diff = with_acts - without_acts
        cohen_d = np.mean(diff) / (np.std(diff) + 1e-8)
        
        # Activation rate (how often > threshold)
        activation_rate_with = np.mean(with_acts > threshold)
        activation_rate_without = np.mean(without_acts > threshold)
        
        feature_stats[feature_idx] = {
            'mean_diff': mean_diff,
            'p_value': p_value,
            'cohen_d': cohen_d,
            'activation_rate_with': activation_rate_with,
            'activation_rate_without': activation_rate_without,
            'mean_with': np.mean(with_acts),
            'mean_without': np.mean(without_acts),
            't_statistic': t_stat
        }
    
    # For sex/age specific analysis
    if demographic_concept == "sex":
        # Separate by male/female
        male_indices = [i for i, v in enumerate(demo_values) if v == "M"]
        female_indices = [i for i, v in enumerate(demo_values) if v == "F"]
        
        for feature_idx in range(num_features):
            male_acts = activations_with_demo[male_indices, feature_idx]
            female_acts = activations_with_demo[female_indices, feature_idx]
            
            if len(male_acts) > 0 and len(female_acts) > 0:
                # Compare male vs female activations
                t_stat_mf, p_value_mf = stats.ttest_ind(male_acts, female_acts)
                feature_stats[feature_idx]['male_mean'] = np.mean(male_acts)
                feature_stats[feature_idx]['female_mean'] = np.mean(female_acts)
                feature_stats[feature_idx]['male_vs_female_p'] = p_value_mf
    
    elif demographic_concept == "age":
        # Separate by young/old (using median split)
        ages = [int(v) for v in demo_values if v is not None]
        median_age = np.median(ages)
        young_indices = [i for i, v in enumerate(demo_values) if v is not None and int(v) < median_age]
        old_indices = [i for i, v in enumerate(demo_values) if v is not None and int(v) >= median_age]
        
        for feature_idx in range(num_features):
            young_acts = activations_with_demo[young_indices, feature_idx]
            old_acts = activations_with_demo[old_indices, feature_idx]
            
            if len(young_acts) > 0 and len(old_acts) > 0:
                # Compare young vs old activations
                t_stat_yo, p_value_yo = stats.ttest_ind(young_acts, old_acts)
                feature_stats[feature_idx]['young_mean'] = np.mean(young_acts)
                feature_stats[feature_idx]['old_mean'] = np.mean(old_acts)
                feature_stats[feature_idx]['young_vs_old_p'] = p_value_yo
                feature_stats[feature_idx]['median_age'] = median_age
    
    return feature_stats, demo_values


def identify_top_features(feature_stats, demographic_concept, p_threshold=0.01, min_effect_size=0.5):
    """Identify the top features for a demographic concept."""
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(feature_stats).T
    df.index.name = 'feature_idx'
    df = df.reset_index()
    
    # Filter by statistical significance and effect size
    significant = df[(df['p_value'] < p_threshold) & (abs(df['cohen_d']) > min_effect_size)]
    
    if demographic_concept == "sex":
        # For sex, also look at male vs female differences
        if 'male_vs_female_p' in df.columns:
            # Male features: higher activation for males
            male_features = significant[
                (significant['mean_diff'] > 0) & 
                (significant.get('male_mean', 0) > significant.get('female_mean', 0))
            ].nlargest(10, 'cohen_d')
            
            # Female features: higher activation for females  
            female_features = significant[
                (significant['mean_diff'] > 0) & 
                (significant.get('female_mean', 0) > significant.get('male_mean', 0))
            ].nlargest(10, 'cohen_d')
            
            return {
                'male': male_features['feature_idx'].tolist(),
                'female': female_features['feature_idx'].tolist(),
                'male_stats': male_features,
                'female_stats': female_features
            }
    
    elif demographic_concept == "age":
        # For age, look at young vs old differences
        if 'young_vs_old_p' in df.columns:
            # Old features: higher activation for older patients
            old_features = significant[
                (significant['mean_diff'] > 0) & 
                (significant.get('old_mean', 0) > significant.get('young_mean', 0))
            ].nlargest(10, 'cohen_d')
            
            # Young features: higher activation for younger patients
            young_features = significant[
                (significant['mean_diff'] > 0) & 
                (significant.get('young_mean', 0) > significant.get('old_mean', 0))
            ].nlargest(10, 'cohen_d')
            
            return {
                'old': old_features['feature_idx'].tolist(),
                'young': young_features['feature_idx'].tolist(),
                'old_stats': old_features,
                'young_stats': young_features,
                'median_age': df['median_age'].iloc[0] if 'median_age' in df.columns else None
            }
    
    # Fallback: just return top positive and negative features
    positive_features = significant[significant['mean_diff'] > 0].nlargest(10, 'cohen_d')
    negative_features = significant[significant['mean_diff'] < 0].nlargest(10, abs(significant['cohen_d']))
    
    return {
        'positive': positive_features['feature_idx'].tolist(),
        'negative': negative_features['feature_idx'].tolist(),
        'positive_stats': positive_features,
        'negative_stats': negative_features
    }


def main():
    parser = argparse.ArgumentParser(description="Identify demographic features in SAE")
    parser.add_argument('--demographic', type=str, choices=['sex', 'age'], required=True)
    parser.add_argument('--num-cases', type=int, default=50, help='Number of cases to analyze')
    parser.add_argument('--model', type=str, default='gemma', choices=['gemma', 'gpt2'])
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output', type=str, help='Output CSV file for results')
    args = parser.parse_args()
    
    # Load model and SAE
    print("Loading model and SAE...")
    model, sae = load_model_and_sae(model_scope=args.model, device=args.device)
    
    # Load cases
    print("Loading patient data...")
    df = load_patient_data("release_test_patients")
    cases = extract_cases_from_dataframe(df)
    
    # Analyze features
    feature_stats, demo_values = analyze_demographic_features(
        model, sae, cases, args.demographic, num_cases=args.num_cases
    )
    
    # Identify top features
    top_features = identify_top_features(feature_stats, args.demographic)
    
    # Print results
    print(f"\n=== Top {args.demographic.upper()} Features ===")
    
    if args.demographic == "sex":
        print(f"\nMALE features: {top_features['male']}")
        print(f"FEMALE features: {top_features['female']}")
        
        print("\nMALE feature statistics:")
        print(top_features['male_stats'][['feature_idx', 'cohen_d', 'p_value', 'male_mean', 'female_mean']])
        
        print("\nFEMALE feature statistics:")
        print(top_features['female_stats'][['feature_idx', 'cohen_d', 'p_value', 'male_mean', 'female_mean']])
        
    elif args.demographic == "age":
        print(f"\nMedian age for split: {top_features.get('median_age', 'N/A')}")
        print(f"\nOLD features: {top_features['old']}")
        print(f"YOUNG features: {top_features['young']}")
        
        print("\nOLD feature statistics:")
        print(top_features['old_stats'][['feature_idx', 'cohen_d', 'p_value', 'old_mean', 'young_mean']])
        
        print("\nYOUNG feature statistics:")
        print(top_features['young_stats'][['feature_idx', 'cohen_d', 'p_value', 'old_mean', 'young_mean']])
    
    # Save full results if requested
    if args.output:
        df = pd.DataFrame(feature_stats).T
        df.to_csv(args.output)
        print(f"\nFull results saved to {args.output}")
    
    # Compare with current features in clamping.py
    print("\n=== Comparison with Current Features ===")
    from src.advai.analysis.clamping import MALE_FEATURES, FEMALE_FEATURES, OLD_FEATURES, YOUNG_FEATURES
    
    if args.demographic == "sex":
        print(f"Current MALE features: {MALE_FEATURES}")
        print(f"Overlap with identified: {set(MALE_FEATURES) & set(top_features['male'])}")
        print(f"\nCurrent FEMALE features: {FEMALE_FEATURES}")
        print(f"Overlap with identified: {set(FEMALE_FEATURES) & set(top_features['female'])}")
    else:
        print(f"Current OLD features: {OLD_FEATURES}")
        print(f"Overlap with identified: {set(OLD_FEATURES) & set(top_features['old'])}")
        print(f"\nCurrent YOUNG features: {YOUNG_FEATURES}")
        print(f"Overlap with identified: {set(YOUNG_FEATURES) & set(top_features['young'])}")


if __name__ == "__main__":
    main()