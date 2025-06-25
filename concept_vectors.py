"""
Concept vector approach for demographic features.
Instead of statistical analysis, directly compute activation differences.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from src.advai.models.loader import load_model_and_sae
from src.advai.data.io import load_patient_data, extract_cases_from_dataframe


def get_activations_for_prompt(prompt: str, model, sae) -> torch.Tensor:
    """Get SAE activations for a given prompt."""
    with torch.no_grad():
        tokenised_prompt = model.to_tokens(prompt)
        model_activations = model.run_with_cache(tokenised_prompt, return_type=None)[1][sae.cfg.hook_name]
        vectorised = model_activations[0, -1, :].unsqueeze(0)
        sae_activations = sae.encode(vectorised)[0]
    return sae_activations


def compute_concept_vector(
    model, 
    sae,
    concept_prompts: List[str],
    baseline_prompts: List[str],
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute a concept vector by averaging activation differences.
    
    Args:
        concept_prompts: Prompts containing the concept (e.g., "A male patient...")
        baseline_prompts: Baseline prompts (e.g., "A patient...")
        normalize: Whether to normalize the resulting vector
    
    Returns:
        Concept vector (difference in activation space)
    """
    concept_acts = []
    baseline_acts = []
    
    print(f"Computing concept vector from {len(concept_prompts)} prompt pairs...")
    
    for concept_prompt, baseline_prompt in zip(concept_prompts, baseline_prompts):
        concept_act = get_activations_for_prompt(concept_prompt, model, sae)
        baseline_act = get_activations_for_prompt(baseline_prompt, model, sae)
        
        concept_acts.append(concept_act.cpu())
        baseline_acts.append(baseline_act.cpu())
    
    # Average the activations
    concept_mean = torch.stack(concept_acts).mean(dim=0)
    baseline_mean = torch.stack(baseline_acts).mean(dim=0)
    
    # Compute the difference vector
    concept_vector = concept_mean - baseline_mean
    
    if normalize:
        concept_vector = concept_vector / (concept_vector.norm() + 1e-8)
    
    return concept_vector


def generate_demographic_prompts(cases: List[Dict], num_cases: int = 50) -> Dict[str, List[Tuple[str, str]]]:
    """
    Generate prompt pairs for demographic concepts.
    
    Returns:
        Dictionary with concept names and (concept_prompt, baseline_prompt) pairs
    """
    prompt_pairs = {
        'male': [],
        'female': [],
        'young': [],
        'old': []
    }
    
    for case in cases[:num_cases]:
        symptoms = case.get('symptoms', 'chest pain')
        
        # Sex-based prompts
        if case.get('sex') == 'M':
            male_prompt = f"A male patient has symptoms: {symptoms}."
            baseline_prompt = f"A patient has symptoms: {symptoms}."
            prompt_pairs['male'].append((male_prompt, baseline_prompt))
            
        elif case.get('sex') == 'F':
            female_prompt = f"A female patient has symptoms: {symptoms}."
            baseline_prompt = f"A patient has symptoms: {symptoms}."
            prompt_pairs['female'].append((female_prompt, baseline_prompt))
        
        # Age-based prompts
        if case.get('age'):
            age = int(case['age'])
            age_prompt = f"A {age}-year-old patient has symptoms: {symptoms}."
            baseline_prompt = f"A patient has symptoms: {symptoms}."
            
            if age < 40:  # Young
                prompt_pairs['young'].append((age_prompt, baseline_prompt))
            elif age > 65:  # Old
                prompt_pairs['old'].append((age_prompt, baseline_prompt))
    
    return prompt_pairs


def compute_all_demographic_vectors(
    model, 
    sae, 
    cases: List[Dict], 
    num_cases: int = 50,
    top_k: int = 20
) -> Dict[str, Dict]:
    """
    Compute concept vectors for all demographic concepts.
    
    Returns:
        Dictionary with concept vectors and top features for each demographic
    """
    prompt_pairs = generate_demographic_prompts(cases, num_cases)
    results = {}
    
    for concept, pairs in prompt_pairs.items():
        if len(pairs) < 5:  # Need minimum samples
            print(f"Skipping {concept}: only {len(pairs)} samples")
            continue
            
        print(f"\nComputing {concept} concept vector from {len(pairs)} samples...")
        
        concept_prompts = [pair[0] for pair in pairs]
        baseline_prompts = [pair[1] for pair in pairs]
        
        # Compute concept vector
        concept_vector = compute_concept_vector(
            model, sae, concept_prompts, baseline_prompts, normalize=False
        )
        
        # Find top features
        abs_values = torch.abs(concept_vector)
        top_indices = torch.topk(abs_values, top_k).indices
        
        top_features = {}
        for idx in top_indices:
            feature_idx = idx.item()
            value = concept_vector[feature_idx].item()
            top_features[feature_idx] = value
        
        results[concept] = {
            'concept_vector': concept_vector,
            'top_features': top_features,
            'num_samples': len(pairs)
        }
        
        print(f"Top {concept} features:")
        for feat_idx, value in list(top_features.items())[:5]:
            print(f"  Feature {feat_idx}: {value:+.4f}")
    
    return results


def apply_concept_vector_clamping(
    activations: torch.Tensor,
    concept_vector: torch.Tensor,
    extent: float = 1.0
) -> torch.Tensor:
    """
    Apply concept vector clamping by adding the scaled concept vector.
    
    Args:
        activations: Original SAE activations [batch, features]
        concept_vector: Concept vector to add [features]
        extent: Scaling factor for the concept vector
    
    Returns:
        Clamped activations
    """
    if not activations.shape[-1] == concept_vector.shape[0]:
        raise ValueError(f"Shape mismatch: {activations.shape} vs {concept_vector.shape}")
    
    # Add the scaled concept vector
    clamped = activations + (extent * concept_vector.unsqueeze(0))
    return clamped


def demonstrate_concept_vectors():
    """
    Demonstrate the concept vector approach with examples.
    """
    print("=== CONCEPT VECTOR APPROACH ===\n")
    
    print("🎯 Concept: Instead of statistical analysis, compute direct differences:")
    print("   male_vector = mean(male_patient_activations - patient_activations)")
    print("   female_vector = mean(female_patient_activations - patient_activations)\n")
    
    print("✅ Advantages:")
    print("   • Direct semantic direction in activation space")
    print("   • No confounding with 'personness' or prompt structure")
    print("   • More interpretable as 'adding maleness' vs statistical correlation")
    print("   • Can work with any concept pair (king-monarch, doctor-person, etc.)\n")
    
    print("🔧 Usage:")
    print("   1. Compute concept vectors from prompt pairs")
    print("   2. Apply by adding: new_acts = old_acts + (extent × concept_vector)")
    print("   3. Much cleaner than statistical feature identification\n")
    
    print("📊 Example concept vectors:")
    print("   • male_vector[1476] = +0.006 (males activate this feature more)")
    print("   • male_vector[1997] = +0.022 (males activate this feature much more)")
    print("   • To make text 'more male': add 3.0 × male_vector")


if __name__ == "__main__":
    demonstrate_concept_vectors()
    
    # Example usage (commented out - requires loading model/data):
    """
    from src.advai.data.io import load_patient_data, extract_cases_from_dataframe
    
    # Load data and model
    cases = extract_cases_from_dataframe(load_patient_data())
    model, sae = load_model_and_sae('gemma', device='cpu')
    
    # Compute all demographic concept vectors
    results = compute_all_demographic_vectors(model, sae, cases, num_cases=50)
    
    # Use for clamping
    male_vector = results['male']['concept_vector']
    clamped_acts = apply_concept_vector_clamping(original_acts, male_vector, extent=3.0)
    """
