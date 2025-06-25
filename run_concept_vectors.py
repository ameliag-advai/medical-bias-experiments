"""
Run concept vector analysis for demographic features.
"""

import torch
import numpy as np
import json
from concept_vectors import compute_all_demographic_vectors, demonstrate_concept_vectors
from src.advai.models.loader import load_model_and_sae
from src.advai.data.io import load_patient_data, extract_cases_from_dataframe


def main():
    print("=== CONCEPT VECTOR DEMOGRAPHIC ANALYSIS ===\n")
    
    # Load model and data
    print("Loading model and SAE...")
    model, sae = load_model_and_sae('gemma', device='cpu')
    
    print("Loading patient data...")
    df = load_patient_data("release_test_patients")
    cases = extract_cases_from_dataframe(df)
    print(f"Loaded {len(cases)} cases\n")
    
    # Compute concept vectors
    print("Computing demographic concept vectors...")
    results = compute_all_demographic_vectors(
        model, sae, cases, 
        num_cases=500,  # Start with 50 for speed
        top_k=20
    )
    
    # Save results
    print("\n=== SAVING RESULTS ===")
    
    # Save concept vectors and top features
    output_data = {}
    for concept, data in results.items():
        # Convert tensors to lists for JSON serialization
        concept_vector = data['concept_vector'].tolist()
        top_features = data['top_features']
        
        output_data[concept] = {
            'concept_vector': concept_vector,
            'top_features': top_features,
            'num_samples': data['num_samples']
        }
        
        print(f"Saved {concept} concept vector ({data['num_samples']} samples)")
    
    # Save to JSON
    with open('demographic_concept_vectors.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to demographic_concept_vectors.json")
    
    # Print summary
    print("\n=== CONCEPT VECTOR SUMMARY ===")
    for concept, data in results.items():
        print(f"\n{concept.upper()} concept vector:")
        print(f"  Samples: {data['num_samples']}")
        print(f"  Top 5 features:")
        
        top_features = data['top_features']
        for i, (feat_idx, value) in enumerate(list(top_features.items())[:5]):
            direction = "↑" if value > 0 else "↓"
            print(f"    {i+1}. Feature {feat_idx}: {value:+.4f} {direction}")
    
    # Demonstrate usage
    print("\n=== USAGE EXAMPLE ===")
    if 'male' in results:
        male_vector = results['male']['concept_vector']
        print(f"To make activations 'more male':")
        print(f"  new_acts = old_acts + (3.0 * male_concept_vector)")
        print(f"  Vector shape: {male_vector.shape}")
        print(f"  Vector norm: {male_vector.norm():.4f}")


if __name__ == "__main__":
    main()
