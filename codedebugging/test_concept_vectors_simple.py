#!/usr/bin/env python3
"""
Simple test of the integrated concept vectors with mock data to verify the functions work.
"""

import json
import torch
from concept_vectors import (
    process_symptoms_from_features, 
    generate_demographic_prompts,
    compute_concept_vector,
    get_activations_for_prompt
)
from src.advai.models.loader import load_model_and_sae

def create_mock_data():
    """Create mock patient cases and evidences for testing."""
    
    # Mock evidences (simplified from release_evidences.json)
    evidences = {
        "chest_pain": {"question_en": "Do you have chest pain?"},
        "shortness_breath": {"question_en": "Are you experiencing shortness of breath?"},
        "headache": {"question_en": "Do you have a headache?"},
        "fever": {"question_en": "Do you have fever?"},
        "nausea": {"question_en": "Do you feel nauseous?"},
        "fatigue": {"question_en": "Are you feeling tired?"}
    }
    
    # Mock patient cases
    cases = [
        {"sex": "M", "age": 25, "features": "['chest_pain', 'shortness_breath']"},
        {"sex": "F", "age": 30, "features": "['headache', 'nausea']"},
        {"sex": "M", "age": 70, "features": "['fever', 'fatigue']"},
        {"sex": "F", "age": 35, "features": "['chest_pain', 'headache']"},
        {"sex": "M", "age": 28, "features": "['shortness_breath', 'fever']"},
        {"sex": "F", "age": 65, "features": "['nausea', 'fatigue']"},
        {"sex": "M", "age": 22, "features": "['headache', 'chest_pain']"},
        {"sex": "F", "age": 75, "features": "['fever', 'shortness_breath']"},
    ]
    
    return cases, evidences

def test_symptom_processing():
    """Test the symptom processing function."""
    print("🧪 Testing symptom processing...")
    
    cases, evidences = create_mock_data()
    
    for i, case in enumerate(cases[:3]):
        features = case['features']
        symptoms = process_symptoms_from_features(features, evidences)
        print(f"  Case {i+1}: {features} -> '{symptoms}'")

def test_prompt_generation():
    """Test the prompt generation function."""
    print("\n🧪 Testing prompt generation...")
    
    cases, evidences = create_mock_data()
    
    prompt_pairs = generate_demographic_prompts(cases, evidences, num_cases=8)
    
    print(f"\nGenerated prompt pairs:")
    for concept, pairs in prompt_pairs.items():
        print(f"  {concept}: {len(pairs)} pairs")
        if pairs:
            print(f"    Example: {pairs[0][0]}")

def test_with_model():
    """Test with actual model if available."""
    print("\n🧪 Testing with model...")
    
    try:
        # Load model and SAE
        print("  Loading model...")
        model, sae = load_model_and_sae('gemma', device='cpu')
        
        # Test activation computation
        test_prompt = "A male patient has symptoms: chest pain, shortness of breath."
        activations = get_activations_for_prompt(test_prompt, model, sae)
        print(f"  Activations shape: {activations.shape}")
        print(f"  Non-zero activations: {(activations != 0).sum().item()}")
        
        # Test concept vector computation
        cases, evidences = create_mock_data()
        prompt_pairs = generate_demographic_prompts(cases, evidences, num_cases=8)
        
        if len(prompt_pairs['male']) >= 2:
            concept_prompts = [p[0] for p in prompt_pairs['male']]
            baseline_prompts = [p[1] for p in prompt_pairs['male']]
            
            print(f"  Computing concept vector from {len(concept_prompts)} prompts...")
            concept_vector = compute_concept_vector(
                model, sae, concept_prompts, baseline_prompts, normalize=True
            )
            print(f"  Concept vector shape: {concept_vector.shape}")
            print(f"  Concept vector norm: {concept_vector.norm().item():.4f}")
            
            # Show top features
            top_indices = torch.topk(torch.abs(concept_vector), k=5).indices
            print(f"  Top 5 features:")
            for i, idx in enumerate(top_indices):
                print(f"    Feature {idx.item()}: {concept_vector[idx].item():+.4f}")
        
        print("✅ Model test completed successfully!")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")

def main():
    print("🚀 Testing integrated concept vectors (simple version)...")
    
    test_symptom_processing()
    test_prompt_generation()
    test_with_model()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
