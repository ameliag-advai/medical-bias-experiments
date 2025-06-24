"""Quick validation of demographic features"""
# Interactive Validation (based on Anthropic's approach)

import torch
from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.clamping import MALE_FEATURES, FEMALE_FEATURES, OLD_FEATURES, YOUNG_FEATURES

def test_feature_on_prompt(model, sae, prompt, feature_indices):
    """Test which features activate for a given prompt."""
    with torch.no_grad():
        tokenised_prompt = model.to_tokens(prompt)
        model_activations = model.run_with_cache(tokenised_prompt, return_type=None)[1][sae.cfg.hook_name]
        vectorised = model_activations[0, -1, :].unsqueeze(0)
        sae_activations = sae(vectorised)[0]
        
    print(f"\nPrompt: {prompt}")
    print(f"Feature activations (threshold > 1.0):")
    for idx in feature_indices:
        activation = sae_activations[idx].item()
        if activation > 1.0:
            print(f"  Feature {idx}: {activation:.2f} ✓")
        else:
            print(f"  Feature {idx}: {activation:.2f}")
    
    # Also show any other highly active features
    all_active = (sae_activations > 1.0).nonzero().squeeze().tolist()
    if isinstance(all_active, int):
        all_active = [all_active]
    other_active = [f for f in all_active if f not in feature_indices]
    if other_active:
        print(f"\nOther active features: {other_active[:10]}")
    
    return sae_activations

# Load model
print("Loading model...")
model, sae = load_model_and_sae(model_scope="gemma", device="cpu")

# Test prompts
test_prompts = {
    "male": [
        "A male patient has symptoms: [fever, cough].",
        "A man has symptoms: [headache, fatigue].",
        "A gentleman presenting with symptoms: [chest pain].",
    ],
    "female": [
        "A female patient has symptoms: [fever, cough].",
        "A woman has symptoms: [headache, fatigue].", 
        "A lady presenting with symptoms: [chest pain].",
    ],
    "old": [
        "A 75-year-old patient has symptoms: [fever, cough].",
        "An elderly patient has symptoms: [headache, fatigue].",
        "An 80-year-old presenting with symptoms: [chest pain].",
    ],
    "young": [
        "A 25-year-old patient has symptoms: [fever, cough].",
        "A young patient has symptoms: [headache, fatigue].",
        "A 20-year-old presenting with symptoms: [chest pain].",
    ],
    "neutral": [
        "A patient has symptoms: [fever, cough].",
        "Patient presenting with symptoms: [headache, fatigue].",
        "Individual with symptoms: [chest pain].",
    ]
}

print("\n=== TESTING MALE FEATURES ===")
for prompt in test_prompts["male"]:
    test_feature_on_prompt(model, sae, prompt, MALE_FEATURES)

print("\n=== TESTING FEMALE FEATURES ===")
for prompt in test_prompts["female"]:
    test_feature_on_prompt(model, sae, prompt, FEMALE_FEATURES)

print("\n=== TESTING OLD FEATURES ===")
for prompt in test_prompts["old"]:
    test_feature_on_prompt(model, sae, prompt, OLD_FEATURES)

print("\n=== TESTING YOUNG FEATURES ===")
for prompt in test_prompts["young"]:
    test_feature_on_prompt(model, sae, prompt, YOUNG_FEATURES)

print("\n=== TESTING NEUTRAL (should have low activations) ===")
all_demographic_features = set(MALE_FEATURES + FEMALE_FEATURES + OLD_FEATURES + YOUNG_FEATURES)
for prompt in test_prompts["neutral"]:
    test_feature_on_prompt(model, sae, prompt, list(all_demographic_features)[:10])