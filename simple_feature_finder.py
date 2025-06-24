"""Simple automated ersion of original method to find demographic features"""

import torch
import numpy as np
from collections import defaultdict
from src.advai.models.loader import load_model_and_sae

def get_active_features(model, sae, prompt, threshold=1.0):
    """Get features that activate above threshold for a prompt."""
    with torch.no_grad():
        tokenised_prompt = model.to_tokens(prompt)
        model_activations = model.run_with_cache(tokenised_prompt, return_type=None)[1][sae.cfg.hook_name]
        vectorised = model_activations[0, -1, :].unsqueeze(0)
        sae_activations = sae(vectorised)[0]
    
    active_indices = (sae_activations > threshold).nonzero().squeeze().tolist()
    if isinstance(active_indices, int):
        active_indices = [active_indices]
    
    return active_indices, sae_activations

# Load model
print("Loading model...")
model, sae = load_model_and_sae(model_scope="gemma", device="cpu")

# Define test prompts
male_prompts = [
    "A male patient has symptoms: [fever, headache, cough].",
    "A man presenting with symptoms: [chest pain, fatigue].",
    "The gentleman has symptoms: [nausea, dizziness].",
    "A male 40-year-old has symptoms: [back pain].",
    "He has symptoms: [shortness of breath].",
]

female_prompts = [
    "A female patient has symptoms: [fever, headache, cough].",
    "A woman presenting with symptoms: [chest pain, fatigue].",
    "The lady has symptoms: [nausea, dizziness].",
    "A female 40-year-old has symptoms: [back pain].",
    "She has symptoms: [shortness of breath].",
]

neutral_prompts = [
    "A patient has symptoms: [fever, headache, cough].",
    "Patient presenting with symptoms: [chest pain, fatigue].",
    "The patient has symptoms: [nausea, dizziness].",
    "A 40-year-old has symptoms: [back pain].",
    "Patient has symptoms: [shortness of breath].",
]

# Collect activations
male_features = defaultdict(int)
female_features = defaultdict(int)
neutral_features = defaultdict(int)

print("\nAnalyzing male prompts...")
for prompt in male_prompts:
    active, _ = get_active_features(model, sae, prompt)
    for feat in active:
        male_features[feat] += 1

print("Analyzing female prompts...")
for prompt in female_prompts:
    active, _ = get_active_features(model, sae, prompt)
    for feat in active:
        female_features[feat] += 1

print("Analyzing neutral prompts...")
for prompt in neutral_prompts:
    active, _ = get_active_features(model, sae, prompt)
    for feat in active:
        neutral_features[feat] += 1

# Find selective features
print("\n=== MALE-SELECTIVE FEATURES ===")
male_selective = []
for feat, count in male_features.items():
    if count >= 4 and neutral_features[feat] <= 1 and female_features[feat] <= 1:
        male_selective.append(feat)
        print(f"Feature {feat}: active in {count}/5 male, {female_features[feat]}/5 female, {neutral_features[feat]}/5 neutral")

print(f"\nTop male features: {male_selective[:10]}")

print("\n=== FEMALE-SELECTIVE FEATURES ===")
female_selective = []
for feat, count in female_features.items():
    if count >= 4 and neutral_features[feat] <= 1 and male_features[feat] <= 1:
        female_selective.append(feat)
        print(f"Feature {feat}: active in {count}/5 female, {male_features[feat]}/5 male, {neutral_features[feat]}/5 neutral")

print(f"\nTop female features: {female_selective[:10]}")

# Do the same for age
print("\n\n=== AGE ANALYSIS ===")

old_prompts = [
    "A 75-year-old patient has symptoms: [fever, headache].",
    "An 80-year-old has symptoms: [chest pain].",
    "An elderly patient has symptoms: [fatigue].",
    "A 70-year-old presenting with symptoms: [shortness of breath].",
    "An 85-year-old with symptoms: [confusion].",
]

young_prompts = [
    "A 25-year-old patient has symptoms: [fever, headache].",
    "A 20-year-old has symptoms: [chest pain].",
    "A young patient has symptoms: [fatigue].",
    "A 30-year-old presenting with symptoms: [shortness of breath].",
    "A 18-year-old with symptoms: [anxiety].",
]

old_features = defaultdict(int)
young_features = defaultdict(int)

print("\nAnalyzing old age prompts...")
for prompt in old_prompts:
    active, _ = get_active_features(model, sae, prompt)
    for feat in active:
        old_features[feat] += 1

print("Analyzing young age prompts...")
for prompt in young_prompts:
    active, _ = get_active_features(model, sae, prompt)
    for feat in active:
        young_features[feat] += 1

print("\n=== OLD-SELECTIVE FEATURES ===")
old_selective = []
for feat, count in old_features.items():
    if count >= 4 and young_features[feat] <= 1 and neutral_features[feat] <= 1:
        old_selective.append(feat)
        print(f"Feature {feat}: active in {count}/5 old, {young_features[feat]}/5 young, {neutral_features[feat]}/5 neutral")

print(f"\nTop old features: {old_selective[:10]}")

print("\n=== YOUNG-SELECTIVE FEATURES ===")
young_selective = []
for feat, count in young_features.items():
    if count >= 4 and old_features[feat] <= 1 and neutral_features[feat] <= 1:
        young_selective.append(feat)
        print(f"Feature {feat}: active in {count}/5 young, {old_features[feat]}/5 old, {neutral_features[feat]}/5 neutral")

print(f"\nTop young features: {young_selective[:10]}")

# Compare with current features
from src.advai.analysis.clamping import MALE_FEATURES, FEMALE_FEATURES, OLD_FEATURES, YOUNG_FEATURES

print("\n\n=== COMPARISON WITH CURRENT FEATURES ===")
print(f"Current MALE: {MALE_FEATURES}")
print(f"Found MALE: {male_selective[:5]}")
print(f"Current FEMALE: {FEMALE_FEATURES}")
print(f"Found FEMALE: {female_selective[:7]}")
print(f"Current OLD: {OLD_FEATURES}")
print(f"Found OLD: {old_selective[:7]}")
print(f"Current YOUNG: {YOUNG_FEATURES}")
print(f"Found YOUNG: {young_selective[:5]}")
