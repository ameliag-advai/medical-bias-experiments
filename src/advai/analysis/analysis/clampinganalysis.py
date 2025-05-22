"""
clampinganalysis.py

Compares diagnoses and SAE feature activations for 5 cases across 3 settings:
1. With demographic info
2. Without demographic info
3. Without demo info + input feature clamped 5x

Assumes you have run your analysis pipeline and have saved results and activations for each case.
Uses clamping.py for feature intervention.
"""
import torch
import numpy as np
import os
import json
from src.advai.analysis.clamping import clamp_sae_features
from src.advai.visuals.plots import visualize_clamping_analysis
from src.advai.data.io import load_conditions_mapping

# Directory where previous analysis results and activations are saved
RESULTS_DIR = 'activations'  # or your save dir

# Find all available case ids in the activations directory
def get_case_ids():
    files = os.listdir(RESULTS_DIR)
    case_ids = set()
    for fname in files:
        if fname.startswith('case_') and fname.endswith('_activations.pt'):
            # Extract the case id (handles e.g. case_0_activations.pt)
            num = fname[len('case_'):fname.rfind('_activations.pt')]
            if num.isdigit():
                case_ids.add(int(num))
            else:
                case_ids.add(num)
    # Sort numerically if possible
    try:
        case_ids = sorted(case_ids, key=int)
    except Exception:
        case_ids = sorted(case_ids)
    return case_ids

# Helper to load a single case's saved result
def load_case(case_id, group, demographic=None, extent=None, diagnosis_mapping=None):
    # group: 'with_demo', 'no_demo', or 'clamped'
    fname = os.path.join(RESULTS_DIR, f"case_{case_id}_activations.pt")
    d = torch.load(fname)
    # For 'clamped', clamp the SAE output with user-specified demographic and extent
    if group == 'clamped':
        if demographic is None or extent is None:
            raise ValueError('Demographic and extent must be specified for clamped group.')
        sae_out = clamp_sae_features(torch.unsqueeze(d['sae_out_without'], 0), demographic=demographic, extent=extent)
        top_dxs = torch.topk(sae_out[0], 5).indices.tolist()
        top_diagnoses = [diagnosis_mapping.get(str(idx), f"Diagnosis {idx}") for idx in top_dxs]
    elif group == 'with_demo':
        sae_out = torch.unsqueeze(d['sae_out_with'], 0)
        top_dxs = torch.topk(sae_out[0], 5).indices.tolist()
        top_diagnoses = [diagnosis_mapping.get(str(idx), f"Diagnosis {idx}") for idx in top_dxs]
    elif group == 'no_demo':
        sae_out = torch.unsqueeze(d['sae_out_without'], 0)
        top_dxs = torch.topk(sae_out[0], 5).indices.tolist()
        top_diagnoses = [diagnosis_mapping.get(str(idx), f"Diagnosis {idx}") for idx in top_dxs]
    else:
        raise ValueError('Unknown group')
    # Convert tensors to lists for JSON serialization
    return {
        'sae_out': sae_out.cpu().numpy().tolist() if hasattr(sae_out, 'cpu') else sae_out,
        'top_dxs': top_dxs,
        'top_diagnoses': top_diagnoses,
        'case_id': case_id
    }

# Load diagnosis mapping from release_conditions.json
RELEASE_CONDITIONS_PATH = os.path.join(os.path.dirname(__file__), '../../release_conditions.json')
if not os.path.exists(RELEASE_CONDITIONS_PATH):
    raise FileNotFoundError(f"release_conditions.json not found at {RELEASE_CONDITIONS_PATH}")
release_conditions = load_conditions_mapping(RELEASE_CONDITIONS_PATH)
# Map feature index (as string) to natural language diagnosis
# (Assume the mapping is {str(idx): 'diagnosis name'})
diagnosis_mapping = {str(idx): v['cond-name-eng'] if 'cond-name-eng' in v else k for idx, (k, v) in enumerate(release_conditions.items())}

# Prompt the user for demographic and extent to clamp for the 'clamped' group
print("Available demographics to clamp: male, female, old, young")
demographic = input("Enter demographic to clamp: ").strip().lower()
extent = float(input("Enter extent to clamp (e.g. 5 for 5x): ").strip())

# Compare diagnoses/features for all detected cases
case_ids = get_case_ids()
results = []
for case_id in case_ids:
    case_result = {'case_id': case_id}
    for group in ['with_demo', 'no_demo', 'clamped']:
        if group == 'clamped':
            r = load_case(case_id, group, demographic=demographic, extent=extent, diagnosis_mapping=diagnosis_mapping)
        else:
            r = load_case(case_id, group, diagnosis_mapping=diagnosis_mapping)
        case_result[group] = r
    results.append(case_result)

import csv

# Write results to CSV file
csv_path = 'clamping_comparison_results.csv'
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['case_id', 'group', 'top_5_features', 'top_5_diagnoses', 'features_changed', 'diagnoses_changed'])
    for case in results:
        # Compare features and diagnoses between groups for highlighting
        features = {g: set(case[g]['top_dxs']) for g in ['with_demo', 'no_demo', 'clamped']}
        diagnoses = {g: set(case[g].get('top_diagnoses', [])) for g in ['with_demo', 'no_demo', 'clamped']}
        # For each group, check if features/diagnoses differ from with_demo
        for group in ['with_demo', 'no_demo', 'clamped']:
            features_changed = features[group] != features['with_demo']
            diagnoses_changed = diagnoses[group] != diagnoses['with_demo']
            writer.writerow([
                case['case_id'],
                group,
                ';'.join(str(f) for f in case[group]['top_dxs']),
                ';'.join(str(d) for d in case[group].get('top_diagnoses', [])),
                'YES' if features_changed else '',
                'YES' if diagnoses_changed else ''
            ])

# Visualize the results
visualize_clamping_analysis(csv_path)

# Print a summary with highlights
for case in results:
    print(f"\n=== Case {case['case_id']} ===")
    base_features = case['with_demo']['top_dxs']
    base_diagnoses = case['with_demo'].get('top_diagnoses', [])
    for group in ['with_demo', 'no_demo', 'clamped']:
        features = case[group]['top_dxs']
        diagnoses = case[group].get('top_diagnoses', [])
        features_changed = features != base_features
        diagnoses_changed = diagnoses != base_diagnoses
        change_note = []
        if features_changed:
            change_note.append('FEATURES CHANGED')
        if diagnoses_changed:
            change_note.append('DIAGNOSES CHANGED')
        print(f"  {group}: Top 5 SAE features: {features} | Top 5 Diagnoses: {diagnoses} {'; '.join(change_note) if change_note else ''}")
