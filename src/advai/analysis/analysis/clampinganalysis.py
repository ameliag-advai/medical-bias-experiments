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
from advai.analysis.clamping import clamp_sae_features
from advai.data.io import load_patient_data, extract_cases_from_dataframe, load_conditions_mapping
from advai.data.build_prompts import build_prompts
from advai.analysis.analyse import analyse_case_for_bias
import csv

# Directory where previous analysis results and activations are saved
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../activations'))

# Find all available case ids in the activations directory
from typing import Set

def get_case_ids() -> Set[str]:
    files = os.listdir(RESULTS_DIR)
    case_ids: Set[str] = set()
    for fname in files:
        if fname.startswith('case_') and fname.endswith('_activations.pt'):
            # Extract the case id (handles e.g. case_0_activations.pt)
            num = fname[len('case_'):fname.rfind('_activations.pt')]
            case_ids.add(str(num))
    case_ids = sorted(case_ids, key=lambda x: int(x) if x.isdigit() else x)
    return set(case_ids)

# Helper to load a single case's saved result
def load_case(case_id, group, demographic=None, extent=None, diagnosis_mapping=None):
    # group: 'with_demo', 'no_demo', or 'clamped'
    fname = os.path.join(RESULTS_DIR, f"case_{case_id}_activations.pt")
    d = torch.load(fname)
    # For 'clamped', clamp the SAE output with user-specified demographic and extent
def get_top_dxs_and_diagnoses(sae_out, diagnosis_mapping):
    top_dxs = torch.topk(sae_out[0], 5).indices.tolist()
    top_diagnoses = [diagnosis_mapping.get(str(idx), f"Diagnosis {idx}") for idx in top_dxs]
    return top_dxs, top_diagnoses


def get_clamped_sae_and_diagnoses(sae_out_without, demographic, extent, diagnosis_mapping):
    clamped_sae = clamp_sae_features(torch.unsqueeze(sae_out_without, 0), demographic=demographic, extent=extent)
    top_dxs, top_diagnoses = get_top_dxs_and_diagnoses(clamped_sae, diagnosis_mapping)
    return clamped_sae, top_dxs, top_diagnoses

    if group == 'clamped':
        if demographic is None or extent is None:
            raise ValueError('Demographic and extent must be specified for clamped group.')
        sae_out, top_dxs, top_diagnoses = get_clamped_sae_and_diagnoses(d['sae_out_without'], demographic, extent, diagnosis_mapping)
    elif group == 'with_demo':
        sae_out = torch.unsqueeze(d['sae_out_with'], 0)
        top_dxs, top_diagnoses = get_top_dxs_and_diagnoses(sae_out, diagnosis_mapping)
    elif group == 'no_demo':
        sae_out = torch.unsqueeze(d['sae_out_without'], 0)
        top_dxs, top_diagnoses = get_top_dxs_and_diagnoses(sae_out, diagnosis_mapping)
    else:
        raise ValueError('Unknown group')
    # Convert tensors to lists for JSON serialization
    return {
        'sae_out': sae_out.cpu().numpy().tolist(),
        'top_dxs': top_dxs,
        'top_diagnoses': top_diagnoses,
        'case_id': case_id
    }

import argparse

def main():
    parser = argparse.ArgumentParser(description="Clamping analysis for SAE features.")
    parser.add_argument('--demographic', type=str, required=True, choices=['male', 'female', 'old', 'young'], help='Demographic to clamp (male, female, old, young)')
    parser.add_argument('--extent', type=float, required=True, help='Extent to clamp (e.g. 5 for 5x)')
    args = parser.parse_args()

    # Load diagnosis mapping from release_conditions.json
    # Always look for release_conditions.json in the project root
    RELEASE_CONDITIONS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../release_conditions.json'))
    if not os.path.exists(RELEASE_CONDITIONS_PATH):
        raise FileNotFoundError(f"release_conditions.json not found at {RELEASE_CONDITIONS_PATH}")
    release_conditions = load_conditions_mapping(RELEASE_CONDITIONS_PATH)
    # Map feature index (as string) to natural language diagnosis
    # (Assume the mapping is {str(idx): 'diagnosis name'})
    diagnosis_mapping = {str(idx): v['cond-name-eng'] if 'cond-name-eng' in v else k for idx, (k, v) in enumerate(release_conditions.items())}

    demographic = args.demographic
    extent = args.extent

    # Compare diagnoses/features for all detected cases
    case_ids = sorted(get_case_ids(), key=lambda x: int(x) if x.isdigit() else x)
    results = []
    for case_id in case_ids:
        case_result = {'case_id': case_id}
        for group in ['with_demo', 'no_demo', 'clamped']:
            r = load_case(case_id, group, demographic=demographic, extent=extent, diagnosis_mapping=diagnosis_mapping)
            case_result[group] = r
        results.append(case_result)

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
                    str(features_changed),
                    str(diagnoses_changed)
                ])

    # Visualization step skipped: visualize_clamping_analysis(csv_path) does not exist
    # To visualize, implement or call a valid plotting function here.

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


if __name__ == "__main__":
    main()
