#!/usr/bin/env python
"""
Compare activation values between two output files to analyze clamping effects.
This script specifically looks at how the "male" feature activations changed with clamping.
"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import json


def load_data(file_path):
    """Load CSV data and return a pandas DataFrame."""
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)


def find_male_activation_features(df):
    """Find activation features that correspond to the 'male' concept."""
    # This is a simplification - in reality we'd need to use the concept vectors
    # to identify which features are most associated with 'male'
    # For now, we'll just look at all activation features
    activation_cols = [col for col in df.columns if col.startswith('activation_')]
    return activation_cols


def compare_activations(file1, file2):
    """Compare activation values between two files."""
    df1 = load_data(file1)
    df2 = load_data(file2)
    
    # Find common cases between the two files
    common_ids = set(df1['case_id']).intersection(set(df2['case_id']))
    if not common_ids:
        print("No common cases found between the files")
        return
    
    print(f"Found {len(common_ids)} common cases")
    
    # Filter to common cases
    df1 = df1[df1['case_id'].isin(common_ids)]
    df2 = df2[df2['case_id'].isin(common_ids)]
    
    # Sort by ID to ensure alignment
    df1 = df1.sort_values('case_id')
    df2 = df2.sort_values('case_id')
    
    # Get activation columns
    activation_cols = find_male_activation_features(df1)
    
    # Calculate differences in activations
    differences = {}
    for col in activation_cols:
        if col in df1.columns and col in df2.columns:
            diff = df2[col].values - df1[col].values
            differences[col] = diff
    
    # Analyze the differences
    diff_magnitudes = {col: np.abs(diff).mean() for col, diff in differences.items()}
    
    # Sort by magnitude of difference
    sorted_diffs = sorted(diff_magnitudes.items(), key=lambda x: x[1], reverse=True)
    
    # Print top differences
    print("\nTop 20 features with largest activation differences:")
    for col, diff in sorted_diffs[:20]:
        print(f"{col}: {diff:.4f}")
    
    # Calculate overall statistics
    all_diffs = np.concatenate([diff for diff in differences.values()])
    print(f"\nOverall activation difference statistics:")
    print(f"Mean absolute difference: {np.abs(all_diffs).mean():.4f}")
    print(f"Max absolute difference: {np.abs(all_diffs).max():.4f}")
    print(f"Standard deviation of differences: {np.std(all_diffs):.4f}")
    
    # Count features with significant changes
    significant_threshold = 0.1  # Arbitrary threshold
    significant_features = sum(1 for diff in diff_magnitudes.values() if diff > significant_threshold)
    print(f"Features with significant changes (>{significant_threshold}): {significant_features} out of {len(activation_cols)}")
    
    # Check if clamping was applied by looking at features_clamped column
    if 'features_clamped' in df2.columns:
        clamped_features = df2['features_clamped'].iloc[0]
        clamping_levels = df2['clamping_levels'].iloc[0] if 'clamping_levels' in df2.columns else 'Unknown'
        print(f"\nClamping applied to features: {clamped_features}")
        print(f"Clamping level: {clamping_levels}")
    
    # Create visualization directory
    output_dir = os.path.dirname(file2)
    vis_dir = os.path.join(output_dir, "activation_comparison")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot histogram of differences
    plt.figure(figsize=(10, 6))
    plt.hist(all_diffs, bins=50)
    plt.title('Histogram of Activation Differences')
    plt.xlabel('Difference (Clamped - Unclamped)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(vis_dir, 'activation_differences_histogram.png'))
    
    # Plot top 10 features with largest differences
    top_features = [col for col, _ in sorted_diffs[:10]]
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(top_features):
        plt.subplot(5, 2, i+1)
        plt.bar(['Unclamped', 'Clamped'], [df1[feature].mean(), df2[feature].mean()])
        plt.title(f'{feature}')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'top_features_comparison.png'))
    
    # Save results to JSON
    results = {
        'file1': os.path.basename(file1),
        'file2': os.path.basename(file2),
        'common_cases': len(common_ids),
        'mean_absolute_difference': float(np.abs(all_diffs).mean()),
        'max_absolute_difference': float(np.abs(all_diffs).max()),
        'std_difference': float(np.std(all_diffs)),
        'significant_features': significant_features,
        'top_differences': [{
            'feature': col,
            'mean_difference': float(diff)
        } for col, diff in sorted_diffs[:50]]
    }
    
    with open(os.path.join(vis_dir, 'activation_comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults and visualizations saved to {vis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare activation values between two output files")
    parser.add_argument("file1", help="Path to first CSV file (unclamped)")
    parser.add_argument("file2", help="Path to second CSV file (clamped)")
    args = parser.parse_args()
    
    compare_activations(args.file1, args.file2)
