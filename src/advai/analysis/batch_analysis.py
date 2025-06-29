import os
import json
import glob
import argparse
import pandas as pd


def analyze_directory(directory):
    """Analyzes all analysis JSON files in a given directory."""
    search_path = os.path.join(directory, "*_analysis_results.json")
    analysis_files = glob.glob(search_path)

    if not analysis_files:
        print(f"No analysis files found in {directory}")
        return

    summary_data = []

    for f in sorted(analysis_files):
        with open(f, 'r') as file:
            data = json.load(file)

            # Extract filename to identify the run
            filename = os.path.basename(f).replace('_analysis_results.json', '.csv')

            # Correctly extract accuracy
            accuracy = data.get('accuracy', {}).get('top1_accuracy', 'N/A')

            # Correctly extract bias metrics for dataset_sex
            bias_data = data.get('bias', {})
            cohen_d = bias_data.get('dataset_sex_effect_size', 'N/A')
            p_value = bias_data.get('dataset_sex_t_test', {}).get('p_value', 'N/A')

            summary_data.append({
                'file': filename,
                'accuracy': accuracy,
                'dataset_sex_cohen_d': cohen_d,
                'dataset_sex_p_value': p_value
            })

    # Create and display summary table
    summary_df = pd.DataFrame(summary_data)
    print("\n--- Analysis Summary ---")
    print(summary_df)

    # Save summary to CSV
    summary_csv_path = os.path.join(directory, 'batch_analysis_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary saved to {summary_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch analysis on clamping experiment results.")
    parser.add_argument("directory", type=str, help="Directory containing the analysis JSON files.")
    args = parser.parse_args()
    analyze_directory(args.directory)
