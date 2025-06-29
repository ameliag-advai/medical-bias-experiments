"""
Batch Analysis Script for Multiple Data Files
Automatically processes all CSV files in a directory
"""

import os
import glob
from pathlib import Path
import argparse
from data_analysis_pipeline import MedicalDataAnalyzer
import json
import pandas as pd


def batch_analyze_directory(directory: str, pattern: str = "*.csv", output_summary: bool = True):
    """
    Analyze all CSV files matching pattern in directory.
    
    Args:
        directory: Directory to search for files
        pattern: File pattern to match (default: *.csv)
        output_summary: Whether to create a summary report
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"❌ Directory {directory} does not exist")
        return
    
    # Find all matching files
    files = list(directory.glob(pattern))
    
    if not files:
        print(f"❌ No files matching '{pattern}' found in {directory}")
        return
    
    print(f"🔍 Found {len(files)} files to analyze:")
    for f in files:
        print(f"   📄 {f.name}")
    
    print("\n" + "="*60)
    
    # Analyze each file
    all_results = {}
    
    for i, file_path in enumerate(files, 1):
        print(f"\n🚀 ANALYZING FILE {i}/{len(files)}: {file_path.name}")
        print("-" * 50)
        
        try:
            analyzer = MedicalDataAnalyzer(file_path)
            results = analyzer.run_full_analysis(save_results=True, generate_plots=True)
            
            if results:
                all_results[str(file_path)] = results
                print(f"✅ Analysis complete for {file_path.name}")
            else:
                print(f"⚠️  No results for {file_path.name}")
                
        except Exception as e:
            print(f"❌ Error analyzing {file_path.name}: {e}")
            continue
    
    # Create summary report
    if output_summary and all_results:
        create_summary_report(all_results, directory)
    
    print(f"\n🎉 BATCH ANALYSIS COMPLETE!")
    print(f"📊 Successfully analyzed {len(all_results)}/{len(files)} files")


def create_summary_report(all_results: dict, output_dir: Path):
    """Create a summary report comparing all analyzed files."""
    print(f"\n📋 CREATING SUMMARY REPORT...")
    
    summary_data = []
    
    for file_path, results in all_results.items():
        file_name = Path(file_path).name
        
        # Extract key metrics
        row = {
            'file_name': file_name,
            'total_rows': results.get('metadata', {}).get('total_rows', 0),
            'activation_features': results.get('metadata', {}).get('activation_features', 0),
        }
        
        # Accuracy metrics
        accuracy = results.get('accuracy', {})
        row['top1_accuracy'] = accuracy.get('top1_accuracy', 0)
        row['top5_accuracy'] = accuracy.get('top5_accuracy', 0)
        
        # Feature statistics
        features = results.get('features', {})
        if 'activation_stats' in features:
            stats = features['activation_stats']
            row['mean_activation'] = stats.get('mean', 0)
            row['activation_sparsity'] = stats.get('sparsity', 0)
        
        # Demographics
        demographics = results.get('demographics', {})
        for demo_col in ['dataset_sex', 'dataset_age']:
            if f'{demo_col}_distribution' in demographics:
                dist = demographics[f'{demo_col}_distribution']
                row[f'{demo_col}_samples'] = sum(dist.values()) if dist else 0
        
        summary_data.append(row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_path = output_dir / 'batch_analysis_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed results
    detailed_path = output_dir / 'batch_analysis_detailed.json'
    with open(detailed_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"📊 Summary saved to: {summary_path}")
    print(f"📋 Detailed results saved to: {detailed_path}")
    
    # Print summary table
    print(f"\n📈 SUMMARY TABLE:")
    print(summary_df.to_string(index=False))


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Batch Analysis for Medical Data Files')
    parser.add_argument('directory', help='Directory containing CSV files to analyze')
    parser.add_argument('--pattern', default='*.csv', help='File pattern to match (default: *.csv)')
    parser.add_argument('--no-summary', action='store_true', help='Skip creating summary report')
    
    args = parser.parse_args()
    
    batch_analyze_directory(
        directory=args.directory,
        pattern=args.pattern,
        output_summary=not args.no_summary
    )


if __name__ == "__main__":
    main()
