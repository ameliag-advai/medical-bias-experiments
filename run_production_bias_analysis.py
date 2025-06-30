#!/usr/bin/env python3
"""
Simple runner for the production bias analysis experiment.
Uses the original alethia data files and generates results in the original format.
"""

import os
import sys

# Add the project root to the path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.advai.production_bias_experiment import main

if __name__ == "__main__":
    # Set default arguments for the alethia project
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Production Bias Analysis")
    parser.add_argument("--num-cases", type=int, default=100, help="Number of cases to process")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Use the available data files
    patient_file = "/Users/amelia/demographic-clamping-analysis/release-test-patients-age-grouped.csv"
    conditions_file = "/Users/amelia/demographic-clamping-analysis/release_conditions.json"
    
    # Check if files exist
    if not os.path.exists(patient_file):
        print(f"❌ Patient data file not found: {patient_file}")
        print("Please ensure the patient data file exists in the expected location.")
        sys.exit(1)
    
    if not os.path.exists(conditions_file):
        print(f"❌ Conditions mapping file not found: {conditions_file}")
        print("Please ensure the conditions mapping file exists in the expected location.")
        sys.exit(1)
    
    # Override sys.argv to pass arguments to the main script
    sys.argv = [
        "production_bias_experiment.py",
        "--patient-file", patient_file,
        "--conditions-file", conditions_file,
        "--num-cases", str(args.num_cases),
        "--device", args.device
    ]
    
    print("🚀 Starting production bias analysis with alethia data...")
    print(f"📊 Processing {args.num_cases} cases on {args.device}")
    print(f"📁 Patient data: {patient_file}")
    print(f"📁 Conditions: {conditions_file}")
    
    main()
