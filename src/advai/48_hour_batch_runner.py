#!/usr/bin/env python3
"""
Batch runner for systematic bias analysis experiments.
Runs all combinations of demographic scenarios and clamping configurations.
"""

import subprocess
import time
import os
import json
from datetime import datetime
import argparse


def run_command(cmd, log_file=None, background=False):
    """Run a command with optional logging and background execution."""
    print(f"🚀 Running: {' '.join(cmd)}")
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    if background:
        # Run in background with nohup
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                ['nohup'] + cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp
            )
        print(f"📝 Background process started (PID: {process.pid}), logging to {log_file}")
        return process
    else:
        # Run in foreground
        if log_file:
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        else:
            result = subprocess.run(cmd)
        return result


def run_baseline_scenarios(num_cases=50, device='cpu', background=False):
    """Run all baseline scenarios (no clamping)."""
    print("=" * 60)
    print("🎯 RUNNING BASELINE SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        {
            'name': 'with_demographics',
            'args': [],
            'description': 'Full demographics (age + sex)'
        },
        {
            'name': 'no_demographics', 
            'args': ['--no-demographics'],
            'description': 'No demographic information'
        }
    ]
    
    processes = []
    
    for scenario in scenarios:
        cmd = [
            'python', '-m', 'src.advai.main',
            '--device', device,
            '--num-cases', str(num_cases),
            '--patient-file', 'src/advai/results_database.csv'
        ] + scenario['args']
        
        log_file = f"logs/baseline_{scenario['name']}.log"
        
        print(f"\n📊 Scenario: {scenario['description']}")
        
        if background:
            process = run_command(cmd, log_file, background=True)
            processes.append({
                'process': process,
                'name': scenario['name'],
                'description': scenario['description']
            })
        else:
            result = run_command(cmd, log_file, background=False)
            if result.returncode != 0:
                print(f"❌ Failed: {scenario['name']}")
            else:
                print(f"✅ Completed: {scenario['name']}")
    
    return processes


def run_clamping_experiments(num_cases=50, device='cpu', background=False):
    """Run systematic clamping experiments."""
    print("=" * 60)
    print("🔧 RUNNING CLAMPING EXPERIMENTS")
    print("=" * 60)
    
    # Clamping configurations
    features = ['male', 'female', 'young', 'old']
    intensities = [1, 5, 10, 100]
    
    processes = []
    
    for feature in features:
        for intensity in intensities:
            cmd = [
                'python', '-m', 'src.advai.main',
                '--device', device,
                '--num-cases', str(num_cases),
                '--patient-file', 'src/advai/results_database.csv',
                '--no-demographics',  # Base scenario: no demographics
                '--clamp',
                '--clamp-features', feature,
                '--clamp-values', str(intensity)
            ]
            
            scenario_name = f"clamp_{feature}_{intensity}x"
            log_file = f"logs/clamping_{scenario_name}.log"
            
            print(f"\n🔧 Clamping: {feature} at {intensity}x intensity")
            
            if background:
                process = run_command(cmd, log_file, background=True)
                processes.append({
                    'process': process,
                    'name': scenario_name,
                    'feature': feature,
                    'intensity': intensity
                })
            else:
                result = run_command(cmd, log_file, background=False)
                if result.returncode != 0:
                    print(f"❌ Failed: {scenario_name}")
                else:
                    print(f"✅ Completed: {scenario_name}")
    
    return processes


def monitor_processes(processes):
    """Monitor background processes and report completion."""
    print(f"\n👀 Monitoring {len(processes)} background processes...")
    
    completed = []
    
    while len(completed) < len(processes):
        time.sleep(30)  # Check every 30 seconds
        
        for proc_info in processes:
            if proc_info['name'] not in completed:
                if proc_info['process'].poll() is not None:
                    # Process completed
                    completed.append(proc_info['name'])
                    if proc_info['process'].returncode == 0:
                        print(f"✅ Completed: {proc_info['name']}")
                    else:
                        print(f"❌ Failed: {proc_info['name']} (exit code: {proc_info['process'].returncode})")
    
    print(f"🎉 All {len(processes)} processes completed!")


def analyze_results():
    """Run analysis pipeline on all output files."""
    print("=" * 60)
    print("📊 ANALYZING RESULTS")
    print("=" * 60)
    
    # Find all output CSV files in subfolders
    output_dir = "src/advai/outputs"
    csv_files = []
    
    # Look for results_database.csv files in timestamped subfolders
    if os.path.exists(output_dir):
        for subfolder in os.listdir(output_dir):
            subfolder_path = os.path.join(output_dir, subfolder)
            if os.path.isdir(subfolder_path):
                csv_file = os.path.join(subfolder_path, "results_database.csv")
                if os.path.exists(csv_file):
                    csv_files.append(csv_file)
    
    print(f"Found {len(csv_files)} output files to analyze")
    
    # Run analysis on each file
    for csv_file in csv_files:
        subfolder_name = os.path.basename(os.path.dirname(csv_file))
        print(f"\n📈 Analyzing: {subfolder_name}/results_database.csv")
        cmd = ['python', '-m', 'src.advai.analysis.data_analysis_pipeline', csv_file]
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f"✅ Analysis completed for {subfolder_name}")
        else:
            print(f"❌ Analysis failed for {subfolder_name}")
    
    # Run batch comparison
    print(f"\n📊 Running batch comparison...")
    cmd = ['python', 'src/advai/analysis/batch_analysis.py', output_dir]
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"✅ Batch analysis completed")
    else:
        print(f"❌ Batch analysis failed")


def create_experiment_log():
    """Create a log of the experiment configuration."""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'systematic_bias_analysis',
        'scenarios': {
            'baseline': ['with_demographics', 'no_demographics'],
            'clamping': {
                'features': ['male', 'female', 'young', 'old'],
                'intensities': [1, 5, 10, 100]
            }
        },
        'total_experiments': 2 + (4 * 4),  # 2 baseline + 16 clamping
        'estimated_runtime_hours': 8
    }
    
    os.makedirs('logs', exist_ok=True)
    with open('logs/experiment_log.json', 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"📝 Experiment log saved to logs/experiment_log.json")


def main():
    parser = argparse.ArgumentParser(description="Batch runner for bias analysis experiments")
    parser.add_argument('--mode', choices=['baseline', 'clamping', 'all', 'analyze'], 
                       default='all', help='Which experiments to run')
    parser.add_argument('--num-cases', type=int, default=50, 
                       help='Number of cases per experiment')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--background', action='store_true', 
                       help='Run experiments in background')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor background processes')
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create experiment log
    create_experiment_log()
    
    all_processes = []
    
    if args.mode in ['baseline', 'all']:
        processes = run_baseline_scenarios(
            num_cases=args.num_cases,
            device=args.device,
            background=args.background
        )
        all_processes.extend(processes)
    
    if args.mode in ['clamping', 'all']:
        processes = run_clamping_experiments(
            num_cases=args.num_cases,
            device=args.device,
            background=args.background
        )
        all_processes.extend(processes)
    
    if args.background and all_processes and args.monitor:
        monitor_processes(all_processes)
    
    if args.mode == 'analyze':
        analyze_results()
    
    print(f"\n🎉 Batch runner completed!")
    print(f"📊 Total experiments: {len(all_processes)}")
    print(f"📝 Check logs/ directory for detailed output")
    print(f"📈 Run with --mode analyze to process results")


if __name__ == "__main__":
    main()
