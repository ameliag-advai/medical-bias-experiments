#!/usr/bin/env python3
"""
Overnight Batch Runner for 48-Hour Medical Bias Analysis
Executes all 28 experimental conditions sequentially
"""

import subprocess
import time
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('overnight_batch.log'),
        logging.StreamHandler()
    ]
)

def run_command(cmd, condition_name):
    """Run a single experimental condition"""
    start_time = time.time()
    logging.info(f"🚀 Starting: {condition_name}")
    logging.info(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/Users/amelia/22406alethia/alethia')
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logging.info(f"✅ Completed: {condition_name} ({duration:.1f}s)")
            return True
        else:
            logging.error(f"❌ Failed: {condition_name}")
            logging.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"💥 Exception in {condition_name}: {str(e)}")
        return False

def main():
    """Execute all experimental conditions"""
    
    # Base command template
    base_cmd = "python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0"
    
    # All experimental conditions
    conditions = [
        # Baseline
        (f"{base_cmd}", "baseline_50cases"),
        
        # Single demographic clamping - Male
        (f"{base_cmd} --clamp --clamp-features male --clamp-values 1", "male_1x"),
        (f"{base_cmd} --clamp --clamp-features male --clamp-values 5", "male_5x"),
        (f"{base_cmd} --clamp --clamp-features male --clamp-values 10", "male_10x"),
        
        # Single demographic clamping - Female
        (f"{base_cmd} --clamp --clamp-features female --clamp-values 1", "female_1x"),
        (f"{base_cmd} --clamp --clamp-features female --clamp-values 5", "female_5x"),
        (f"{base_cmd} --clamp --clamp-features female --clamp-values 10", "female_10x"),
        
        # Single demographic clamping - Old
        (f"{base_cmd} --clamp --clamp-features old --clamp-values 1", "old_1x"),
        (f"{base_cmd} --clamp --clamp-features old --clamp-values 5", "old_5x"),
        (f"{base_cmd} --clamp --clamp-features old --clamp-values 10", "old_10x"),
        
        # Single demographic clamping - Young
        (f"{base_cmd} --clamp --clamp-features young --clamp-values 1", "young_1x"),
        (f"{base_cmd} --clamp --clamp-features young --clamp-values 5", "young_5x"),
        (f"{base_cmd} --clamp --clamp-features young --clamp-values 10", "young_10x"),
        
        # Combined demographic clamping - Old + Male
        (f"{base_cmd} --clamp --clamp-features old male --clamp-values 1", "old_male_1x"),
        (f"{base_cmd} --clamp --clamp-features old male --clamp-values 5", "old_male_5x"),
        (f"{base_cmd} --clamp --clamp-features old male --clamp-values 10", "old_male_10x"),
        
        # Combined demographic clamping - Old + Female
        (f"{base_cmd} --clamp --clamp-features old female --clamp-values 1", "old_female_1x"),
        (f"{base_cmd} --clamp --clamp-features old female --clamp-values 5", "old_female_5x"),
        (f"{base_cmd} --clamp --clamp-features old female --clamp-values 10", "old_female_10x"),
        
        # Combined demographic clamping - Young + Male
        (f"{base_cmd} --clamp --clamp-features young male --clamp-values 1", "young_male_1x"),
        (f"{base_cmd} --clamp --clamp-features young male --clamp-values 5", "young_male_5x"),
        (f"{base_cmd} --clamp --clamp-features young male --clamp-values 10", "young_male_10x"),
        
        # Combined demographic clamping - Young + Female
        (f"{base_cmd} --clamp --clamp-features young female --clamp-values 1", "young_female_1x"),
        (f"{base_cmd} --clamp --clamp-features young female --clamp-values 5", "young_female_5x"),
        (f"{base_cmd} --clamp --clamp-features young female --clamp-values 10", "young_female_10x"),
    ]
    
    logging.info(f"🌙 Starting overnight batch run with {len(conditions)} conditions")
    logging.info(f"📊 Total estimated cases: {len(conditions) * 50} = {len(conditions) * 50}")
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i, (cmd, name) in enumerate(conditions, 1):
        logging.info(f"\n📋 Condition {i}/{len(conditions)}: {name}")
        
        if run_command(cmd, name):
            successful += 1
        else:
            failed += 1
            
        # Brief pause between conditions
        time.sleep(2)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    logging.info(f"\n🏁 Batch run completed!")
    logging.info(f"✅ Successful: {successful}")
    logging.info(f"❌ Failed: {failed}")
    logging.info(f"⏱️  Total time: {total_duration/3600:.1f} hours")
    logging.info(f"📁 Check outputs in: src/advai/outputs/")

if __name__ == "__main__":
    main()
