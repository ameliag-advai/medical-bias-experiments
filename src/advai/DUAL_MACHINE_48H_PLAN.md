# Dual-Machine 48-Hour Medical Bias Detection Plan
**Mac Mini M4 + Large Server Strategy**  
**Start Time**: June 28, 8:00 PM  
**End Time**: June 30, 8:00 PM  

## Machine Allocation Strategy

### **Mac Mini M4** (Development & Quick Tests)
- **Strengths**: Fast for development, immediate feedback, good for debugging
- **Use for**: Code development, small test runs (1-5 cases), debugging, analysis
- **Limitations**: Smaller memory, fewer cores for large batch runs

### **Large Server** (Production Runs)
- **Strengths**: High memory, many cores, can handle large datasets
- **Use for**: Full experimental runs (50-100+ cases), overnight batch processing
- **Limitations**: Less interactive, setup time for transfers

---

## PHASE 1: Mac Mini Setup & Validation (Tonight: 8:00 PM - Midnight)

### 1A: Immediate Validation (8:00-9:00 PM) - Mac Mini
**CRITICAL**: Test clamping values and output format

```bash
# Test single case with different clamping values
cd /Users/amelia/22406alethia/alethia

# Test baseline (no clamping)
python -m src.advai.main --device cpu --num-cases 1 --patient-file src/advai/results_database.csv

# Test clamping values: 1, 5, 10, 100
python -m src.advai.main --device cpu --num-cases 1 --patient-file src/advai/results_database.csv --clamp --clamp-features male --clamp-values 1
python -m src.advai.main --device cpu --num-cases 1 --patient-file src/advai/results_database.csv --clamp --clamp-features male --clamp-values 5
python -m src.advai.main --device cpu --num-cases 1 --patient-file src/advai/results_database.csv --clamp --clamp-features male --clamp-values 10
python -m src.advai.main --device cpu --num-cases 1 --patient-file src/advai/results_database.csv --clamp --clamp-features male --clamp-values 100
```

**Expected Output**: `results_database_YYYYMMDD_HHMMSS.csv` files in `src/advai/outputs/`

### 1B: Verify CSV Format Compliance (9:00-9:30 PM) - Mac Mini
Check that output files match the expected format:
- Headers match `FIELD_NAMES` or `CLAMPING_FIELD_NAMES` from `constants_v2.py`
- Activation columns: `activation_0` through `activation_N` (where N = SAE dimensions)
- Clamping info properly recorded in `features_clamped` and `clamping_levels` columns

### 1C: Create Server Transfer Scripts (9:30-10:30 PM) - Mac Mini
```bash
# Create server sync script
cat > sync_to_server.sh << 'EOF'
#!/bin/bash
# Sync codebase to server
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
  /Users/amelia/22406alethia/alethia/ \
  user@server:/path/to/alethia/

# Sync results back from server
rsync -avz user@server:/path/to/alethia/src/advai/outputs/ \
  /Users/amelia/22406alethia/alethia/src/advai/outputs/
EOF
chmod +x sync_to_server.sh
```

### 1D: Create Experiment Configuration (10:30-11:30 PM) - Mac Mini
```python
# File: src/advai/experiments/experiment_config.py
BASELINE_EXPERIMENTS = [
    {"name": "age_sex", "demographics": ["age", "sex"], "clamp": False},
    {"name": "age_only", "demographics": ["age"], "clamp": False},
    {"name": "sex_only", "demographics": ["sex"], "clamp": False},
    {"name": "no_demo", "demographics": [], "clamp": False},
]

CLAMPING_EXPERIMENTS = [
    # Test each demographic feature at multiple intensities
    {"name": f"clamp_{feature}_{intensity}x", 
     "demographics": [], 
     "clamp": True, 
     "clamp_features": [feature], 
     "clamp_values": [intensity]}
    for feature in ["male", "female", "young", "old"]
    for intensity in [1, 5, 10, 100]
]

# Total: 4 baseline + 16 clamping = 20 experiments
```

### 1E: Update Batch Runner for CSV Format (11:30 PM-Midnight) - Mac Mini
Ensure batch runner uses the correct output format and file naming from `pipeline.py`.

---

## PHASE 2: Server Setup & Large Runs (Day 2: 8:00 AM - Midnight)

### 2A: Server Preparation (8:00-10:00 AM) - Both Machines
**Mac Mini**: Sync code to server
**Server**: Set up environment, test single case

```bash
# On server - test environment
python -m src.advai.main --device cpu --num-cases 1 --patient-file src/advai/results_database.csv
```

### 2B: Parallel Development Strategy (10:00 AM - 6:00 PM)

#### **Mac Mini Tasks** (Interactive Development):
- **Analysis Pipeline Development** (10:00 AM - 2:00 PM)
  - Create enhanced analysis scripts for the CSV format
  - Develop visualization tools for results comparison
  - Test analysis on small datasets (1-5 cases)

- **Real-time Monitoring Tools** (2:00 PM - 4:00 PM)
  - Create server monitoring scripts
  - Develop progress tracking for batch runs
  - Set up automated result collection

- **Statistical Analysis Framework** (4:00 PM - 6:00 PM)
  - Implement bias detection algorithms
  - Create statistical significance testing
  - Develop effect size calculations

#### **Server Tasks** (Background Processing):
- **Medium-scale Testing** (10:00 AM - 2:00 PM)
  - Run all 20 experiments with 10 cases each
  - Validate output format consistency
  - Test error handling and recovery

- **Large-scale Preparation** (2:00 PM - 6:00 PM)
  - Prepare datasets with 50-100 cases
  - Set up batch processing queue
  - Configure overnight run parameters

### 2C: Evening Coordination (6:00 PM - Midnight)
**6:00-8:00 PM**: Sync results from server, analyze medium-scale runs on Mac Mini
**8:00-10:00 PM**: Start large-scale overnight runs on server (50-100 cases × 20 experiments)
**10:00 PM-Midnight**: Mac Mini analysis of preliminary results, prepare final analysis pipeline

---

## PHASE 3: Results Analysis (Day 3: 8:00 AM - 8:00 PM)

### 3A: Results Collection (8:00-10:00 AM)
**Server**: Complete overnight runs, prepare results for transfer
**Mac Mini**: Sync all results, validate data integrity

### 3B: Comprehensive Analysis (10:00 AM - 6:00 PM) - Mac Mini
- **Statistical Analysis** (10:00 AM - 2:00 PM)
  - Compare baseline vs clamping effects
  - Calculate effect sizes and significance
  - Identify optimal clamping intensities

- **Visualization Generation** (2:00 PM - 4:00 PM)
  - Create comprehensive plots and heatmaps
  - Generate comparison matrices
  - Produce publication-ready figures

- **Report Generation** (4:00 PM - 6:00 PM)
  - Automated report compilation
  - Key findings extraction
  - Methodology documentation

### 3C: Final Validation (6:00-8:00 PM) - Both Machines
**Server**: Run validation experiments on key findings
**Mac Mini**: Final analysis and documentation

---

## Key Technical Details

### CSV Output Format (from `constants_v2.py`):
```python
# Baseline experiments use FIELD_NAMES:
["case_id", "dataset_age", "dataset_sex", "dataset_symptoms", "diagnosis", 
 "prompt", "demographics", "prompt_age", "prompt_sex", "features_clamped", 
 "clamping_levels", "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4", 
 "diagnosis_5", "diagnosis_1_logits", "diagnosis_2_logits", "diagnosis_3_logits", 
 "diagnosis_4_logits", "diagnosis_5_logits", "top5", "top5_logits", 
 "correct_top1", "correct_top5"] + ["activation_0", ..., "activation_N"]

# Clamping experiments use CLAMPING_FIELD_NAMES (same structure)
```

### File Naming Convention:
- Output files: `results_database_YYYYMMDD_HHMMSS.csv`
- Location: `src/advai/outputs/`
- Generated by: `run_analysis_pipeline()` in `pipeline.py`

### Clamping Values to Test:
Based on your memory about bidirectional activations:
- **Intensities**: 1x, 5x, 10x, 100x (not just scaling, but setting to specific values)
- **Features**: male, female, young, old (from demographic concept vectors)
- **Strategy**: Set features to positive/negative values rather than just scaling

### Resource Allocation:
- **Mac Mini**: Development (1-5 cases), analysis, visualization
- **Server**: Production runs (50-100 cases), batch processing, overnight runs
- **Transfer**: Automated sync scripts for code and results

### Success Metrics:
- **Statistical Power**: 50+ cases per condition for significance testing
- **Effect Detection**: Cohen's d > 0.2 for meaningful bias effects
- **Coverage**: All 20 experimental conditions completed
- **Reproducibility**: Consistent results across multiple runs

This plan maximizes both machines' strengths while ensuring the output format matches your existing analysis pipeline.
