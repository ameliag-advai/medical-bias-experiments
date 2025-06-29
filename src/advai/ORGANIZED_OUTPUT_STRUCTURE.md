# Organized Output Structure

## Overview
All experimental runs now create organized subfolders with timestamped names to prevent overwriting and enable easy tracking of multiple experiments.

## Folder Structure

```
src/advai/outputs/
├── 20250628_202758_baseline/
│   ├── results_database.csv
│   └── run_summary.txt
├── 20250628_202759_clamping_male/
│   ├── results_database.csv
│   └── run_summary.txt
├── 20250628_202800_clamping_female/
│   ├── results_database.csv
│   └── run_summary.txt
├── 20250628_202801_clamping_male-female/
│   ├── results_database.csv
│   └── run_summary.txt
└── 20250628_202802_baseline/
    ├── results_database.csv
    └── run_summary.txt
```

## Folder Naming Convention

**Format**: `YYYYMMDD_HHMMSS_EXPERIMENT_TYPE[_FEATURES]`

### Components:
1. **Timestamp**: `YYYYMMDD_HHMMSS` (e.g., `20250628_202758`)
2. **Experiment Type**: 
   - `baseline` - No clamping experiments
   - `clamping` - Feature clamping experiments
3. **Features** (for clamping only): 
   - Single feature: `_male`, `_female`, `_young`, `_old`
   - Multiple features: `_male-female`, `_young-old`

### Examples:
- `20250628_202758_baseline` - Baseline experiment with demographics
- `20250628_202759_clamping_male` - Clamping male features
- `20250628_202800_clamping_female` - Clamping female features  
- `20250628_202801_clamping_male-female` - Clamping both male and female features

## File Contents

Each subfolder contains:
- **`results_database.csv`** - Main results file with standardized format
  - All cases from the experiment run
  - Consistent column structure (from `constants_v2.py`)
  - Includes activation columns (`activation_0` through `activation_N`)
- **`run_summary.txt`** - Detailed run metadata and timing information
  - Start/end times and total duration
  - Number of cases processed and experimental parameters
  - Command executed and clamping configuration
  - Data sources and model information
  - Complete run audit trail

## Benefits

1. **No Overwriting**: Each run gets its own timestamped folder
2. **Easy Identification**: Folder names clearly indicate experiment type and parameters
3. **Batch Analysis**: Updated batch runner automatically finds all results in subfolders
4. **Organization**: Clean separation of different experimental conditions
5. **Reproducibility**: Easy to track and compare different runs

## Updated Tools

### Pipeline (`src/advai/analysis/pipeline.py`)
- Automatically creates timestamped subfolders
- Uses consistent `results_database.csv` filename
- Prints output directory for confirmation

### Batch Runner (`src/advai/48_hour_batch_runner.py`)
- Updated to search for CSV files in subfolders
- Analyzes all `results_database.csv` files found
- Reports progress by subfolder name

### Analysis Pipeline (`src/advai/analysis/data_analysis_pipeline.py`)
- Works seamlessly with new structure
- Generates analysis results in same subfolder as source data

## Usage

No changes needed to existing commands - the new structure is automatic:

```bash
# Single run - creates timestamped subfolder automatically
python -m src.advai.main --device cpu --num-cases 10

# Batch runs - each gets its own subfolder
python src/advai/48_hour_batch_runner.py --mode all --num-cases 50

# Analysis - automatically finds all subfolders
python src/advai/48_hour_batch_runner.py --mode analyze
```

## Migration

Existing CSV files in the main `outputs/` directory will still be found and analyzed, but new runs will use the organized subfolder structure.
