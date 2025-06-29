# Non-48 Hour Project Files

This folder contains files that are not needed for the 48-hour medical bias analysis project but may be useful for future development or reference.

## Moved Files and Reasons:

### Analysis Files (Outdated/Replaced):
- **`analysis/clamping.py`** - Old clamping implementation, replaced by `clamping_v2.py`
- **`analysis/constants.py`** - Old constants, replaced by `constants_v2.py`
- **`analysis/clamping_analysis.py`** - Post-hoc analysis tool, not needed for main pipeline
- **`analysis/compare_activations.py`** - Debugging/comparison tool, not needed for production

### Old Results and Outputs:
- **`old_outputs/`** - Contains all previous experimental results:
  - `results_database_*.csv` - Old timestamped results
  - `results_database_*_analysis/` - Old analysis directories
  - `results_database_*.json` - Old analysis JSON files
  - `batch_analysis_summary.csv` - Old batch analysis summary
  - `activation_comparison/` - Old activation comparison results

### Root Level Files:
- **`results_database.csv`** - Old results file from root directory

## Key Differences:

### Clamping v1 vs v2:
- **v1 (`clamping.py`)**: Simple multiplication approach, doesn't handle bidirectional activations properly
- **v2 (`clamping_v2.py`)**: Advanced directional clamping that handles positive/negative activations correctly

### Constants v1 vs v2:
- **v1 (`constants.py`)**: Basic feature lists
- **v2 (`constants_v2.py`)**: Enhanced with directional information and statistical analysis

## Usage Notes:
- Files in this folder are **NOT** used by the 48-hour project pipeline
- The main pipeline uses only the v2 versions and current analysis tools
- These files are preserved for reference and potential future use
- Do not import from this folder in the main codebase

## Restoration:
If you need any of these files back in the main codebase:
1. Copy (don't move) the file back to its original location
2. Update any import statements as needed
3. Test thoroughly before using in production
