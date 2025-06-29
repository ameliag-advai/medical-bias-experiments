# 48-Hour Batch Command List
**Generated**: June 28, 2025 - 9:55 PM  
**Total Conditions**: 28  
**Cases per condition**: 50-100 (adjustable)

## Base Command Template
```bash
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0
```

## 1. BASELINE CONDITIONS (4 commands)

### 1.1 Pure Baseline (no demographics, no clamping)
```bash
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0
```

### 1.2 Demographic Prompts Only (no clamping)
```bash
# Note: This would require implementing demographic prompt variations in the pipeline
# For now, baseline covers this case
```

### 1.3-1.4 Reserved for future prompt variations

## 2. SINGLE DEMOGRAPHIC CLAMPING (12 commands)

### 2.1 Male Clamping (3 intensities)
```bash
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features male --clamp-values 1
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features male --clamp-values 5
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features male --clamp-values 10
```

### 2.2 Female Clamping (3 intensities)
```bash
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features female --clamp-values 1
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features female --clamp-values 5
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features female --clamp-values 10
```

### 2.3 Old Clamping (3 intensities)
```bash
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features old --clamp-values 1
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features old --clamp-values 5
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features old --clamp-values 10
```

### 2.4 Young Clamping (3 intensities)
```bash
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features young --clamp-values 1
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features young --clamp-values 5
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features young --clamp-values 10
```

## 3. COMBINED DEMOGRAPHIC CLAMPING (12 commands)

### 3.1 Old + Male (3 intensities)
```bash
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features old male --clamp-values 1
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features old male --clamp-values 5
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features old male --clamp-values 10
```

### 3.2 Old + Female (3 intensities)
```bash
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features old female --clamp-values 1
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features old female --clamp-values 5
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features old female --clamp-values 10
```

### 3.3 Young + Male (3 intensities)
```bash
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features young male --clamp-values 1
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features young male --clamp-values 5
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features young male --clamp-values 10
```

### 3.4 Young + Female (3 intensities)
```bash
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features young female --clamp-values 1
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features young female --clamp-values 5
python -m src.advai.main --device cpu --num-cases 50 --patient-file /Users/amelia/22406alethia/alethia/release_test_patients --start-case 0 --clamp --clamp-features young female --clamp-values 10
```

## EXECUTION NOTES

### Overnight Batch Execution (11 PM - 7 AM)
- **Total Runtime Estimate**: ~8 hours for 25 conditions × 50 cases each
- **Average per condition**: ~19 minutes (based on 2:22 for 1 case)
- **Total cases**: 1,250 cases
- **Output**: 25 timestamped folders in `src/advai/outputs/`

### Monitoring Commands
```bash
# Check running processes
ps aux | grep python

# Monitor output directories
ls -la src/advai/outputs/

# Check latest run summary
tail -f src/advai/outputs/*/run_summary.txt
```

### Error Recovery
- Each command is independent
- Failed runs can be restarted individually
- Output folders are timestamped to avoid conflicts
- Progress can be monitored via folder creation

### Scaling Options
- **Conservative**: 25 cases per condition (total: 625 cases, ~4 hours)
- **Standard**: 50 cases per condition (total: 1,250 cases, ~8 hours)  
- **Aggressive**: 100 cases per condition (total: 2,500 cases, ~16 hours)

**Recommendation**: Start with 50 cases per condition for first overnight run.
