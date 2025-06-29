# Automated Data Analysis Pipeline

This pipeline automatically analyzes medical diagnosis data files with SAE feature activations, providing comprehensive insights into model performance, demographic effects, and feature patterns.

## 🚀 Quick Start

### Single File Analysis
```bash
# Analyze a single CSV file
python data_analysis_pipeline.py your_data_file.csv

# Skip generating plots
python data_analysis_pipeline.py your_data_file.csv --no-plots

# Skip saving results
python data_analysis_pipeline.py your_data_file.csv --no-save
```

### Batch Analysis
```bash
# Analyze all CSV files in a directory
python batch_analysis.py /path/to/data/directory

# Analyze files with specific pattern
python batch_analysis.py /path/to/data/directory --pattern "*output*.csv"

# Skip summary report
python batch_analysis.py /path/to/data/directory --no-summary
```

## 📊 What Gets Analyzed

### 1. **Accuracy Metrics**
- Overall Top-1 and Top-5 diagnostic accuracy
- Accuracy breakdown by demographics (age, sex)
- Performance comparison across different groups

### 2. **Diagnosis Distribution**
- Most common true diagnoses
- Most frequent predicted diagnoses
- Distribution patterns and imbalances

### 3. **Feature Activation Analysis**
- SAE feature activation statistics (mean, std, sparsity)
- Most active features identification
- Activation pattern analysis

### 4. **Demographic Effects**
- Distribution of demographic groups
- Impact of demographics on accuracy
- Feature activation differences by demographics

### 5. **Clamping Effects** (if present)
- Analysis of feature clamping levels
- Impact of clamping on diagnostic accuracy
- Clamping strategy effectiveness

## 📁 Output Files

For each analyzed file `data.csv`, the pipeline generates:

```
data_analysis/                    # Analysis directory
├── accuracy_by_demographics.png  # Accuracy comparison plots
├── activation_heatmap.png       # Feature activation patterns
└── diagnosis_distribution.png   # Diagnosis frequency charts

data_analysis_results.json       # Detailed analysis results
```

For batch analysis, additional files are created:
```
batch_analysis_summary.csv       # Summary table of all files
batch_analysis_detailed.json     # Complete results for all files
```

## 🔧 Expected Data Format

The pipeline expects CSV/TSV files with these columns:

### Required Columns:
- `diagnosis` - True diagnosis
- `correct_top1` - Whether top prediction is correct (Yes/No or True/False)
- `correct_top5` - Whether correct diagnosis is in top 5 (Yes/No or True/False)

### Optional Columns:
- `dataset_age`, `dataset_sex` - Patient demographics
- `prompt_age`, `prompt_sex` - Prompt demographics
- `activation_*` - SAE feature activations (e.g., activation_0, activation_1, ...)
- `features_clamped`, `clamping_levels` - Clamping information
- `diagnosis_1`, `diagnosis_2`, ... - Top predicted diagnoses

## 📈 Analysis Results Structure

The JSON results file contains:

```json
{
  "accuracy": {
    "top1_accuracy": 0.379,
    "top5_accuracy": 0.414,
    "accuracy_by_dataset_sex": {"male": 0.524, "female": 0.000}
  },
  "diagnosis": {
    "true_diagnosis_counts": {"URTI": 5, "PSVT": 4},
    "predicted_diagnosis_counts": {"Guillain-Barré syndrome": 10}
  },
  "features": {
    "activation_stats": {
      "mean": -0.0096,
      "sparsity": 0.00,
      "most_active_features": {"activation_1": 0.6401}
    }
  },
  "demographics": {
    "dataset_sex_distribution": {"male": 21, "female": 8}
  },
  "metadata": {
    "total_rows": 29,
    "activation_features": 15
  }
}
```

## 🛠️ Customization

### Adding New Analysis Types

To add custom analysis, extend the `MedicalDataAnalyzer` class:

```python
from data_analysis_pipeline import MedicalDataAnalyzer

class CustomAnalyzer(MedicalDataAnalyzer):
    def analyze_custom_metric(self):
        # Your custom analysis here
        return results
    
    def run_full_analysis(self, **kwargs):
        results = super().run_full_analysis(**kwargs)
        results['custom'] = self.analyze_custom_metric()
        return results
```

### Modifying Visualizations

Edit the `generate_visualizations()` method in `data_analysis_pipeline.py` to customize plots.

## 🔍 Troubleshooting

### Common Issues:

1. **CSV Parsing Errors**: The pipeline automatically handles tab-separated files and malformed data
2. **Missing Columns**: Analysis continues with available columns, warnings are shown
3. **Empty Data**: Pipeline gracefully handles empty or invalid files
4. **Memory Issues**: Large files are processed in chunks where possible

### Debug Mode:
```python
# For debugging, create analyzer directly
from data_analysis_pipeline import MedicalDataAnalyzer

analyzer = MedicalDataAnalyzer('your_file.csv')
print(f"Loaded columns: {analyzer.df.columns.tolist()}")
print(f"Data shape: {analyzer.df.shape}")
```

## 📞 Support

The pipeline is designed to be robust and handle various data formats automatically. If you encounter issues:

1. Check the console output for specific error messages
2. Verify your data format matches the expected structure
3. Try the manual debugging approach shown above

## 🎯 Example Usage

```bash
# Analyze the example file
python data_analysis_pipeline.py exampleoutputfordatapipeline.csv

# This will generate:
# - exampleoutputfordatapipeline_analysis_results.json
# - exampleoutputfordatapipeline_analysis/ directory with plots
# - Console output with key findings
```

The pipeline automatically detects your data format and provides comprehensive analysis with minimal setup required!
