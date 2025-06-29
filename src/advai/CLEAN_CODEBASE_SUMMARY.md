# Clean Codebase Summary - 48 Hour Project

## ✅ **Current Active Files (48-Hour Project)**

### **Core Pipeline Files:**
- **`analysis/pipeline.py`** - Main analysis pipeline with timestamped outputs
- **`analysis/clamping_v2.py`** - Advanced directional clamping (handles +/- activations)
- **`analysis/constants_v2.py`** - Enhanced constants with directional feature info
- **`analysis/analyse.py`** - Core analysis functions (updated to use v2)
- **`main.py`** - Main entry point (cleaned, post-hoc analysis disabled)

### **Data Processing:**
- **`data/prompt_builder.py`** - Prompt generation and demographic handling
- **`data/io.py`** - Data input/output utilities
- **`data/example_templates.py`** - Template variations for experiments

### **Model Loading:**
- **`models/loader.py`** - SAE and Gemma model loading utilities

### **Batch Processing:**
- **`48_hour_batch_runner.py`** - Batch runner for multiple experimental conditions
- **`analysis/batch_analysis.py`** - Batch analysis utilities
- **`analysis/data_analysis_pipeline.py`** - Comprehensive analysis pipeline

### **Utilities:**
- **`analysis/summary.py`** - Result summarization
- **`visuals/plots.py`** - Visualization utilities

### **Documentation:**
- **`48_HOUR_PROJECT_PLAN.md`** - Main project plan with overnight run strategy
- **`DUAL_MACHINE_48H_PLAN.md`** - Multi-machine coordination plan
- **`ORGANIZED_OUTPUT_STRUCTURE.md`** - Output folder organization docs

### **Output Directory:**
- **`outputs/`** - Clean, empty directory ready for timestamped run folders

---

## 🗂️ **Moved to `non48/` Folder**

### **Outdated/Replaced Files:**
- **`analysis/clamping.py`** - Old clamping (simple multiplication)
- **`analysis/constants.py`** - Old constants without directional info
- **`analysis/clamping_analysis.py`** - Post-hoc analysis tool
- **`analysis/compare_activations.py`** - Debugging comparison tool

### **Old Results:**
- **`old_outputs/`** - All previous experimental results
- **`results_database.csv`** - Old root-level results file

### **Old Documentation:**
- **`PLAN.TXT`** - Original research plan document

---

## 🎯 **Key Improvements Made:**

1. **✅ Clamping System Upgraded**: Now uses `clamping_v2.py` with proper bidirectional activation handling
2. **✅ Constants Enhanced**: `constants_v2.py` includes directional feature information
3. **✅ Clean Output Structure**: Empty `outputs/` directory ready for organized timestamped runs
4. **✅ Updated Imports**: All active files use v2 versions
5. **✅ Removed Dependencies**: No broken imports or missing files
6. **✅ Preserved History**: All old files safely stored in `non48/` with documentation

---

## 🚀 **Ready for 48-Hour Project:**

### **Core Pipeline Components:**
- ✅ Main analysis pipeline with timestamped output folders
- ✅ Advanced clamping with directional feature handling
- ✅ Batch processing for multiple experimental conditions
- ✅ Comprehensive data analysis and statistical testing
- ✅ Clean output organization with run summaries

### **Experimental Capabilities:**
- ✅ 4 prompt variations (with/without demographics)
- ✅ 24+ clamping conditions (demographics × intensities)
- ✅ Large-scale batch processing (100-500 cases per condition)
- ✅ Automated analysis and statistical testing
- ✅ Organized output with audit trails

### **Next Steps:**
1. **Feature Validation** (Step 8 in plan) - Validate clamping features before overnight runs
2. **Overnight Execution** - Run 28+ conditions × 100-500 cases
3. **Analysis & Interpretation** - Process results and generate findings

The codebase is now clean, organized, and ready for the 48-hour medical bias analysis project!
