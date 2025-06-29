# 48-Hour Medical Bias Detection Project Plan
**Updated**: June 28, 2025 - 9:52 PM  
**Timeline**: 48 hours of focused experimental runs  
**Goal**: Comprehensive medical diagnosis bias analysis using SAE feature clamping

## Project Status: ✅ **PIPELINE TESTED & READY**

### ✅ **COMPLETED TASKS**:
- [x] **Codebase cleanup** - Moved old files to `non48/`, updated imports to v2
- [x] **Pipeline integration** - Fixed interactive mode, added non-interactive support
- [x] **Basic testing** - Baseline run completed successfully (1 case, 2:22 runtime)
- [x] **Clamping testing** - Male clamping test in progress
- [x] **Output structure** - Timestamped folders with run summaries
- [x] **Batch processing** - `48_hour_batch_runner.py` ready
- [x] **Statistical analysis** - Comprehensive analysis pipeline available

### 🔄 **CURRENT STATUS** (10:04 PM):
- **Baseline test**: ✅ Completed (20250628_205814_baseline) - 2:22 runtime
- **Male clamping test**: ✅ Completed (20250628_215116_clamping) - 4:33 runtime
- **Female clamping test**: 🔄 Running (20250628_220521_clamping)
- **Combined demographics test**: 🔄 Running (old+male, 10x intensity)
- **Pipeline**: ✅ Fully validated and ready for batch execution

---

## **EXECUTION STRATEGY**

### **Overnight Runs** (11 PM - 7 AM): 
- **Focus**: Longest, most compute-intensive experimental conditions
- **Target**: All 28 experimental conditions with large case counts (50-100 per condition)
- **Total Cases**: 1,400 to 2,800+ cases
- **Advantage**: Maximize compute during sleep hours

### **Active Hours** (7 AM - 11 PM):
- **Focus**: Analysis, interpretation, documentation
- **Tasks**: Process overnight results, statistical analysis, report generation
- **Advantage**: Human oversight for complex analysis tasks

---

## **EXPERIMENTAL CONDITIONS** (28 Total)

### **Prompt Variations** (4 base conditions):
1. **Baseline**: No demographic info, no clamping
2. **Demographic Prompts**: Include age/sex in prompts, no clamping  
3. **Clamping Only**: No demographic prompts, apply clamping
4. **Both**: Demographic prompts + clamping

### **Clamping Variations** (24 conditions):
- **Demographics**: Male, Female, Old, Young, Old+Male, Old+Female, Young+Male, Young+Female (8 groups)
- **Intensities**: 1x, 5x, 10x (3 levels)
- **Total**: 8 × 3 = 24 clamping conditions

---**Total Experimental Conditions**: 4 prompt + 24 clamping = 28 conditions

---

## IMMEDIATE NEXT STEPS (Tonight: 9:52 PM - 11:00 PM)

### Phase 1: Final Validation & Setup (1 hour)

1. **✅ Complete Clamping Test** (10 min)
   - Verify male clamping test completes successfully
   - Check output format and data quality
   - Compare baseline vs clamped results

2. **🔄 Test Multiple Conditions** (30 min)
   - Test female clamping: `--clamp --clamp-features female --clamp-values 5`
   - Test combined demographics: `--clamp --clamp-features old male --clamp-values 10`
   - Test different intensities: `--clamp-values 1` and `--clamp-values 10`
   - Verify all outputs are properly formatted

3. **📋 Create Batch Command List** (20 min)
   - Generate all 28 command variations for overnight runs
   - Set case counts for overnight runs (50-100 cases per condition)
   - Prepare logging and monitoring setup

### Phase 1B: Dataset Validation (30 min)
4. **Validate existing test dataset** (30 min)
   - ✅ **Dataset confirmed**: `release_test_patients` (134,530 cases)
   - Verify data format and demographic distribution
   - Check columns: AGE, SEX, PATHOLOGY, EVIDENCES, DIFFERENTIAL_DIAGNOSIS
   - Sample cases for initial testing (1-10 cases)
   - **No dataset creation needed** - substantial dataset already available
   - Template 3: "Patient (sex Y) has symptoms..."
   - Template 4: "Patient has symptoms..." (no demographics)

**Note**: Dataset preparation time reduced from 1.5 hours to 30 minutes due to existing comprehensive dataset.

### Phase 1C: Overnight Run Preparation (1 hour)
6. **Prepare for overnight computational runs** (1 hour)
   - Configure batch runner for maximum case counts
   - Set up all 28+ experimental conditions
   - Test with small runs (1-5 cases) to verify pipeline
   - Prepare monitoring and logging systems
   - **Goal**: Ready to start overnight runs by 11 PM

**END DAY 1 - SLEEP: Midnight - 8:00 AM**

---

## DAY 2: June 29 (8:00 AM - Midnight) - 16 Hours

### Phase 2A: **FEATURE VALIDATION & OVERNIGHT RUNS** (10:00 PM - 7:00 AM) - 9 Hours

**CRITICAL: Feature Selection Confirmation (10:00 PM - 11:00 PM) - 1 Hour**
8. **Validate clamping features before large-scale runs** (1 hour)
   - **Load SAE model and examine feature activations**
   - **Test current demographic features**: `MALE_FEATURES`, `FEMALE_FEATURES`, `OLD_FEATURES`, `YOUNG_FEATURES`
   - **Run small test (5-10 cases) with current feature sets**
   - **Analyze activation patterns**: Check for positive/negative values, magnitude ranges
   - **Confirm clamping strategy**: Validate that current multiplication approach works correctly
   - **Document feature selection rationale**: Record which features are most effective
   - **⚠️ STOP POINT**: Do not proceed to overnight runs until features are validated

9. **Start maximum-scale overnight runs** (11:00 PM)
   - **Large-scale runs**: 100-500 cases per condition
   - All 4 baseline conditions (different prompt templates)
   - All 24+ clamping conditions (male, female, young, old × multiple intensities)
   - **Estimated total**: 28+ conditions × 100-500 cases = 2,800-14,000+ total cases
   - **Runtime estimate**: 6-8 hours (perfect for overnight)
   - Set up monitoring scripts and error recovery
   - **Sleep while runs execute**: Midnight - 7:00 AM

### Phase 2B: Morning Analysis (8:00 AM - 12:00 PM) - 4 Hours
10. **Process overnight results** (2 hours)
   - Check run completion status
   - Validate output files and data integrity
   - Run automated analysis pipeline on all results
   - Generate summary statistics

11. **Initial results analysis** (2 hours)
    - Load and examine all results databases
    - Run comprehensive bias analysis
    - Identify significant patterns and effects
    - Generate preliminary findings

### Phase 2C: Deep Analysis and Interpretation (12:00 PM - 6:00 PM) - 6 Hours
12. **Comprehensive bias analysis** (3 hours)
    - Statistical significance testing across all conditions
    - Effect size calculations (Cohen's d)
    - Demographic parity analysis
    - Intersectionality effects
    - Confidence intervals and uncertainty quantification

13. **Results interpretation and documentation** (3 hours)
    - Identify strongest bias effects
    - Document clamping effectiveness
    - Compare baseline vs intervention conditions
    - Prepare findings summary
    - Document methodology and reproducibility

14. **Prepare paper outline and figures** (1 hour)
    - Draft methodology section
    - Prepare figure templates
    - Create results section structure

### Phase 2D: Evening Monitoring (6:00 PM - Midnight) - 6 Hours
15. **Monitor overnight runs** (2 hours)
    - Check progress and logs
    - Handle any errors or restarts
    - Optimize remaining runs

16. **Final analysis and validation** (3 hours)
    - Cross-validate findings across conditions
    - Statistical robustness checks
    - Identify limitations and caveats
    - Prepare final results summary

17. **Final preparation for overnight completion** (1 hour)
    - Set up final analysis scripts to run automatically
    - Prepare comprehensive analysis pipeline
    - Set alarms for monitoring

**END DAY 2 - SLEEP: Midnight - 8:00 AM (Long runs continue)**

---

## DAY 3: June 30 (8:00 AM - 7:30 PM) - 11.5 Hours

### Phase 3A: Results Collection (8:00 AM - 10:00 AM) - 2 Hours
18. **Collect and validate overnight results** (1 hour)
    - Verify all 28+ conditions completed successfully
    - Check data integrity and file completeness
    - Identify any failed runs for re-execution
    - Consolidate all results databases

19. **Run automated analysis pipeline** (1 hour)
    - Execute comprehensive bias detection suite on all results
    - Generate statistical summaries and comparisons
    - Create batch analysis reports

### Phase 3B: Deep Analysis and Interpretation (10:00 AM - 2:00 PM) - 4 Hours
20. **Comprehensive bias analysis** (2 hours)
    - Compare prompt-level vs clamping-induced bias
    - Analyze clamping intensity effects (1x vs 5x vs 10x vs 100x)
    - Identify demographic features with strongest bias
    - Calculate effect sizes and statistical significance

21. **Cross-condition analysis** (2 hours)
    - Compare effectiveness across different clamping strategies
    - Identify optimal intervention parameters
    - Analyze demographic feature interactions
    - Document unexpected findings and patterns

### Phase 3C: Documentation and Conclusions (2:00 PM - 6:00 PM) - 4 Hours
22. **Generate comprehensive findings report** (2 hours)
    - Executive summary of key bias findings
    - Detailed methodology and experimental design
    - Statistical results interpretation
    - Implications for medical AI systems
    - Limitations and future research directions

23. **Create reproducibility documentation** (2 hours)
    - Document all experimental parameters and configurations
    - Create step-by-step replication guide
    - Package all analysis code and scripts
    - Prepare data sharing protocols
    - Write detailed README

### Phase 3D: Final Validation (6:00 PM - 7:30 PM) - 1.5 Hours
24. **Validate key findings** (1 hour)
    - Re-run critical comparisons
    - Verify statistical calculations
    - Check for any anomalies

25. **Final documentation** (30 min)
    - Update project plan with actual results
    - Create final deliverables list
    - Prepare handoff documentation

---

## Key Files to Create (DO NOT MODIFY EXISTING FILES)

### New Files Needed:
1. `src/advai/experiments/experiment_config.py` - Configuration system
2. `src/advai/experiments/batch_runner.py` - Automated experiment runner
3. `src/advai/data/expanded_test_dataset.csv` - Larger test dataset
4. `src/advai/data/prompt_templates.py` - Template variations
5. `src/advai/analysis/comprehensive_batch_analysis.py` - Enhanced analysis
6. `src/advai/analysis/visualization_suite.py` - Visualization tools
7. `src/advai/analysis/bias_detection_suite.py` - Advanced bias metrics
8. `src/advai/analysis/report_generator.py` - Automated reporting
9. `src/advai/experiments/run_all_experiments.sh` - Master run script
10. `src/advai/experiments/monitor_runs.py` - Progress monitoring

### Critical Success Factors:
- **Start overnight runs by noon on Day 2**
- **Use CPU device for consistency** (`--device cpu`)
- **Run with sufficient cases** (50-100 minimum for statistical power)
- **Monitor disk space** (28 conditions × large datasets = significant storage)
- **Backup results frequently**

### Expected Outcomes:
- Quantitative bias measurements across all demographic features
- Comparison of prompt-level vs internal clamping effects
- Optimal clamping intensities for bias detection
- Statistical significance of demographic bias
- Reproducible experimental framework

### Risk Mitigation:
- Start long runs early to account for failures
- Create checkpointing for partial results
- Have backup analysis methods ready
- Prepare manual intervention procedures

This plan maximizes your 48-hour window while respecting your sleep schedule and ensuring the most computationally intensive work runs overnight.
