"""
Automated Data Analysis Pipeline for Medical Diagnosis Data
Analyzes CSV files with patient data, predictions, and SAE activations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Any
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Advanced statistical imports for Anthropic-style analysis
try:
    from scipy import stats
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available - some advanced statistical analyses will be skipped")

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️  statsmodels not available - multiple comparison corrections will be skipped")

class MedicalDataAnalyzer:
    """Comprehensive analyzer for medical diagnosis data with SAE features."""
    
    def __init__(self, csv_path: str):
        """Initialize analyzer with data file."""
        self.csv_path = Path(csv_path)
        self.df = None
        self.analysis_results = {}
        self.load_data()
    
    def load_data(self):
        """Load and validate the CSV data."""
        try:
            # Try different parsing strategies
            try:
                self.df = pd.read_csv(self.csv_path)
            except pd.errors.ParserError:
                print("⚠️  CSV parsing error, trying with tab separator...")
                try:
                    self.df = pd.read_csv(self.csv_path, sep='\t')
                except:
                    print("⚠️  Tab parsing failed, trying with error handling...")
                    self.df = pd.read_csv(self.csv_path, on_bad_lines='skip')
            except:
                print("⚠️  Standard parsing failed, trying with different separator...")
                self.df = pd.read_csv(self.csv_path, sep=None, engine='python')
            
            if self.df is not None and len(self.df) > 0:
                print(f"✅ Loaded {len(self.df)} rows from {self.csv_path.name}")
                print(f"📊 Columns: {len(self.df.columns)} columns")
                # Basic data validation
                self.validate_data()
            else:
                print("❌ No data loaded")
                
        except Exception as e:
            print(f"❌ Error loading {self.csv_path}: {e}")
            print("🔧 Attempting manual parsing...")
            try:
                # Manual parsing as fallback
                with open(self.csv_path, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) > 1:
                    header = lines[0].strip().split(',')
                    data_rows = []
                    
                    for i, line in enumerate(lines[1:], 1):
                        try:
                            row = line.strip().split(',')
                            # Pad or truncate to match header length
                            if len(row) != len(header):
                                if len(row) < len(header):
                                    row.extend([''] * (len(header) - len(row)))
                                else:
                                    row = row[:len(header)]
                            data_rows.append(row)
                        except:
                            print(f"⚠️  Skipping malformed line {i}")
                            continue
                    
                    self.df = pd.DataFrame(data_rows, columns=header)
                    print(f"✅ Manual parsing successful: {len(self.df)} rows")
                    self.validate_data()
                else:
                    print("❌ File appears to be empty or invalid")
                    self.df = pd.DataFrame()
            except Exception as e2:
                print(f"❌ Manual parsing also failed: {e2}")
                self.df = pd.DataFrame()
    
    def validate_data(self):
        """Validate data structure and identify key columns."""
        required_cols = ['diagnosis', 'correct_top1', 'correct_top5']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing required columns: {missing_cols}")
        
        # Identify activation columns
        self.activation_cols = [col for col in self.df.columns if col.startswith('activation_')]
        print(f"🧠 Found {len(self.activation_cols)} activation features")
        
        # Identify demographic columns
        self.demo_cols = []
        for col in ['dataset_age', 'dataset_sex', 'prompt_age', 'prompt_sex']:
            if col in self.df.columns:
                self.demo_cols.append(col)
        print(f"👥 Found demographic columns: {self.demo_cols}")
    
    def analyze_accuracy_metrics(self) -> Dict[str, Any]:
        """Analyze diagnostic accuracy metrics."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for analysis")
            return {}
        
        print("\n=== 🎯 ACCURACY ANALYSIS ===")
        
        results = {}
        
        # Overall accuracy
        if 'correct_top1' in self.df.columns:
            top1_acc = self.df['correct_top1'].value_counts(normalize=True)
            results['top1_accuracy'] = top1_acc.get('Yes', 0) if 'Yes' in top1_acc else top1_acc.get(True, 0)
            print(f"📈 Top-1 Accuracy: {results['top1_accuracy']:.3f}")
        
        if 'correct_top5' in self.df.columns:
            top5_acc = self.df['correct_top5'].value_counts(normalize=True)
            results['top5_accuracy'] = top5_acc.get('Yes', 0) if 'Yes' in top5_acc else top5_acc.get(True, 0)
            print(f"📈 Top-5 Accuracy: {results['top5_accuracy']:.3f}")
        
        # Accuracy by demographics
        for demo_col in self.demo_cols:
            if demo_col in self.df.columns and 'correct_top1' in self.df.columns:
                demo_acc = self.df.groupby(demo_col)['correct_top1'].apply(
                    lambda x: (x == 'Yes').mean() if x.dtype == 'object' else x.mean()
                )
                results[f'accuracy_by_{demo_col}'] = demo_acc.to_dict()
                print(f"📊 Accuracy by {demo_col}:")
                for group, acc in demo_acc.items():
                    print(f"   {group}: {acc:.3f}")
        
        return results
    
    def analyze_diagnosis_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of diagnoses."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for analysis")
            return {}
        
        print("\n=== 🏥 DIAGNOSIS DISTRIBUTION ===")
        
        results = {}
        
        if 'diagnosis' in self.df.columns:
            # True diagnosis distribution
            true_diag = self.df['diagnosis'].value_counts()
            results['true_diagnosis_counts'] = true_diag.to_dict()
            print(f"📋 Most common diagnoses:")
            for diag, count in true_diag.head(10).items():
                print(f"   {diag}: {count} ({count/len(self.df):.2%})")
        
        # Predicted diagnosis distribution
        pred_cols = [col for col in self.df.columns if col.startswith('diagnosis_')]
        if pred_cols:
            pred_diag = self.df[pred_cols[0]].value_counts()
            results['predicted_diagnosis_counts'] = pred_diag.to_dict()
            print(f"🔮 Top predicted diagnoses:")
            for diag, count in pred_diag.head(5).items():
                print(f"   {diag}: {count}")
        
        return results
    
    def analyze_feature_activations(self) -> Dict[str, Any]:
        """Analyze SAE feature activation patterns."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for analysis")
            return {}
        
        print("\n=== 🧠 FEATURE ACTIVATION ANALYSIS ===")
        
        results = {}
        
        if not self.activation_cols:
            print("⚠️  No activation columns found")
            return results
        
        # Basic activation statistics
        activation_data = self.df[self.activation_cols].astype(float)
        
        results['activation_stats'] = {
            'mean': activation_data.mean().mean(),
            'std': activation_data.std().mean(),
            'min': activation_data.min().min(),
            'max': activation_data.max().max(),
            'sparsity': (activation_data == 0).mean().mean()
        }
        
        print(f"📊 Activation Statistics:")
        print(f"   Mean activation: {results['activation_stats']['mean']:.4f}")
        print(f"   Std activation: {results['activation_stats']['std']:.4f}")
        print(f"   Sparsity (% zeros): {results['activation_stats']['sparsity']:.2%}")
        
        # Most active features
        feature_activity = activation_data.abs().mean().sort_values(ascending=False)
        results['most_active_features'] = feature_activity.head(10).to_dict()
        
        print(f"🔥 Most active features:")
        for feat, activity in feature_activity.head(5).items():
            print(f"   {feat}: {activity:.4f}")
        
        return results
    
    def analyze_demographic_effects(self) -> Dict[str, Any]:
        """Analyze how demographics affect predictions and activations."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for analysis")
            return {}
        
        print("\n=== 👥 DEMOGRAPHIC EFFECTS ANALYSIS ===")
        
        results = {}
        
        for demo_col in self.demo_cols:
            if demo_col not in self.df.columns:
                continue
                
            print(f"\n📈 Analysis for {demo_col}:")
            
            # Demographic distribution
            demo_dist = self.df[demo_col].value_counts()
            results[f'{demo_col}_distribution'] = demo_dist.to_dict()
            
            print(f"   Distribution: {dict(demo_dist)}")
            
            # Effect on accuracy
            if 'correct_top1' in self.df.columns:
                demo_acc = self.df.groupby(demo_col)['correct_top1'].apply(
                    lambda x: (x == 'Yes').mean() if x.dtype == 'object' else x.mean()
                )
                results[f'{demo_col}_accuracy_effect'] = demo_acc.to_dict()
            
            # Effect on feature activations (sample of features)
            if self.activation_cols:
                sample_features = self.activation_cols[:5]  # Analyze first 5 features
                for feat in sample_features:
                    feat_by_demo = self.df.groupby(demo_col)[feat].mean()
                    if demo_col not in results:
                        results[demo_col] = {}
                    results[demo_col][f'{feat}_mean'] = feat_by_demo.to_dict()
        
        return results
    
    def analyze_clamping_effects(self) -> Dict[str, Any]:
        """Analyze effects of feature clamping if present."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for analysis")
            return {}
        
        print("\n=== 🔧 CLAMPING EFFECTS ANALYSIS ===")
        
        results = {}
        
        # Check for clamping-related columns
        clamping_cols = [col for col in self.df.columns if 'clamp' in col.lower()]
        
        if not clamping_cols:
            print("⚠️  No clamping columns detected")
            return results
        
        print(f"🔧 Found clamping columns: {clamping_cols}")
        
        for clamp_col in clamping_cols:
            if clamp_col in self.df.columns:
                clamp_dist = self.df[clamp_col].value_counts()
                results[f'{clamp_col}_distribution'] = clamp_dist.to_dict()
                
                # Effect on accuracy
                if 'correct_top1' in self.df.columns:
                    clamp_acc = self.df.groupby(clamp_col)['correct_top1'].apply(
                        lambda x: (x == 'Yes').mean() if x.dtype == 'object' else x.mean()
                    )
                    results[f'{clamp_col}_accuracy_effect'] = clamp_acc.to_dict()
                    
                    print(f"📊 {clamp_col} effect on accuracy:")
                    for level, acc in clamp_acc.items():
                        print(f"   {level}: {acc:.3f}")
        
        return results
    
    def analyze_bias_metrics(self) -> Dict[str, Any]:
        """Analyze bias using Anthropic's BBQ-style framework."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for bias analysis")
            return {}
        
        print("\n=== 🎯 ANTHROPIC-STYLE BIAS METRICS ANALYSIS ===")
        
        results = {}
        
        # BBQ-style bias scoring (-1 to +1 scale)
        for demo_col in self.demo_cols:
            if demo_col not in self.df.columns or 'correct_top1' not in self.df.columns:
                continue
                
            print(f"📊 Analyzing bias for {demo_col}...")
            
            # Convert correct_top1 to numeric for analysis
            correct_numeric = self.df['correct_top1'].apply(
                lambda x: 1 if x == 'Yes' or x == True or x == 1 else 0
            )
            
            # Add to dataframe temporarily for groupby operations
            temp_col = '_temp_correct_numeric'
            self.df[temp_col] = correct_numeric
            
            # Statistical significance testing between groups
            groups = self.df.groupby(demo_col)[temp_col].agg(['mean', 'count', 'std'])
            
            if len(groups) >= 2 and SCIPY_AVAILABLE:
                # Effect size calculation (Cohen's d) for first two groups
                group_names = list(groups.index)
                group1_data = correct_numeric[self.df[demo_col] == group_names[0]]
                group2_data = correct_numeric[self.df[demo_col] == group_names[1]]
                
                if len(group1_data) > 1 and len(group2_data) > 1:
                    # Cohen's d calculation
                    pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                        (len(group2_data) - 1) * group2_data.var()) / 
                                       (len(group1_data) + len(group2_data) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = abs(group1_data.mean() - group2_data.mean()) / pooled_std
                        results[f'{demo_col}_effect_size'] = cohens_d
                        
                        # Interpret effect size
                        if cohens_d < 0.2:
                            effect_interpretation = "negligible"
                        elif cohens_d < 0.5:
                            effect_interpretation = "small"
                        elif cohens_d < 0.8:
                            effect_interpretation = "medium"
                        else:
                            effect_interpretation = "large"
                        
                        results[f'{demo_col}_effect_interpretation'] = effect_interpretation
                        print(f"   Cohen's d: {cohens_d:.3f} ({effect_interpretation})")
                    
                    # T-test for statistical significance
                    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                    results[f'{demo_col}_t_test'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                    print(f"   T-test p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")
            
            # Multiple comparison correction for more than 2 groups
            if len(groups) > 2 and STATSMODELS_AVAILABLE:
                # Pairwise comparisons
                p_values = []
                comparisons = []
                
                for i, group1 in enumerate(group_names):
                    for j, group2 in enumerate(group_names[i+1:], i+1):
                        group1_data = correct_numeric[self.df[demo_col] == group1]
                        group2_data = correct_numeric[self.df[demo_col] == group2]
                        
                        if len(group1_data) > 1 and len(group2_data) > 1 and SCIPY_AVAILABLE:
                            _, p_val = stats.ttest_ind(group1_data, group2_data)
                            p_values.append(p_val)
                            comparisons.append(f"{group1}_vs_{group2}")
                
                if p_values:
                    # Bonferroni correction
                    rejected, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
                    
                    results[f'{demo_col}_multiple_comparisons'] = {
                        'comparisons': comparisons,
                        'raw_p_values': p_values,
                        'corrected_p_values': p_corrected.tolist(),
                        'significant_after_correction': rejected.tolist()
                    }
                    
                    significant_count = sum(rejected)
                    print(f"   Multiple comparisons: {significant_count}/{len(p_values)} significant after Bonferroni correction")
            
            # Bias score calculation (BBQ-style -1 to +1 scale)
            group_means = groups['mean']
            if len(group_means) >= 2:
                # Calculate bias as deviation from equal performance
                expected_performance = group_means.mean()
                bias_scores = {}
                
                for group, performance in group_means.items():
                    # Normalize to -1 to +1 scale
                    bias_score = (performance - expected_performance) / expected_performance if expected_performance > 0 else 0
                    bias_scores[str(group)] = bias_score
                
                results[f'{demo_col}_bias_scores'] = bias_scores
                
                # Overall bias magnitude
                overall_bias = np.std(list(bias_scores.values()))
                results[f'{demo_col}_overall_bias_magnitude'] = overall_bias
                print(f"   Overall bias magnitude: {overall_bias:.3f}")
        
        # Clean up temporary column
        if temp_col in self.df.columns:
            self.df.drop(columns=[temp_col], inplace=True)
        
        return results
    
    def analyze_feature_correlations(self) -> Dict[str, Any]:
        """Analyze feature correlations using Anthropic's methodology."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for correlation analysis")
            return {}
        
        print("\n=== 🔗 FEATURE CORRELATION ANALYSIS ===")
        
        results = {}
        
        if not self.activation_cols:
            print("⚠️  No activation columns found for correlation analysis")
            return results
        
        activation_data = self.df[self.activation_cols].astype(float)
        
        # Cross-feature correlation matrix
        print("📊 Computing correlation matrix...")
        corr_matrix = activation_data.corr()
        
        # Anthropic's threshold: r ≤ 0.3 for "weakly correlated"
        # Count correlations excluding diagonal (self-correlations)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle_corrs = corr_matrix.where(mask)
        
        weak_correlations = (upper_triangle_corrs.abs() <= 0.3).sum().sum()
        total_pairs = len(self.activation_cols) * (len(self.activation_cols) - 1) / 2
        
        weak_correlation_percentage = weak_correlations / total_pairs if total_pairs > 0 else 0
        results['weak_correlation_percentage'] = weak_correlation_percentage
        results['total_feature_pairs'] = int(total_pairs)
        results['weak_correlation_count'] = int(weak_correlations)
        
        print(f"🔗 Weak correlations (≤0.3): {weak_correlations}/{int(total_pairs)} ({weak_correlation_percentage:.1%})")
        
        # Strong correlation analysis
        strong_correlations = (upper_triangle_corrs.abs() > 0.7).sum().sum()
        results['strong_correlation_count'] = int(strong_correlations)
        results['strong_correlation_percentage'] = strong_correlations / total_pairs if total_pairs > 0 else 0
        
        print(f"🔗 Strong correlations (>0.7): {strong_correlations}/{int(total_pairs)} ({strong_correlations/total_pairs:.1%})")
        
        # Dead feature detection (features that never activate)
        dead_features = (activation_data == 0).all(axis=0)
        dead_feature_count = dead_features.sum()
        results['dead_features_count'] = int(dead_feature_count)
        results['dead_features_percentage'] = dead_feature_count / len(self.activation_cols)
        
        if dead_feature_count > 0:
            results['dead_feature_names'] = dead_features[dead_features].index.tolist()
            print(f"💀 Dead features (never activate): {dead_feature_count}/{len(self.activation_cols)} ({dead_feature_count/len(self.activation_cols):.1%})")
        
        # Feature activation frequency vs interpretability
        activation_freq = (activation_data > 0).mean()
        results['activation_frequency_stats'] = {
            'mean': activation_freq.mean(),
            'std': activation_freq.std(),
            'min': activation_freq.min(),
            'max': activation_freq.max(),
            'distribution': activation_freq.describe().to_dict()
        }
        
        print(f"📈 Activation frequency: mean={activation_freq.mean():.3f}, std={activation_freq.std():.3f}")
        
        # Most and least active features
        most_active = activation_freq.nlargest(5)
        least_active = activation_freq.nsmallest(5)
        
        results['most_active_features'] = most_active.to_dict()
        results['least_active_features'] = least_active.to_dict()
        
        print(f"🔥 Most active features: {list(most_active.index[:3])}")
        print(f"❄️  Least active features: {list(least_active.index[:3])}")
        
        # Kurtosis analysis for basis privilege detection
        if SCIPY_AVAILABLE:
            feature_kurtosis = activation_data.apply(lambda x: stats.kurtosis(x, nan_policy='omit'))
            results['kurtosis_stats'] = {
                'mean': feature_kurtosis.mean(),
                'std': feature_kurtosis.std(),
                'high_kurtosis_features': feature_kurtosis[feature_kurtosis > 3].index.tolist()
            }
            
            high_kurtosis_count = len(results['kurtosis_stats']['high_kurtosis_features'])
            print(f"📊 High kurtosis features (>3): {high_kurtosis_count}/{len(self.activation_cols)}")
        
        return results
    
    def analyze_steering_effects(self) -> Dict[str, Any]:
        """Analyze feature steering effects using Anthropic's framework."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for steering analysis")
            return {}
        
        print("\n=== 🎛️ FEATURE STEERING EFFECTS ANALYSIS ===")
        
        results = {}
        
        # Look for steering/clamping columns
        steering_cols = [col for col in self.df.columns if any(keyword in col.lower() 
                        for keyword in ['clamp', 'steer', 'intervention', 'manipulate', 'features_clamped'])]
        
        if not steering_cols:
            print("⚠️  No steering/clamping columns detected")
            return results
        
        print(f"🎛️ Found steering columns: {steering_cols}")
        
        for steer_col in steering_cols:
            print(f"\n📊 Analyzing {steer_col}...")
            
            # Get unique conditions (excluding NaN)
            conditions = self.df[steer_col].dropna().unique()
            condition_counts = self.df[steer_col].value_counts(dropna=True)
            
            print(f"   Conditions: {list(conditions)}")
            
            if len(conditions) < 2:
                print(f"   ⚠️  Insufficient conditions for analysis (need ≥2, found {len(conditions)})")
                continue
            
            # Identify baseline condition (look for common baseline indicators)
            baseline_indicators = ['baseline', 'control', 'none', 'no', '0', 'original', 'unmodified']
            baseline_condition = None
            
            for condition in conditions:
                if str(condition).lower() in baseline_indicators:
                    baseline_condition = condition
                    break
            
            # If no explicit baseline, use the most common condition
            if baseline_condition is None:
                baseline_condition = condition_counts.index[0]
                print(f"   Using most common condition as baseline: {baseline_condition}")
            
            baseline_data = self.df[self.df[steer_col] == baseline_condition]
            intervention_data = self.df[self.df[steer_col] != baseline_condition]
            
            if len(baseline_data) == 0 or len(intervention_data) == 0:
                print(f"   ⚠️  Insufficient data for comparison")
                continue
            
            # Convert correct_top1 to numeric
            baseline_correct = baseline_data['correct_top1'].apply(
                lambda x: 1 if x == 'Yes' or x == True or x == 1 else 0
            )
            intervention_correct = intervention_data['correct_top1'].apply(
                lambda x: 1 if x == 'Yes' or x == True or x == 1 else 0
            )
            
            # Effect size on accuracy
            baseline_acc = baseline_correct.mean()
            intervention_acc = intervention_correct.mean()
            
            accuracy_change = {
                'baseline_accuracy': baseline_acc,
                'intervention_accuracy': intervention_acc,
                'absolute_change': intervention_acc - baseline_acc,
                'relative_change': (intervention_acc - baseline_acc) / baseline_acc if baseline_acc > 0 else 0,
                'baseline_n': len(baseline_data),
                'intervention_n': len(intervention_data)
            }
            
            # Statistical significance of the change
            if SCIPY_AVAILABLE and len(baseline_correct) > 1 and len(intervention_correct) > 1:
                t_stat, p_value = stats.ttest_ind(baseline_correct, intervention_correct)
                accuracy_change['t_statistic'] = t_stat
                accuracy_change['p_value'] = p_value
                accuracy_change['significant'] = p_value < 0.05
                
                print(f"   Accuracy change: {baseline_acc:.3f} → {intervention_acc:.3f} ({accuracy_change['absolute_change']:+.3f})")
                print(f"   Statistical significance: p={p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")
            
            results[f'{steer_col}_accuracy_effect'] = accuracy_change
            
            # Off-target effects monitoring across demographics
            for demo_col in self.demo_cols:
                if demo_col not in self.df.columns:
                    continue
                
                print(f"   📊 Checking off-target effects for {demo_col}...")
                
                # Baseline demographic performance
                baseline_demo_correct = baseline_data.groupby(demo_col)['correct_top1'].apply(
                    lambda x: (x == 'Yes').mean() if x.dtype == 'object' else x.mean()
                )
                
                # Intervention demographic performance
                intervention_demo_correct = intervention_data.groupby(demo_col)['correct_top1'].apply(
                    lambda x: (x == 'Yes').mean() if x.dtype == 'object' else x.mean()
                )
                
                # Calculate bias change across demographics
                bias_change = {}
                for group in baseline_demo_correct.index:
                    if group in intervention_demo_correct.index:
                        baseline_perf = baseline_demo_correct[group]
                        intervention_perf = intervention_demo_correct[group]
                        absolute_change = intervention_perf - baseline_perf
                        relative_change = absolute_change / baseline_perf if baseline_perf > 0 else 0
                        
                        bias_change[str(group)] = {
                            'baseline': baseline_perf,
                            'intervention': intervention_perf,
                            'absolute_change': absolute_change,
                            'relative_change': relative_change
                        }
                
                results[f'{steer_col}_{demo_col}_bias_change'] = bias_change
                
                # Calculate overall bias change (variance across groups)
                if len(bias_change) >= 2:
                    baseline_variance = np.var(list(baseline_demo_correct.values))
                    intervention_variance = np.var(list(intervention_demo_correct.values))
                    bias_variance_change = intervention_variance - baseline_variance
                    
                    results[f'{steer_col}_{demo_col}_bias_variance_change'] = {
                        'baseline_variance': baseline_variance,
                        'intervention_variance': intervention_variance,
                        'variance_change': bias_variance_change,
                        'bias_reduction': bias_variance_change < 0
                    }
                    
                    print(f"     Bias variance change: {bias_variance_change:+.4f} ({'reduced' if bias_variance_change < 0 else 'increased'})")
            
            # Feature activation changes (if activation columns available)
            if self.activation_cols:
                print(f"   🧠 Analyzing feature activation changes...")
                
                baseline_activations = baseline_data[self.activation_cols].astype(float).mean()
                intervention_activations = intervention_data[self.activation_cols].astype(float).mean()
                
                activation_changes = intervention_activations - baseline_activations
                
                # Top changed features
                top_increased = activation_changes.nlargest(5)
                top_decreased = activation_changes.nsmallest(5)
                
                results[f'{steer_col}_activation_changes'] = {
                    'top_increased_features': top_increased.to_dict(),
                    'top_decreased_features': top_decreased.to_dict(),
                    'mean_absolute_change': activation_changes.abs().mean(),
                    'total_features_changed': (activation_changes.abs() > 0.01).sum()
                }
                
                print(f"     Features with largest increases: {list(top_increased.index[:3])}")
                print(f"     Features with largest decreases: {list(top_decreased.index[:3])}")
        
        return results
    
    def bootstrap_analysis(self, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Perform bootstrap analysis for uncertainty quantification."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for bootstrap analysis")
            return {}
        
        print(f"\n=== 🎲 BOOTSTRAP UNCERTAINTY ANALYSIS (n={n_bootstrap}) ===")
        
        results = {}
        
        # Convert correct_top1 to numeric
        correct_numeric = self.df['correct_top1'].apply(
            lambda x: 1 if x == 'Yes' or x == True or x == 1 else 0
        )
        
        # Overall accuracy bootstrap
        print("📊 Bootstrapping overall accuracy...")
        bootstrap_accuracies = []
        
        for i in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(correct_numeric, size=len(correct_numeric), replace=True)
            bootstrap_accuracies.append(bootstrap_sample.mean())
        
        bootstrap_accuracies = np.array(bootstrap_accuracies)
        
        results['overall_accuracy_bootstrap'] = {
            'mean': bootstrap_accuracies.mean(),
            'std': bootstrap_accuracies.std(),
            'ci_95_lower': np.percentile(bootstrap_accuracies, 2.5),
            'ci_95_upper': np.percentile(bootstrap_accuracies, 97.5),
            'ci_99_lower': np.percentile(bootstrap_accuracies, 0.5),
            'ci_99_upper': np.percentile(bootstrap_accuracies, 99.5)
        }
        
        print(f"   Overall accuracy: {bootstrap_accuracies.mean():.3f} ± {bootstrap_accuracies.std():.3f}")
        print(f"   95% CI: [{results['overall_accuracy_bootstrap']['ci_95_lower']:.3f}, {results['overall_accuracy_bootstrap']['ci_95_upper']:.3f}]")
        
        # Demographic group bootstrap
        for demo_col in self.demo_cols:
            if demo_col not in self.df.columns:
                continue
                
            print(f"📊 Bootstrapping {demo_col} accuracy...")
            
            demo_groups = self.df[demo_col].unique()
            demo_bootstrap_results = {}
            
            for group in demo_groups:
                group_data = correct_numeric[self.df[demo_col] == group]
                
                if len(group_data) < 10:  # Skip groups with too few samples
                    continue
                
                group_bootstrap_accuracies = []
                
                for i in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(group_data, size=len(group_data), replace=True)
                    group_bootstrap_accuracies.append(bootstrap_sample.mean())
                
                group_bootstrap_accuracies = np.array(group_bootstrap_accuracies)
                
                demo_bootstrap_results[str(group)] = {
                    'mean': group_bootstrap_accuracies.mean(),
                    'std': group_bootstrap_accuracies.std(),
                    'ci_95_lower': np.percentile(group_bootstrap_accuracies, 2.5),
                    'ci_95_upper': np.percentile(group_bootstrap_accuracies, 97.5),
                    'sample_size': len(group_data)
                }
                
                print(f"   {group}: {group_bootstrap_accuracies.mean():.3f} ± {group_bootstrap_accuracies.std():.3f}")
            
            results[f'{demo_col}_bootstrap'] = demo_bootstrap_results
            
            # Bootstrap bias quantification between groups
            if len(demo_bootstrap_results) >= 2:
                group_names = list(demo_bootstrap_results.keys())
                bias_bootstrap = []
                
                for i in range(n_bootstrap):
                    # Sample group accuracies from their bootstrap distributions
                    group_accs = []
                    for group_name in group_names:
                        group_data = correct_numeric[self.df[demo_col] == group_name]
                        if len(group_data) >= 10:
                            bootstrap_sample = np.random.choice(group_data, size=len(group_data), replace=True)
                            group_accs.append(bootstrap_sample.mean())
                    
                    if len(group_accs) >= 2:
                        # Calculate bias as standard deviation of group accuracies
                        bias_bootstrap.append(np.std(group_accs))
                
                if bias_bootstrap:
                    bias_bootstrap = np.array(bias_bootstrap)
                    results[f'{demo_col}_bias_bootstrap'] = {
                        'mean_bias': bias_bootstrap.mean(),
                        'bias_std': bias_bootstrap.std(),
                        'bias_ci_95_lower': np.percentile(bias_bootstrap, 2.5),
                        'bias_ci_95_upper': np.percentile(bias_bootstrap, 97.5)
                    }
                    
                    print(f"   Bias magnitude: {bias_bootstrap.mean():.3f} ± {bias_bootstrap.std():.3f}")
        
        # Feature activation bootstrap (if available)
        if self.activation_cols:
            print("🧠 Bootstrapping feature activations...")
            
            # Sample a subset of features for computational efficiency
            sample_features = self.activation_cols[:min(20, len(self.activation_cols))]
            activation_data = self.df[sample_features].astype(float)
            
            feature_bootstrap_results = {}
            
            for feature in sample_features:
                feature_data = activation_data[feature].values
                feature_bootstrap_means = []
                
                for i in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(feature_data, size=len(feature_data), replace=True)
                    feature_bootstrap_means.append(bootstrap_sample.mean())
                
                feature_bootstrap_means = np.array(feature_bootstrap_means)
                
                feature_bootstrap_results[feature] = {
                    'mean': feature_bootstrap_means.mean(),
                    'std': feature_bootstrap_means.std(),
                    'ci_95_lower': np.percentile(feature_bootstrap_means, 2.5),
                    'ci_95_upper': np.percentile(feature_bootstrap_means, 97.5)
                }
            
            results['feature_activation_bootstrap'] = feature_bootstrap_results
            print(f"   Bootstrapped {len(sample_features)} features")
        
        return results
    
    def analyze_intersectionality(self) -> Dict[str, Any]:
        """Analyze intersectional bias effects across demographic combinations."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for intersectionality analysis")
            return {}
        
        print("\n=== ⚡ INTERSECTIONALITY ANALYSIS ===")
        
        results = {}
        
        # Convert correct_top1 to numeric for analysis
        correct_numeric = self.df['correct_top1'].apply(
            lambda x: 1 if x == 'Yes' or x == True or x == 1 else 0
        )
        
        # Add to dataframe temporarily for groupby operations
        temp_col = '_temp_correct_numeric'
        self.df[temp_col] = correct_numeric
        
        # Find available demographic columns
        available_demo_cols = [col for col in self.demo_cols if col in self.df.columns]
        
        if len(available_demo_cols) < 2:
            print("⚠️  Need at least 2 demographic columns for intersectionality analysis")
            return results
        
        print(f"📊 Analyzing intersections of: {available_demo_cols}")
        
        # Pairwise intersectionality analysis
        for i, demo1 in enumerate(available_demo_cols):
            for demo2 in available_demo_cols[i+1:]:
                print(f"\n🔍 Analyzing {demo1} × {demo2} intersection...")
                
                # Create intersection groups
                intersection_key = f"{demo1}_x_{demo2}"
                self.df[intersection_key] = self.df[demo1].astype(str) + "_x_" + self.df[demo2].astype(str)
                
                # Analyze performance across intersectional groups
                intersection_groups = self.df.groupby(intersection_key)[temp_col].agg(['mean', 'count'])
                
                # Filter out groups with too few samples
                min_samples = 5
                valid_groups = intersection_groups[intersection_groups['count'] >= min_samples]
                
                if len(valid_groups) < 2:
                    print(f"   ⚠️  Insufficient data for {intersection_key} analysis")
                    continue
                
                # Calculate intersectional bias metrics
                group_accuracies = valid_groups['mean']
                
                intersectional_results = {
                    'group_accuracies': group_accuracies.to_dict(),
                    'group_counts': valid_groups['count'].to_dict(),
                    'overall_variance': group_accuracies.var(),
                    'accuracy_range': group_accuracies.max() - group_accuracies.min(),
                    'best_performing_group': group_accuracies.idxmax(),
                    'worst_performing_group': group_accuracies.idxmin(),
                    'best_accuracy': group_accuracies.max(),
                    'worst_accuracy': group_accuracies.min()
                }
                
                print(f"   Accuracy range: {intersectional_results['accuracy_range']:.3f}")
                print(f"   Best: {intersectional_results['best_performing_group']} ({intersectional_results['best_accuracy']:.3f})")
                print(f"   Worst: {intersectional_results['worst_performing_group']} ({intersectional_results['worst_accuracy']:.3f})")
                
                # Statistical significance testing across intersectional groups
                if SCIPY_AVAILABLE and len(valid_groups) >= 2:
                    # ANOVA test for multiple groups
                    group_data = []
                    group_names = []
                    
                    for group_name in valid_groups.index:
                        group_correct = correct_numeric[self.df[intersection_key] == group_name]
                        if len(group_correct) >= min_samples:
                            group_data.append(group_correct.values)
                            group_names.append(group_name)
                    
                    if len(group_data) >= 2:
                        f_stat, p_value = stats.f_oneway(*group_data)
                        intersectional_results['anova'] = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                        
                        print(f"   ANOVA p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")
                
                # Compare intersectional effects to individual demographic effects
                demo1_groups = self.df.groupby(demo1)[temp_col].mean()
                demo2_groups = self.df.groupby(demo2)[temp_col].mean()
                
                individual_variance = (demo1_groups.var() + demo2_groups.var()) / 2
                intersectional_variance = group_accuracies.var()
                
                intersectional_results['variance_comparison'] = {
                    'individual_avg_variance': individual_variance,
                    'intersectional_variance': intersectional_variance,
                    'amplification_factor': intersectional_variance / individual_variance if individual_variance > 0 else 0,
                    'intersectional_amplification': intersectional_variance > individual_variance
                }
                
                amplification = intersectional_results['variance_comparison']['amplification_factor']
                print(f"   Bias amplification factor: {amplification:.2f}x ({'amplified' if amplification > 1 else 'reduced'})")
                
                results[intersection_key] = intersectional_results
                
                # Clean up temporary column
                self.df.drop(columns=[intersection_key], inplace=True)
        
        # Triple intersectionality (if 3+ demographics available)
        if len(available_demo_cols) >= 3:
            print(f"\n🔍 Analyzing triple intersection: {available_demo_cols[:3]}...")
            
            demo1, demo2, demo3 = available_demo_cols[:3]
            triple_key = f"{demo1}_x_{demo2}_x_{demo3}"
            
            self.df[triple_key] = (self.df[demo1].astype(str) + "_x_" + 
                                  self.df[demo2].astype(str) + "_x_" + 
                                  self.df[demo3].astype(str))
            
            triple_groups = self.df.groupby(triple_key)[temp_col].agg(['mean', 'count'])
            valid_triple_groups = triple_groups[triple_groups['count'] >= 3]  # Lower threshold for triple
            
            if len(valid_triple_groups) >= 2:
                triple_accuracies = valid_triple_groups['mean']
                
                triple_results = {
                    'group_accuracies': triple_accuracies.to_dict(),
                    'group_counts': valid_triple_groups['count'].to_dict(),
                    'overall_variance': triple_accuracies.var(),
                    'accuracy_range': triple_accuracies.max() - triple_accuracies.min(),
                    'total_groups': len(valid_triple_groups)
                }
                
                print(f"   Triple intersection groups: {len(valid_triple_groups)}")
                print(f"   Accuracy range: {triple_results['accuracy_range']:.3f}")
                print(f"   Variance: {triple_results['overall_variance']:.4f}")
                
                results['triple_intersection'] = triple_results
            
            # Clean up temporary column
            self.df.drop(columns=[triple_key], inplace=True)
        
        # Intersectionality summary metrics
        if results:
            all_variances = [result['overall_variance'] for result in results.values() 
                           if isinstance(result, dict) and 'overall_variance' in result]
            all_ranges = [result['accuracy_range'] for result in results.values() 
                         if isinstance(result, dict) and 'accuracy_range' in result]
            
            if all_variances and all_ranges:
                results['summary'] = {
                    'mean_intersectional_variance': np.mean(all_variances),
                    'max_intersectional_variance': np.max(all_variances),
                    'mean_accuracy_range': np.mean(all_ranges),
                    'max_accuracy_range': np.max(all_ranges),
                    'total_intersections_analyzed': len(all_variances)
                }
                
                print(f"\n📊 Intersectionality Summary:")
                print(f"   Mean variance: {results['summary']['mean_intersectional_variance']:.4f}")
                print(f"   Max accuracy range: {results['summary']['max_accuracy_range']:.3f}")
        
        # Clean up temporary column
        if temp_col in self.df.columns:
            self.df.drop(columns=[temp_col], inplace=True)
        
        return results
    
    def analyze_scaling_patterns(self) -> Dict[str, Any]:
        """Analyze scaling patterns in feature activations using power law analysis."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for scaling analysis")
            return {}
        
        print("\n=== 📈 SCALING PATTERNS ANALYSIS ===")
        
        results = {}
        
        if not self.activation_cols:
            print("⚠️  No activation columns found for scaling analysis")
            return results
        
        activation_data = self.df[self.activation_cols].astype(float)
        
        # Power law analysis of feature activation distributions
        print("📊 Analyzing power law distributions...")
        
        power_law_results = {}
        
        for feature in self.activation_cols[:min(50, len(self.activation_cols))]:  # Limit for efficiency
            feature_data = activation_data[feature].values
            
            # Remove zeros and negative values for power law analysis
            positive_data = feature_data[feature_data > 0]
            
            if len(positive_data) < 10:
                continue
            
            # Fit power law using log-log regression
            if SCIPY_AVAILABLE:
                # Sort data in descending order
                sorted_data = np.sort(positive_data)[::-1]
                ranks = np.arange(1, len(sorted_data) + 1)
                
                # Log-log regression
                log_ranks = np.log(ranks)
                log_values = np.log(sorted_data)
                
                # Remove infinite values
                valid_mask = np.isfinite(log_ranks) & np.isfinite(log_values)
                if np.sum(valid_mask) < 5:
                    continue
                
                log_ranks_clean = log_ranks[valid_mask]
                log_values_clean = log_values[valid_mask]
                
                # Linear regression in log space
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks_clean, log_values_clean)
                
                power_law_results[feature] = {
                    'power_law_exponent': -slope,  # Negative because of rank-frequency relationship
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'intercept': intercept,
                    'std_error': std_err,
                    'n_points': len(positive_data),
                    'good_fit': r_value**2 > 0.8 and p_value < 0.05
                }
        
        if power_law_results:
            # Summary statistics
            exponents = [result['power_law_exponent'] for result in power_law_results.values()]
            r_squared_values = [result['r_squared'] for result in power_law_results.values()]
            good_fits = [result['good_fit'] for result in power_law_results.values()]
            
            results['power_law_analysis'] = {
                'features_analyzed': len(power_law_results),
                'mean_exponent': np.mean(exponents),
                'std_exponent': np.std(exponents),
                'median_exponent': np.median(exponents),
                'mean_r_squared': np.mean(r_squared_values),
                'good_fits_count': sum(good_fits),
                'good_fits_percentage': sum(good_fits) / len(good_fits),
                'feature_results': power_law_results
            }
            
            print(f"   Features analyzed: {len(power_law_results)}")
            print(f"   Mean power law exponent: {np.mean(exponents):.3f} ± {np.std(exponents):.3f}")
            print(f"   Good fits (R² > 0.8): {sum(good_fits)}/{len(good_fits)} ({sum(good_fits)/len(good_fits):.1%})")
        
        # Activation magnitude scaling
        print("📊 Analyzing activation magnitude scaling...")
        
        # Calculate activation statistics across features
        feature_means = activation_data.mean()
        feature_stds = activation_data.std()
        feature_maxs = activation_data.max()
        
        # Relationship between mean and variance (scaling patterns)
        if SCIPY_AVAILABLE:
            # Mean-variance relationship
            valid_mask = (feature_means > 0) & (feature_stds > 0)
            if np.sum(valid_mask) > 5:
                log_means = np.log(feature_means[valid_mask])
                log_vars = np.log(feature_stds[valid_mask]**2)
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_means, log_vars)
                
                results['mean_variance_scaling'] = {
                    'scaling_exponent': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'intercept': intercept,
                    'interpretation': 'super-linear' if slope > 1 else 'sub-linear' if slope < 1 else 'linear'
                }
                
                print(f"   Mean-variance scaling exponent: {slope:.3f} ({results['mean_variance_scaling']['interpretation']})")
                print(f"   R²: {r_value**2:.3f}, p-value: {p_value:.4f}")
        
        # Feature activation frequency scaling
        print("📊 Analyzing activation frequency scaling...")
        
        activation_frequencies = (activation_data > 0).mean()
        activation_magnitudes = activation_data.mean()
        
        # Relationship between frequency and magnitude
        if SCIPY_AVAILABLE:
            valid_mask = (activation_frequencies > 0) & (activation_magnitudes > 0)
            if np.sum(valid_mask) > 5:
                freq_mag_corr, freq_mag_p = stats.pearsonr(
                    activation_frequencies[valid_mask], 
                    activation_magnitudes[valid_mask]
                )
                
                results['frequency_magnitude_relationship'] = {
                    'correlation': freq_mag_corr,
                    'p_value': freq_mag_p,
                    'significant': freq_mag_p < 0.05,
                    'interpretation': 'positive' if freq_mag_corr > 0.1 else 'negative' if freq_mag_corr < -0.1 else 'weak'
                }
                
                print(f"   Frequency-magnitude correlation: {freq_mag_corr:.3f} (p={freq_mag_p:.4f})")
        
        # Sparsity scaling patterns
        print("📊 Analyzing sparsity patterns...")
        
        # Calculate sparsity metrics
        sparsity_l0 = (activation_data == 0).mean()  # Fraction of zero activations
        sparsity_l1 = activation_data.abs().mean()   # L1 norm (average absolute activation)
        
        results['sparsity_analysis'] = {
            'l0_sparsity_mean': sparsity_l0.mean(),
            'l0_sparsity_std': sparsity_l0.std(),
            'l1_norm_mean': sparsity_l1.mean(),
            'l1_norm_std': sparsity_l1.std(),
            'sparsity_distribution': sparsity_l0.describe().to_dict()
        }
        
        print(f"   Mean L0 sparsity: {sparsity_l0.mean():.3f} ± {sparsity_l0.std():.3f}")
        print(f"   Mean L1 norm: {sparsity_l1.mean():.3f} ± {sparsity_l1.std():.3f}")
        
        # Heavy-tailed distribution analysis
        if SCIPY_AVAILABLE:
            print("📊 Analyzing heavy-tailed distributions...")
            
            heavy_tail_results = {}
            
            for feature in self.activation_cols[:min(20, len(self.activation_cols))]:
                feature_data = activation_data[feature].values
                positive_data = feature_data[feature_data > 0]
                
                if len(positive_data) < 20:
                    continue
                
                # Kurtosis (measure of tail heaviness)
                kurt = stats.kurtosis(positive_data)
                
                # Tail ratio (95th percentile / median)
                if np.median(positive_data) > 0:
                    tail_ratio = np.percentile(positive_data, 95) / np.median(positive_data)
                else:
                    tail_ratio = np.inf
                
                heavy_tail_results[feature] = {
                    'kurtosis': kurt,
                    'tail_ratio': tail_ratio,
                    'heavy_tailed': kurt > 3 or tail_ratio > 10
                }
            
            if heavy_tail_results:
                kurtosis_values = [result['kurtosis'] for result in heavy_tail_results.values()]
                tail_ratios = [result['tail_ratio'] for result in heavy_tail_results.values() if np.isfinite(result['tail_ratio'])]
                heavy_tailed_count = sum(result['heavy_tailed'] for result in heavy_tail_results.values())
                
                results['heavy_tail_analysis'] = {
                    'mean_kurtosis': np.mean(kurtosis_values),
                    'mean_tail_ratio': np.mean(tail_ratios) if tail_ratios else 0,
                    'heavy_tailed_features': heavy_tailed_count,
                    'heavy_tailed_percentage': heavy_tailed_count / len(heavy_tail_results),
                    'feature_results': heavy_tail_results
                }
                
                print(f"   Heavy-tailed features: {heavy_tailed_count}/{len(heavy_tail_results)} ({heavy_tailed_count/len(heavy_tail_results):.1%})")
                print(f"   Mean kurtosis: {np.mean(kurtosis_values):.2f}")
        
        return results
    
    def analyze_reconstruction_quality(self) -> Dict[str, Any]:
        """Analyze SAE reconstruction quality metrics."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for reconstruction quality analysis")
            return {}
        
        print("\n=== 🔧 SAE RECONSTRUCTION QUALITY ANALYSIS ===")
        
        results = {}
        
        if not self.activation_cols:
            print("⚠️  No activation columns found for reconstruction analysis")
            return results
        
        activation_data = self.df[self.activation_cols].astype(float)
        
        # L0 sparsity analysis (number of non-zero features)
        print("📊 Analyzing L0 sparsity...")
        
        l0_sparsity = (activation_data != 0).sum(axis=1)  # Number of active features per sample
        
        results['l0_sparsity'] = {
            'mean': l0_sparsity.mean(),
            'std': l0_sparsity.std(),
            'median': l0_sparsity.median(),
            'min': l0_sparsity.min(),
            'max': l0_sparsity.max(),
            'percentiles': {
                '25th': l0_sparsity.quantile(0.25),
                '75th': l0_sparsity.quantile(0.75),
                '90th': l0_sparsity.quantile(0.90),
                '95th': l0_sparsity.quantile(0.95)
            },
            'total_features': len(self.activation_cols),
            'sparsity_ratio': l0_sparsity.mean() / len(self.activation_cols)
        }
        
        print(f"   Mean L0 sparsity: {l0_sparsity.mean():.1f}/{len(self.activation_cols)} features ({results['l0_sparsity']['sparsity_ratio']:.1%})")
        print(f"   Median L0 sparsity: {l0_sparsity.median():.1f} features")
        
        # L1 sparsity analysis (sum of absolute activations)
        print("📊 Analyzing L1 sparsity...")
        
        l1_sparsity = activation_data.abs().sum(axis=1)  # L1 norm per sample
        
        results['l1_sparsity'] = {
            'mean': l1_sparsity.mean(),
            'std': l1_sparsity.std(),
            'median': l1_sparsity.median(),
            'min': l1_sparsity.min(),
            'max': l1_sparsity.max(),
            'percentiles': {
                '25th': l1_sparsity.quantile(0.25),
                '75th': l1_sparsity.quantile(0.75),
                '90th': l1_sparsity.quantile(0.90),
                '95th': l1_sparsity.quantile(0.95)
            }
        }
        
        print(f"   Mean L1 norm: {l1_sparsity.mean():.3f}")
        print(f"   Median L1 norm: {l1_sparsity.median():.3f}")
        
        # Feature density analysis
        print("📊 Analyzing feature density...")
        
        feature_activation_rates = (activation_data != 0).mean()  # Fraction of samples where each feature is active
        
        results['feature_density'] = {
            'mean_activation_rate': feature_activation_rates.mean(),
            'std_activation_rate': feature_activation_rates.std(),
            'median_activation_rate': feature_activation_rates.median(),
            'min_activation_rate': feature_activation_rates.min(),
            'max_activation_rate': feature_activation_rates.max(),
            'dead_features': (feature_activation_rates == 0).sum(),
            'always_active_features': (feature_activation_rates == 1).sum(),
            'rare_features': (feature_activation_rates < 0.01).sum(),
            'common_features': (feature_activation_rates > 0.5).sum()
        }
        
        dead_pct = results['feature_density']['dead_features'] / len(self.activation_cols)
        rare_pct = results['feature_density']['rare_features'] / len(self.activation_cols)
        
        print(f"   Mean feature activation rate: {feature_activation_rates.mean():.3f}")
        print(f"   Dead features: {results['feature_density']['dead_features']}/{len(self.activation_cols)} ({dead_pct:.1%})")
        print(f"   Rare features (<1%): {results['feature_density']['rare_features']}/{len(self.activation_cols)} ({rare_pct:.1%})")
        
        # Activation magnitude distribution analysis
        print("📊 Analyzing activation magnitudes...")
        
        # Only consider non-zero activations
        nonzero_activations = activation_data[activation_data != 0].values.flatten()
        nonzero_activations = nonzero_activations[~np.isnan(nonzero_activations)]
        
        if len(nonzero_activations) > 0:
            results['activation_magnitudes'] = {
                'mean': np.mean(nonzero_activations),
                'std': np.std(nonzero_activations),
                'median': np.median(nonzero_activations),
                'min': np.min(nonzero_activations),
                'max': np.max(nonzero_activations),
                'percentiles': {
                    '1st': np.percentile(nonzero_activations, 1),
                    '5th': np.percentile(nonzero_activations, 5),
                    '25th': np.percentile(nonzero_activations, 25),
                    '75th': np.percentile(nonzero_activations, 75),
                    '95th': np.percentile(nonzero_activations, 95),
                    '99th': np.percentile(nonzero_activations, 99)
                },
                'positive_fraction': np.mean(nonzero_activations > 0),
                'negative_fraction': np.mean(nonzero_activations < 0)
            }
            
            print(f"   Non-zero activation mean: {np.mean(nonzero_activations):.3f}")
            print(f"   Non-zero activation range: [{np.min(nonzero_activations):.3f}, {np.max(nonzero_activations):.3f}]")
            print(f"   Positive/negative split: {results['activation_magnitudes']['positive_fraction']:.1%}/{results['activation_magnitudes']['negative_fraction']:.1%}")
        
        # Reconstruction quality metrics (if we can infer them)
        print("📊 Analyzing reconstruction patterns...")
        
        # Feature co-activation patterns
        if len(self.activation_cols) > 1:
            # Sample a subset for computational efficiency
            sample_features = self.activation_cols[:min(100, len(self.activation_cols))]
            sample_data = activation_data[sample_features]
            
            # Binary activation matrix
            binary_activations = (sample_data != 0).astype(int)
            
            # Co-activation frequency
            coactivation_matrix = binary_activations.T @ binary_activations / len(binary_activations)
            
            # Remove diagonal (self-coactivation)
            np.fill_diagonal(coactivation_matrix.values, 0)
            
            coactivation_values = coactivation_matrix.values.flatten()
            coactivation_values = coactivation_values[coactivation_values > 0]
            
            if len(coactivation_values) > 0:
                results['coactivation_patterns'] = {
                    'mean_coactivation': np.mean(coactivation_values),
                    'std_coactivation': np.std(coactivation_values),
                    'max_coactivation': np.max(coactivation_values),
                    'high_coactivation_pairs': np.sum(coactivation_values > 0.5),
                    'total_pairs': len(coactivation_values)
                }
                
                high_coact_pct = results['coactivation_patterns']['high_coactivation_pairs'] / results['coactivation_patterns']['total_pairs']
                print(f"   Mean co-activation rate: {np.mean(coactivation_values):.3f}")
                print(f"   High co-activation pairs (>50%): {results['coactivation_patterns']['high_coactivation_pairs']}/{results['coactivation_patterns']['total_pairs']} ({high_coact_pct:.1%})")
        
        # Polysemanticity indicators
        print("📊 Analyzing polysemanticity indicators...")
        
        # Feature activation consistency across different contexts
        if 'correct_top1' in self.df.columns:
            correct_mask = self.df['correct_top1'].apply(lambda x: x == 'Yes' or x == True or x == 1)
            incorrect_mask = ~correct_mask
            
            if correct_mask.sum() > 0 and incorrect_mask.sum() > 0:
                correct_activations = activation_data[correct_mask].mean()
                incorrect_activations = activation_data[incorrect_mask].mean()
                
                # Features that activate differently for correct vs incorrect predictions
                activation_diff = (correct_activations - incorrect_activations).abs()
                
                results['context_sensitivity'] = {
                    'mean_activation_difference': activation_diff.mean(),
                    'std_activation_difference': activation_diff.std(),
                    'highly_sensitive_features': (activation_diff > activation_diff.quantile(0.9)).sum(),
                    'stable_features': (activation_diff < activation_diff.quantile(0.1)).sum(),
                    'total_features': len(activation_diff)
                }
                
                sensitive_pct = results['context_sensitivity']['highly_sensitive_features'] / len(activation_diff)
                stable_pct = results['context_sensitivity']['stable_features'] / len(activation_diff)
                
                print(f"   Context-sensitive features: {results['context_sensitivity']['highly_sensitive_features']}/{len(activation_diff)} ({sensitive_pct:.1%})")
                print(f"   Context-stable features: {results['context_sensitivity']['stable_features']}/{len(activation_diff)} ({stable_pct:.1%})")
        
        # Overall reconstruction quality summary
        results['quality_summary'] = {
            'sparsity_level': 'high' if results['l0_sparsity']['sparsity_ratio'] < 0.1 else 'medium' if results['l0_sparsity']['sparsity_ratio'] < 0.3 else 'low',
            'feature_utilization': 'good' if results['feature_density']['dead_features'] / len(self.activation_cols) < 0.1 else 'poor',
            'activation_balance': 'balanced' if 0.3 <= results['activation_magnitudes']['positive_fraction'] <= 0.7 else 'imbalanced',
            'total_samples': len(self.df),
            'total_features': len(self.activation_cols)
        }
        
        print(f"\n📋 Quality Summary:")
        print(f"   Sparsity level: {results['quality_summary']['sparsity_level']}")
        print(f"   Feature utilization: {results['quality_summary']['feature_utilization']}")
        print(f"   Activation balance: {results['quality_summary']['activation_balance']}")
        
        return results
    
    def run_full_analysis(self, save_results: bool = True, generate_plots: bool = True) -> Dict[str, Any]:
        """Run complete analysis pipeline."""
        print(f"\n🚀 RUNNING FULL ANALYSIS ON {self.csv_path.name}")
        print("=" * 60)
        
        # Check if data loaded successfully
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for analysis")
            return {}
        
        # Run all analyses
        self.analysis_results['accuracy'] = self.analyze_accuracy_metrics()
        self.analysis_results['diagnosis'] = self.analyze_diagnosis_distribution()
        self.analysis_results['features'] = self.analyze_feature_activations()
        self.analysis_results['demographics'] = self.analyze_demographic_effects()
        self.analysis_results['clamping'] = self.analyze_clamping_effects()
        self.analysis_results['bias'] = self.analyze_bias_metrics()
        self.analysis_results['correlations'] = self.analyze_feature_correlations()
        self.analysis_results['steering'] = self.analyze_steering_effects()
        self.analysis_results['bootstrap'] = self.bootstrap_analysis()
        self.analysis_results['intersectionality'] = self.analyze_intersectionality()
        self.analysis_results['scaling'] = self.analyze_scaling_patterns()
        self.analysis_results['reconstruction'] = self.analyze_reconstruction_quality()
        
        # Add metadata
        self.analysis_results['metadata'] = {
            'file_path': str(self.csv_path),
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'activation_features': len(self.activation_cols),
            'demographic_columns': self.demo_cols
        }
        
        # Save results
        if save_results:
            results_path = self.csv_path.parent / f"{self.csv_path.stem}_analysis_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            print(f"💾 Analysis results saved to {results_path}")
        
        # Generate visualizations
        if generate_plots:
            self.generate_visualizations()
        
        print("\n✅ ANALYSIS COMPLETE!")
        return self.analysis_results
    
    def generate_visualizations(self, output_dir: str = None):
        """Generate visualization plots."""
        if output_dir is None:
            output_dir = self.csv_path.parent / f"{self.csv_path.stem}_analysis"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n=== 📊 GENERATING VISUALIZATIONS ===")
        print(f"📁 Output directory: {output_dir}")
        
        plt.style.use('default')
        
        # 1. Accuracy distribution
        if 'correct_top1' in self.df.columns:
            plt.figure(figsize=(10, 6))
            
            # Accuracy by demographics
            if self.demo_cols:
                fig, axes = plt.subplots(1, len(self.demo_cols), figsize=(5*len(self.demo_cols), 5))
                if len(self.demo_cols) == 1:
                    axes = [axes]
                
                for i, demo_col in enumerate(self.demo_cols):
                    if demo_col in self.df.columns:
                        demo_acc = self.df.groupby(demo_col)['correct_top1'].apply(
                            lambda x: (x == 'Yes').mean() if x.dtype == 'object' else x.mean()
                        ).dropna()
                        
                        if not demo_acc.empty:
                            demo_acc.plot(kind='bar', ax=axes[i], title=f'Accuracy by {demo_col}')
                            axes[i].set_ylabel('Accuracy')
                            axes[i].tick_params(axis='x', rotation=45)
                        else:
                            axes[i].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[i].transAxes)
                            axes[i].set_title(f'Accuracy by {demo_col}')
                            axes[i].set_xticks([])
                            axes[i].set_yticks([])
                
                plt.tight_layout()
                plt.savefig(output_dir / 'accuracy_by_demographics.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. Feature activation heatmap
        if self.activation_cols and len(self.activation_cols) > 1:
            plt.figure(figsize=(12, 8))
            
            # Sample of activations for visualization
            sample_size = min(100, len(self.df))
            sample_features = self.activation_cols[:20]  # First 20 features
            
            activation_sample = self.df[sample_features].head(sample_size).astype(float)
            
            plt.imshow(activation_sample.T, cmap='RdBu_r', aspect='auto')
            plt.title('Feature Activation Patterns (Sample)')
            plt.ylabel('Features')
            plt.xlabel('Samples')
            plt.tight_layout()
            plt.savefig(output_dir / 'activation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Diagnosis distribution
        if 'diagnosis' in self.df.columns:
            plt.figure(figsize=(12, 6))
            
            top_diagnoses = self.df['diagnosis'].value_counts().head(15)
            top_diagnoses.plot(kind='bar')
            plt.title('Top 15 Diagnosis Distribution')
            plt.xlabel('Diagnosis')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / 'diagnosis_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Visualizations saved to {output_dir}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Automated Medical Data Analysis Pipeline')
    parser.add_argument('csv_file', help='Path to CSV file to analyze')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--no-save', action='store_true', help='Skip saving results')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = MedicalDataAnalyzer(args.csv_file)
    results = analyzer.run_full_analysis(
        save_results=not args.no_save,
        generate_plots=not args.no_plots
    )
    
    # Print summary
    print(f"\n📋 ANALYSIS SUMMARY:")
    if 'accuracy' in results and 'top1_accuracy' in results['accuracy']:
        print(f"   🎯 Top-1 Accuracy: {results['accuracy']['top1_accuracy']:.3f}")
    if 'features' in results and 'activation_stats' in results['features']:
        print(f"   🧠 Mean Activation: {results['features']['activation_stats']['mean']:.4f}")
        print(f"   🧠 Sparsity: {results['features']['activation_stats']['sparsity']:.2%}")


if __name__ == "__main__":
    main()
