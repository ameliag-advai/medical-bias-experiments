#!/usr/bin/env python3
"""
Actual Clamping Equivalence Check

This script loads the generated test cases and executes them through
the actual pipeline with proper SAE loading and clamping to validate 
that demographic clamping produces equivalent results to including 
demographics in prompts.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import logging

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the existing pipeline components
from src.advai.models.loader import load_model_and_sae
from src.advai.analysis.analyse import run_prompt
from src.advai.analysis.clamping_v2 import clamp_sae_features
from src.advai.analysis.constants_v2 import (
    MALE_FEATURES_WITH_DIRECTIONS, FEMALE_FEATURES_WITH_DIRECTIONS, 
    OLD_FEATURES_WITH_DIRECTIONS, YOUNG_FEATURES_WITH_DIRECTIONS,
    PEDIATRIC_FEATURES_WITH_DIRECTIONS, ADOLESCENT_FEATURES_WITH_DIRECTIONS, 
    YOUNG_ADULT_FEATURES_WITH_DIRECTIONS, MIDDLE_AGE_FEATURES_WITH_DIRECTIONS, 
    SENIOR_FEATURES_WITH_DIRECTIONS
)

# Set up device
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

# Disable gradients for inference
torch.set_grad_enabled(False)

class EquivalenceChecker:
    """Actual pipeline for running equivalence checks with real SAE."""
    
    def __init__(self, device=device):
        self.device = device
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model and SAE will be loaded on first use
        self.model = None
        self.sae = None
        
    def load_models(self):
        """Load the language model and SAE using the existing loader."""
        if self.model is None or self.sae is None:
            self.logger.info("Loading model and SAE...")
            self.model, self.sae = load_model_and_sae(model_scope="gemma", device=self.device)
            self.logger.info("✅ Model and SAE loaded successfully")
    
    def convert_features_to_demographic_groups(self, clamping_features: Dict[str, float]) -> List[str]:
        """Convert feature IDs to demographic group names for clamping."""
        demographic_groups = []
        
        # Convert string keys to integers
        feature_ids = [int(k) for k in clamping_features.keys()]
        
        # Check for age features
        if any(f in PEDIATRIC_FEATURES_WITH_DIRECTIONS for f in feature_ids):
            demographic_groups.append("pediatric")
        elif any(f in ADOLESCENT_FEATURES_WITH_DIRECTIONS for f in feature_ids):
            demographic_groups.append("adolescent")
        elif any(f in YOUNG_ADULT_FEATURES_WITH_DIRECTIONS for f in feature_ids):
            demographic_groups.append("young_adult")
        elif any(f in MIDDLE_AGE_FEATURES_WITH_DIRECTIONS for f in feature_ids):
            demographic_groups.append("middle_age")
        elif any(f in SENIOR_FEATURES_WITH_DIRECTIONS for f in feature_ids):
            demographic_groups.append("senior")
        
        # Check for sex features
        if any(f in MALE_FEATURES_WITH_DIRECTIONS for f in feature_ids):
            demographic_groups.append("male")
        elif any(f in FEMALE_FEATURES_WITH_DIRECTIONS for f in feature_ids):
            demographic_groups.append("female")
        
        return demographic_groups
    
    def run_prompt_with_demographics(self, prompt: str) -> Dict[str, Any]:
        """Run a prompt with demographics included in the text."""
        if self.model is None or self.sae is None:
            self.load_models()
        
        # Run without clamping
        result = run_prompt(
            prompt=prompt,
            model=self.model,
            sae=self.sae,
            clamping=False,
            clamp_features=None,
            clamp_value=0.0
        )
        
        return {
            "prompt": prompt,
            "activations": result,
            "method": "demographic_prompt"
        }
    
    def run_prompt_with_clamping(self, prompt: str, clamping_features: Dict[str, float]) -> Dict[str, Any]:
        """Run a neutral prompt with demographic clamping."""
        if self.model is None or self.sae is None:
            self.load_models()
        
        # Convert features to demographic groups
        demographic_groups = self.convert_features_to_demographic_groups(clamping_features)
        
        # Run with clamping - use intensity 1.0 for equivalence test
        result = run_prompt(
            prompt=prompt,
            model=self.model,
            sae=self.sae,
            clamping=True,
            clamp_features=demographic_groups,
            clamp_value=1.0
        )
        
        return {
            "prompt": prompt,
            "activations": result,
            "clamping_features": clamping_features,
            "demographic_groups": demographic_groups,
            "method": "clamping"
        }
    
    def compare_activations(self, result_a: Dict[str, Any], result_b: Dict[str, Any]) -> Dict[str, Any]:
        """Compare SAE activations between two results."""
        
        activations_a = result_a["activations"]
        activations_b = result_b["activations"]
        
        # Extract activation values (skip metadata keys)
        activation_keys = [k for k in activations_a.keys() if k.startswith("activation_")]
        
        if not activation_keys:
            return {
                "equivalent": False,
                "reason": "No activation data found",
                "similarity_score": 0.0
            }
        
        # Calculate similarity metrics
        similarities = []
        differences = []
        
        for key in activation_keys:
            val_a = activations_a.get(key, 0.0)
            val_b = activations_b.get(key, 0.0)
            
            # Cosine similarity for individual activations
            if abs(val_a) > 1e-8 or abs(val_b) > 1e-8:
                similarity = (val_a * val_b) / (abs(val_a) * abs(val_b) + 1e-8)
                similarities.append(similarity)
            
            # Absolute difference
            differences.append(abs(val_a - val_b))
        
        # Overall metrics
        mean_similarity = np.mean(similarities) if similarities else 0.0
        mean_difference = np.mean(differences)
        max_difference = max(differences) if differences else 0.0
        
        # Equivalence criteria
        # Consider equivalent if:
        # 1. Mean similarity > 0.8 (high correlation)
        # 2. Mean difference < 0.1 (small absolute differences)
        # 3. Max difference < 0.5 (no huge outliers)
        
        equivalent = (
            mean_similarity > 0.8 and 
            mean_difference < 0.1 and 
            max_difference < 0.5
        )
        
        return {
            "equivalent": equivalent,
            "mean_similarity": mean_similarity,
            "mean_difference": mean_difference,
            "max_difference": max_difference,
            "num_activations_compared": len(activation_keys),
            "similarity_score": mean_similarity
        }

def load_test_cases(test_file: str) -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    print(f"📂 Loading test cases from: {test_file}")
    
    with open(test_file, 'r') as f:
        test_cases = json.load(f)
    
    print(f"✅ Loaded {len(test_cases)} test cases")
    return test_cases

def run_equivalence_check(test_cases: List[Dict[str, Any]], 
                         checker: EquivalenceChecker,
                         max_cases: int = None,
                         output_dir: str = "equivalence_results") -> Dict[str, Any]:
    """Run the equivalence check on test cases."""
    
    print(f"\n🧪 RUNNING ACTUAL EQUIVALENCE CHECK")
    print(f"📊 Total test cases: {len(test_cases)}")
    if max_cases:
        test_cases = test_cases[:max_cases]
        print(f"🔢 Limited to first {max_cases} cases")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    equivalence_counts = {"equivalent": 0, "different": 0, "failed": 0}
    test_type_stats = {}
    
    # Run tests
    for i, test_case in enumerate(tqdm(test_cases, desc="Running equivalence tests")):
        try:
            print(f"\n🧪 Test {i+1}/{len(test_cases)}: {test_case['test_type']}")
            print(f"   Patient: {test_case['age_group']} {test_case['sex']}")
            print(f"   Prompt A: {test_case['prompt_with_demographics'][:80]}...")
            print(f"   Prompt B: {test_case['neutral_prompt'][:80]}...")
            
            # Run prompt with demographics
            print("   🔄 Running demographic prompt...")
            result_a = checker.run_prompt_with_demographics(test_case["prompt_with_demographics"])
            
            # Run neutral prompt with clamping
            print("   🔄 Running clamped prompt...")
            result_b = checker.run_prompt_with_clamping(
                test_case["neutral_prompt"],
                test_case["clamping_features"]
            )
            
            # Compare results
            print("   📊 Comparing activations...")
            comparison = checker.compare_activations(result_a, result_b)
            
            # Record result
            result = {
                "test_id": i,
                "case_id": test_case["case_id"],
                "test_type": test_case["test_type"],
                "age_group": test_case["age_group"],
                "sex": test_case["sex"],
                "prompt_a": test_case["prompt_with_demographics"],
                "prompt_b": test_case["neutral_prompt"],
                "clamping_features": test_case["clamping_features"],
                "result_a": result_a,
                "result_b": result_b,
                "comparison": comparison,
                "equivalent": comparison["equivalent"]
            }
            results.append(result)
            
            # Update counts
            if comparison["equivalent"]:
                equivalence_counts["equivalent"] += 1
                print(f"   ✅ EQUIVALENT (similarity: {comparison['similarity_score']:.3f})")
            else:
                equivalence_counts["different"] += 1
                print(f"   ❌ DIFFERENT (similarity: {comparison['similarity_score']:.3f})")
            
            # Update test type stats
            test_type = test_case["test_type"]
            if test_type not in test_type_stats:
                test_type_stats[test_type] = {"equivalent": 0, "different": 0, "failed": 0, "total": 0}
            
            test_type_stats[test_type]["total"] += 1
            if comparison["equivalent"]:
                test_type_stats[test_type]["equivalent"] += 1
            else:
                test_type_stats[test_type]["different"] += 1
                
        except Exception as e:
            print(f"❌ Error processing test case {i}: {e}")
            equivalence_counts["failed"] += 1
            
            # Still record the failure
            if test_case["test_type"] not in test_type_stats:
                test_type_stats[test_case["test_type"]] = {"equivalent": 0, "different": 0, "failed": 0, "total": 0}
            test_type_stats[test_case["test_type"]]["failed"] += 1
            test_type_stats[test_case["test_type"]]["total"] += 1
            continue
    
    # Calculate statistics
    total_tests = len(results)
    successful_tests = equivalence_counts["equivalent"] + equivalence_counts["different"]
    equivalence_rate = equivalence_counts["equivalent"] / successful_tests if successful_tests > 0 else 0
    
    # Calculate per-test-type statistics
    for test_type in test_type_stats:
        stats = test_type_stats[test_type]
        successful = stats["equivalent"] + stats["different"]
        stats["equivalence_rate"] = stats["equivalent"] / successful if successful > 0 else 0
    
    summary = {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "equivalence_counts": equivalence_counts,
        "overall_equivalence_rate": equivalence_rate,
        "test_type_stats": test_type_stats,
        "timestamp": datetime.now().isoformat(),
        "device_used": device
    }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f"actual_equivalence_check_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            "summary": summary,
            "detailed_results": results
        }, f, indent=2, default=str)
    
    print(f"💾 Results saved: {results_file}")
    
    return summary, results

def print_results_summary(summary: Dict[str, Any]):
    """Print a summary of the equivalence check results."""
    
    print(f"\n" + "="*80)
    print("📊 ACTUAL EQUIVALENCE CHECK RESULTS")
    print("="*80)
    
    print(f"\n🔢 Overall Statistics:")
    print(f"   Device used: {summary.get('device_used', 'unknown')}")
    print(f"   Total tests run: {summary['total_tests']}")
    print(f"   Successful tests: {summary['successful_tests']}")
    print(f"   Failed tests: {summary['equivalence_counts']['failed']}")
    print(f"   Equivalent results: {summary['equivalence_counts']['equivalent']}")
    print(f"   Different results: {summary['equivalence_counts']['different']}")
    print(f"   Overall equivalence rate: {summary['overall_equivalence_rate']:.1%}")
    
    print(f"\n📋 By Test Type:")
    for test_type, stats in summary['test_type_stats'].items():
        successful = stats['equivalent'] + stats['different']
        print(f"   {test_type}:")
        print(f"     Total: {stats['total']}")
        print(f"     Successful: {successful}")
        print(f"     Equivalent: {stats['equivalent']} ({stats['equivalence_rate']:.1%})")
        print(f"     Different: {stats['different']}")
        print(f"     Failed: {stats['failed']}")
    
    print(f"\n✅ Interpretation:")
    overall_rate = summary['overall_equivalence_rate']
    if overall_rate >= 0.9:
        print("   🎯 EXCELLENT: >90% equivalence validates demographic clamping!")
    elif overall_rate >= 0.7:
        print("   ✅ GOOD: >70% equivalence shows clamping is mostly working")
    elif overall_rate >= 0.5:
        print("   ⚠️  MODERATE: ~50% equivalence suggests some issues with clamping")
    else:
        print("   ❌ POOR: <50% equivalence indicates significant clamping problems")

def main():
    parser = argparse.ArgumentParser(description="Run actual clamping equivalence check")
    parser.add_argument("--test-file", 
                       default="patient_specific_equivalence/patient_specific_equivalence_tests_20250629_224636.json",
                       help="JSON file with test cases")
    parser.add_argument("--max-cases", type=int, default=10,
                       help="Maximum number of test cases to run")
    parser.add_argument("--output-dir", default="equivalence_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("🧪 ACTUAL CLAMPING EQUIVALENCE CHECK")
    print(f"📁 Test file: {args.test_file}")
    print(f"🔢 Max cases: {args.max_cases}")
    print(f"📂 Output: {args.output_dir}")
    print(f"💻 Device: {device}")
    print()
    
    # Load test cases
    if not os.path.exists(args.test_file):
        print(f"❌ Test file not found: {args.test_file}")
        print("   Run patient_specific_equivalence_test.py first to generate test cases")
        return
    
    test_cases = load_test_cases(args.test_file)
    
    # Initialize checker
    checker = EquivalenceChecker(device=device)
    
    # Run equivalence check
    summary, results = run_equivalence_check(
        test_cases, 
        checker, 
        max_cases=args.max_cases,
        output_dir=args.output_dir
    )
    
    # Print results
    print_results_summary(summary)
    
    print(f"\n🎯 ACTUAL EQUIVALENCE CHECK COMPLETE!")
    print(f"📊 Tested {summary['total_tests']} cases")
    print(f"✅ {summary['overall_equivalence_rate']:.1%} equivalence rate")

if __name__ == "__main__":
    main()
