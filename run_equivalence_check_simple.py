#!/usr/bin/env python3
"""
Simple Equivalence Check using Existing Pipeline

This script runs equivalence tests by calling the existing main.py pipeline
with different configurations and comparing the results.
"""

import argparse
import json
import os
import subprocess
import tempfile
import csv
from datetime import datetime
from typing import Dict, List, Any
import ast

def load_test_cases(test_file: str) -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    print(f"📂 Loading test cases from: {test_file}")
    
    with open(test_file, 'r') as f:
        test_cases = json.load(f)
    
    print(f"✅ Loaded {len(test_cases)} test cases")
    return test_cases

def create_patient_csv(case_id: int, prompt: str, age: int = 30, sex: str = 'M') -> str:
    """Create a temporary patient CSV file."""
    temp_file = f"/tmp/temp_patient_{case_id}_{datetime.now().strftime('%H%M%S_%f')}.csv"
    
    with open(temp_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'case_text', 'AGE', 'SEX'])
        writer.writerow([case_id, prompt, age, sex])
    
    return temp_file

def extract_features_from_clamping_dict(clamping_features: Dict[str, float]) -> List[str]:
    """Extract demographic groups from clamping features."""
    # Convert string keys to integers
    feature_ids = [int(k) for k in clamping_features.keys()]
    
    demographic_groups = []
    
    # Age feature mappings (from constants_v2.py)
    pediatric_features = [3296, 14423, 5565]
    adolescent_features = [801, 7398, 5565]
    young_adult_features = [7398, 1999, 5565]
    middle_age_features = [13032, 1999, 11060, 5565]
    senior_features = [11060, 13032, 6679, 5565]
    
    # Sex feature mappings
    male_features = [11096, 13353, 8409, 12221]
    female_features = [387, 6221, 5176, 12813]
    
    # Check for age features
    if any(f in pediatric_features for f in feature_ids):
        demographic_groups.append("pediatric")
    elif any(f in adolescent_features for f in feature_ids):
        demographic_groups.append("adolescent")
    elif any(f in young_adult_features for f in feature_ids):
        demographic_groups.append("young_adult")
    elif any(f in middle_age_features for f in feature_ids):
        demographic_groups.append("middle_age")
    elif any(f in senior_features for f in feature_ids):
        demographic_groups.append("senior")
    
    # Check for sex features
    if any(f in male_features for f in feature_ids):
        demographic_groups.append("male")
    elif any(f in female_features for f in feature_ids):
        demographic_groups.append("female")
    
    return demographic_groups

def run_pipeline_test(prompt: str, case_id: int, clamp_features: List[str] = None) -> Dict[str, Any]:
    """Run the pipeline and return results."""
    
    # Create temporary patient file
    patient_file = create_patient_csv(case_id, prompt)
    
    try:
        # Build command
        cmd = [
            "python3", "-m", "src.advai.main",
            "--patient-file", patient_file,
            "--num-cases", "1",
            "--device", "cpu"
        ]
        
        # Add clamping if specified
        if clamp_features:
            cmd.extend(["--clamp"])
            cmd.extend(["--clamp-features"] + clamp_features)
            cmd.extend(["--clamp-intensity", "1.0"])
        
        print(f"   🔧 Running: {' '.join(cmd)}")
        
        # Run command with timeout
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120,  # 2 minute timeout
            cwd="/Users/amelia/22406alethia/alethia"
        )
        
        # Parse output
        output = result.stdout
        error = result.stderr
        
        # Try to extract meaningful results from output
        diagnosis_info = "unknown"
        if "diagnosis" in output.lower() or "condition" in output.lower():
            lines = output.split('\n')
            for line in lines:
                if any(word in line.lower() for word in ['diagnosis', 'condition', 'disease']):
                    diagnosis_info = line.strip()
                    break
        
        return {
            "success": result.returncode == 0,
            "output": output,
            "error": error,
            "diagnosis": diagnosis_info,
            "command": ' '.join(cmd),
            "return_code": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": "Command timed out after 2 minutes",
            "diagnosis": "timeout",
            "command": ' '.join(cmd),
            "return_code": -1
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "diagnosis": "error",
            "command": ' '.join(cmd),
            "return_code": -1
        }
    finally:
        # Clean up temp file
        if os.path.exists(patient_file):
            os.remove(patient_file)

def compare_pipeline_results(result_a: Dict[str, Any], result_b: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two pipeline results."""
    
    # Both must be successful
    both_successful = result_a["success"] and result_b["success"]
    
    if not both_successful:
        return {
            "equivalent": False,
            "reason": f"Command failures: A={result_a['success']}, B={result_b['success']}",
            "diagnosis_a": result_a["diagnosis"],
            "diagnosis_b": result_b["diagnosis"],
            "error_a": result_a.get("error", ""),
            "error_b": result_b.get("error", "")
        }
    
    # Compare outputs (simple text comparison)
    output_a = result_a["output"].strip()
    output_b = result_b["output"].strip()
    
    # Simple similarity metrics
    exact_match = output_a == output_b
    
    # Word-level comparison
    words_a = set(output_a.lower().split())
    words_b = set(output_b.lower().split())
    
    if len(words_a) == 0 and len(words_b) == 0:
        word_overlap = 1.0
    elif len(words_a) == 0 or len(words_b) == 0:
        word_overlap = 0.0
    else:
        word_overlap = len(words_a.intersection(words_b)) / len(words_a.union(words_b))
    
    # Consider equivalent if high word overlap
    equivalent = exact_match or word_overlap > 0.8
    
    return {
        "equivalent": equivalent,
        "exact_match": exact_match,
        "word_overlap": word_overlap,
        "diagnosis_a": result_a["diagnosis"],
        "diagnosis_b": result_b["diagnosis"],
        "output_similarity": word_overlap
    }

def run_equivalence_check(test_cases: List[Dict[str, Any]], 
                         max_cases: int = None,
                         output_dir: str = "equivalence_results") -> Dict[str, Any]:
    """Run the equivalence check on test cases."""
    
    print(f"\n🧪 RUNNING SIMPLE EQUIVALENCE CHECK")
    print(f"📊 Total test cases: {len(test_cases)}")
    if max_cases:
        test_cases = test_cases[:max_cases]
        print(f"🔢 Limited to first {max_cases} cases")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    equivalence_counts = {"equivalent": 0, "different": 0, "failed": 0}
    test_type_stats = {}
    
    # Run tests
    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 Test {i+1}/{len(test_cases)}: {test_case['test_type']}")
        print(f"   Patient: {test_case['age_group']} {test_case['sex']}")
        
        try:
            # Extract clamping features
            clamping_features_dict = test_case["clamping_features"]
            if isinstance(clamping_features_dict, str):
                # Parse string representation of dict/list
                try:
                    clamping_features_dict = ast.literal_eval(clamping_features_dict)
                except:
                    print(f"   ⚠️  Could not parse clamping features: {clamping_features_dict}")
                    continue
            
            clamp_groups = extract_features_from_clamping_dict(clamping_features_dict)
            
            print(f"   Demographic prompt: {test_case['prompt_with_demographics'][:60]}...")
            print(f"   Neutral prompt: {test_case['neutral_prompt'][:60]}...")
            print(f"   Clamping groups: {clamp_groups}")
            
            # Run prompt with demographics (no clamping)
            print("   🔄 Running demographic prompt...")
            result_a = run_pipeline_test(test_case["prompt_with_demographics"], i*2)
            
            # Run neutral prompt with clamping
            print("   🔄 Running clamped prompt...")
            result_b = run_pipeline_test(test_case["neutral_prompt"], i*2+1, clamp_groups)
            
            # Compare results
            comparison = compare_pipeline_results(result_a, result_b)
            
            # Record result
            result = {
                "test_id": i,
                "case_id": test_case["case_id"],
                "test_type": test_case["test_type"],
                "age_group": test_case["age_group"],
                "sex": test_case["sex"],
                "prompt_a": test_case["prompt_with_demographics"],
                "prompt_b": test_case["neutral_prompt"],
                "clamping_groups": clamp_groups,
                "result_a": result_a,
                "result_b": result_b,
                "comparison": comparison
            }
            results.append(result)
            
            # Update counts
            if not (result_a["success"] and result_b["success"]):
                equivalence_counts["failed"] += 1
                print(f"   ❌ FAILED: {comparison.get('reason', 'Unknown error')}")
            elif comparison["equivalent"]:
                equivalence_counts["equivalent"] += 1
                print(f"   ✅ EQUIVALENT (similarity: {comparison['output_similarity']:.2f})")
            else:
                equivalence_counts["different"] += 1
                print(f"   ❌ DIFFERENT (similarity: {comparison['output_similarity']:.2f})")
            
            # Update test type stats
            test_type = test_case["test_type"]
            if test_type not in test_type_stats:
                test_type_stats[test_type] = {"equivalent": 0, "different": 0, "failed": 0, "total": 0}
            
            test_type_stats[test_type]["total"] += 1
            if not (result_a["success"] and result_b["success"]):
                test_type_stats[test_type]["failed"] += 1
            elif comparison["equivalent"]:
                test_type_stats[test_type]["equivalent"] += 1
            else:
                test_type_stats[test_type]["different"] += 1
                
        except Exception as e:
            print(f"❌ Error processing test case {i}: {e}")
            equivalence_counts["failed"] += 1
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
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f"simple_equivalence_check_{timestamp}.json")
    
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
    print("📊 SIMPLE EQUIVALENCE CHECK RESULTS")
    print("="*80)
    
    print(f"\n🔢 Overall Statistics:")
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
    parser = argparse.ArgumentParser(description="Run simple equivalence check using existing pipeline")
    parser.add_argument("--test-file", 
                       default="patient_specific_equivalence/patient_specific_equivalence_tests_20250629_224636.json",
                       help="JSON file with test cases")
    parser.add_argument("--max-cases", type=int, default=3,
                       help="Maximum number of test cases to run")
    parser.add_argument("--output-dir", default="equivalence_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("🧪 SIMPLE EQUIVALENCE CHECK USING EXISTING PIPELINE")
    print(f"📁 Test file: {args.test_file}")
    print(f"🔢 Max cases: {args.max_cases}")
    print(f"📂 Output: {args.output_dir}")
    print()
    
    # Load test cases
    if not os.path.exists(args.test_file):
        print(f"❌ Test file not found: {args.test_file}")
        print("   Run patient_specific_equivalence_test.py first to generate test cases")
        return
    
    test_cases = load_test_cases(args.test_file)
    
    # Run equivalence check
    summary, results = run_equivalence_check(
        test_cases, 
        max_cases=args.max_cases,
        output_dir=args.output_dir
    )
    
    # Print results
    print_results_summary(summary)
    
    print(f"\n🎯 SIMPLE EQUIVALENCE CHECK COMPLETE!")
    print(f"📊 Tested {summary['total_tests']} cases")
    print(f"✅ {summary['overall_equivalence_rate']:.1%} equivalence rate")

if __name__ == "__main__":
    main()
