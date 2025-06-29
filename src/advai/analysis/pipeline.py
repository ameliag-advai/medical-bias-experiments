"""This module provides a pipeline for analyzing bias in AI models for a given dataset."""
from datetime import datetime
import csv
import io
import json
import os
import sys
from itertools import product
from tqdm import tqdm
from typing import Tuple

import torch

from .analyse import (
    run_prompt,
    compile_results,
    extract_top_diagnoses,
)
from .clamping_v2 import clamp_sae_features as clamp_activations
from .constants_v2 import (
    FIELD_NAMES,
    CLAMPING_FIELD_NAMES,
    MALE_FEATURES,
    FEMALE_FEATURES,
    OLD_FEATURES,
    YOUNG_FEATURES,
)
from .summary import generate_summary, write_output
from ..data.io import (
    extract_cases_from_dataframe,
    load_conditions_mapping,
    load_patient_data,
)
from ..data.example_templates import TEMPLATE_SETS
from ..data.prompt_builder import PromptBuilder, get_subsets
from ..visuals.plots import visualize_feature_overlaps

# Setup global outputs directory and timestamped CSV path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "src", "advai", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def get_templates(demographic_concepts: list[str]):
    key = frozenset(demographic_concepts)
    return TEMPLATE_SETS.get(key)


def data_preprocessing(
    patient_data_path: str,
    conditions_json_path: str,
    evidences_json_path: str,
    num_cases: int = 1,
    start_case: int = 0,
) -> Tuple:
    """Load and preprocess patient data and conditions mapping.

    :param patient_data_path: Path to the patient data CSV file.
    :param conditions_json_path: Path to the conditions mapping JSON file.
    :param num_cases: Number of cases to analyze from the dataset.
    :return: A tuple containing the cases and conditions mapping.
    """
    df = load_patient_data(patient_data_path)
    cases = extract_cases_from_dataframe(df)
    if num_cases:
        cases = cases[start_case : start_case + num_cases]

    conditions_mapping = load_conditions_mapping(conditions_json_path)

    with open(evidences_json_path, "r") as f:
        evidences = json.load(f)

    return cases, conditions_mapping, evidences


def process_case_result(prompt_outputs, pairs_to_compare, case_id=None, case_info=None):
    """Process the activations and compare them for each pair.

    :param prompt_outputs: Dictionary of prompt outputs for each demographic combination.
    :param pairs_to_compare: List of pairs of demographic combinations to compare.
    :param case_id: Optional identifier for the case.
    :param case_info: Optional additional information about the case.
    :return: A dictionary with the results of the comparisons for each pair.
    """
    case_result = {}
    for pair in pairs_to_compare:
        prompt_output_1 = prompt_outputs[pair[0]]
        prompt_output_2 = prompt_outputs[pair[1]]
        if prompt_output_1 is None or prompt_output_2 is None:
            case_result[pair] = None
        else:
            case_result[pair] = compile_results(
                prompt_output_1,
                prompt_output_2,
                pair,
                case_id=case_id,
                case_info=case_info,
            )
    return case_result


def run_analysis_pipeline(
    patient_data_path,
    conditions_json_path,
    evidences_json_path,
    model,
    sae,
    num_cases: int = 1,
    start_case: int = 0,
    topk: int = 5,
    demographic_concepts: list[str] = ["age", "sex", "male", "female", "old", "young"],
    concepts_to_test: list[str] = ["male", "female", "old", "young"],
    save_dir: str = "activations",
    output_name: str = None,
    clamping: bool = False,
    clamp_features: list[str] = MALE_FEATURES + FEMALE_FEATURES + OLD_FEATURES + YOUNG_FEATURES,
    clamp_values: list[int] = [0, 5, 10],
    interactive: bool = True,
) -> str:
    """Run the full analysis pipeline including loading data, generating prompts,
    analyzing bias, and writing results to disk.

    :param patient_data_path: Path to the patient data CSV file.
    :param conditions_json_path: Path to the conditions mapping JSON file.
    :param model: The model to use for generating activations.
    :param sae: The SAE model to use for generating activations.
    :param num_cases: Number of cases to analyze from the dataset.
    :param demographic_concepts: List of demographic concepts to include in the prompts.
    :param concepts_to_test: List of concepts to test for bias.
    :param save_dir: Directory to save the generated prompts and results.
    :param output_name: Optional name for the output files.
    :return: Path to the analysis output file.
    """
    
    # Track run timing and metadata
    import time
    import sys
    run_start_time = time.time()
    run_start_datetime = datetime.now()
    
    print(f"🚀 Starting analysis pipeline at {run_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    # Get the patient data and conditions mapping
    cases, conditions_mapping, evidences = data_preprocessing(
        patient_data_path,
        conditions_json_path,
        evidences_json_path,
        num_cases=num_cases,
        start_case=start_case,
    )

    # Get prompt templates - interactive or default
    if interactive:
        user_instruction = (
            "Enter the full prompt template. Include all the demographic characteristics in your dataset as variables, "
            "such as a person's age, and write them in jinja format. Include all other relevant variables, such as the person's symptoms. "
            "For example: 'Patient has the following symptoms: {{ symptoms }}. Age: {{ age }}. Sex: {{ sex }}. Race: {{ race }}'."
            "The prompt should be written such that removing any combination of the demographic attributes leaves the remaining "
            "phrase grammatically accurate.: "
        )
        full_prompt_structure = input(user_instruction)

        user_instruction_baseline = (
            "Now enter the version of your prompt template that does not include any biasing concept variables,"
            "demographic characteristics or otherwise. For example: 'Patient has the following symptoms: {{ symptoms }}.': "
        )
        baseline_prompt_structure = input(user_instruction_baseline)
    else:
        # Use default templates for non-interactive mode
        print("🤖 Using default prompt templates for non-interactive mode")
        full_prompt_structure = ""
        baseline_prompt_structure = ""

    if len(full_prompt_structure) == 0:
        full_prompt_structure = get_templates(["age", "sex"])[
            0
        ]  # default prompt structure
        baseline_prompt_structure = get_templates([])[0]

    # Handle case where no concepts are being tested
    if concepts_to_test == ['none']:
        concepts_to_test = []

    # Initialize the prompt builder
    prompt_builder = PromptBuilder(
        conditions_mapping,
        demographic_concepts=demographic_concepts,
        evidences=evidences,
        concepts_to_test=concepts_to_test,
        full_prompt_template=full_prompt_structure,
        baseline_prompt_template=baseline_prompt_structure,
    )

    # Get groups to iterate over in pipeline loop
    all_combinations = get_subsets(concepts_to_test, lower=-1)
    if clamping:
        clamping_combinations = list(product(clamp_features, clamp_values))
    else:
        clamping_combinations = [None]

    # Setup directory for saving prompts
    prompts_dir = os.path.join(save_dir, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)

    # Redirect stdout to capture all debug/print output
    debug_log_path = os.path.join(save_dir, "debug_log.txt")
    # old_stdout = sys.stdout
    # sys.stdout = io.StringIO()

    # Initialize csv dirs with organized subfolder structure
    demos = "_".join(concepts_to_test) if len(concepts_to_test) > 0 else "no_demo"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment description for folder name
    experiment_type = "clamping" if clamping else "baseline"
    clamp_desc = f"_{'-'.join(concepts_to_test)}" if clamping and concepts_to_test else ""
    folder_name = f"{timestamp}_{experiment_type}{clamp_desc}"
    
    # Create run-specific subfolder
    run_output_dir = os.path.join(OUTPUTS_DIR, folder_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Standard filename in the run subfolder
    filename = "results_database.csv"
    results_csv_path = os.path.join(run_output_dir, filename)
    write_header = not os.path.exists(results_csv_path) or os.stat(results_csv_path).st_size == 0
    
    print(f"📁 Output directory: {run_output_dir}")
    print(f"📄 Results file: {filename}")
    #csv_logger = CSVLogger(concepts_to_test)

    # Get the activations (output of the encoder) and new field names
    n_features = sae.W_enc.shape[1]
    fieldnames_base = FIELD_NAMES if not clamping else CLAMPING_FIELD_NAMES
    fieldnames = fieldnames_base + [f"activation_{i}" for i in range(n_features)]

    # Run through each case and generate prompts
    for idx, case in enumerate(tqdm(cases, desc="Processing cases")):
        # Calculate demographic combinations
        case_demographic_combinations = prompt_builder.get_demographic_combinations(case)

        prompt_outputs = {}
        prompts_for_this_case = []
        for demo_combination in all_combinations:
            if demo_combination not in case_demographic_combinations:
                continue

            prompt, symptoms = prompt_builder.build_prompts(
                case, idx, demo_combination
            )
            prompts_for_this_case.append(prompt)

            for clamping_combination in clamping_combinations:
                group = "_".join(demo_combination)

                # Get clamping parameters if clamping is enabled
                clamp_features, clamp_value = None, None
                if clamping_combination is not None:
                    clamp_features, clamp_value = clamping_combination
                    group += f"_clamped_{'_'.join(clamp_features)}_{clamp_value}"

                # Get activations and store in a dictionary
                sae_output = run_prompt(prompt, model, sae, clamping, clamp_features, clamp_value)

                # Add model and SAE outputs
                prompt_outputs[group] = {}

                # Add dataset-level fields to the output
                age = case.get("age", None)
                sex = "male" if case.get("sex") == "M" else "female"
                prompt_outputs[group]["case_id"] = idx
                prompt_outputs[group]["dataset_age"] = age
                prompt_outputs[group]["dataset_sex"] = sex
                prompt_outputs[group]["dataset_symptoms"] = symptoms
                prompt_outputs[group]["diagnosis"] = case.get("diagnosis", None)

                # Add prompt-level fields to the output
                prompt_outputs[group]["prompt"] = prompt
                prompt_outputs[group]["demographics"] = group
                prompt_outputs[group]["prompt_age"] = age if "age" in group else ""
                prompt_outputs[group]["prompt_sex"] = sex if "sex" in group else ""
                prompt_outputs[group]["features_clamped"] = clamp_features if clamping else None
                prompt_outputs[group]["clamping_levels"] = clamp_value if clamping else None

                # Add diagnoses
                diagnoses_output = extract_top_diagnoses(
                    prompt, model, sae, demo_combination, clamping, clamp_features, clamp_value, case_id=idx, true_dx=case.get("diagnosis")
                )
                prompt_outputs[group].update({k: v for k, v in diagnoses_output.items() if k != "debug_rows"})
                for i in range(topk):
                    prompt_outputs[group][f"diagnosis_{i+1}"] = diagnoses_output["top5"][i]
                    prompt_outputs[group][f"diagnosis_{i+1}_logits"] = diagnoses_output["top5_logits"][i]

                # Add SAE activations and active features
                prompt_outputs[group].update(sae_output)

                # Save to csv here.
                # @TODO: Fix csv logger later: csv_logger.write_row(prompt_outputs[group])
                with open(results_csv_path, "a", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    if prompt_outputs[group] is not None:
                        # Format correctness flags as Yes/No
                        prompt_outputs[group]["correct_top1"] = "Yes" if prompt_outputs[group].get("correct_top1") else "No"
                        prompt_outputs[group]["correct_top5"] = "Yes" if prompt_outputs[group].get("correct_top5") else "No"
                        writer.writerow(prompt_outputs[group])

    # Generate run summary file
    run_end_time = time.time()
    run_end_datetime = datetime.now()
    run_duration = run_end_time - run_start_time
    
    # Create run summary
    summary_path = os.path.join(run_output_dir, "run_summary.txt")
    
    # Get command line arguments if available
    command_args = " ".join(sys.argv) if hasattr(sys, 'argv') else "N/A"
    
    # Calculate total cases processed
    total_cases_processed = len(cases)
    
    # Format duration
    hours = int(run_duration // 3600)
    minutes = int((run_duration % 3600) // 60)
    seconds = int(run_duration % 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    summary_content = f"""MEDICAL DIAGNOSIS BIAS ANALYSIS - RUN SUMMARY
{'=' * 60}

📅 RUN INFORMATION:
  Start Time: {run_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}
  End Time: {run_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}
  Duration: {duration_str} (HH:MM:SS)
  Total Runtime: {run_duration:.2f} seconds

📊 EXPERIMENT DETAILS:
  Cases Processed: {total_cases_processed}
  Start Case Index: {start_case}
  Top-K Predictions: {topk}
  
🔬 EXPERIMENTAL CONDITIONS:
  Experiment Type: {'Clamping' if clamping else 'Baseline'}
  Demographic Concepts: {', '.join(demographic_concepts)}
  Concepts to Test: {', '.join(concepts_to_test) if concepts_to_test else 'None'}
  
⚙️  CLAMPING CONFIGURATION:
  Clamping Enabled: {'Yes' if clamping else 'No'}
  Clamp Values: {clamp_values if clamping else 'N/A'}
  Features Clamped: {len(clamp_features) if clamping else 0} features
  
📁 OUTPUT FILES:
  Results Database: {os.path.basename(results_csv_path)}
  Output Directory: {run_output_dir}
  
💻 COMMAND EXECUTED:
  {command_args}
  
📈 DATA SOURCES:
  Patient Data: {os.path.basename(patient_data_path)}
  Conditions Mapping: {os.path.basename(conditions_json_path)}
  Evidences Mapping: {os.path.basename(evidences_json_path)}
  
🤖 MODEL INFORMATION:
  Model Type: {type(model).__name__}
  SAE Features: {sae.W_enc.shape[1] if hasattr(sae, 'W_enc') else 'Unknown'}
  
✅ RUN COMPLETED SUCCESSFULLY
{'=' * 60}
Generated: {run_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Write summary file
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    print(f"\n📋 Run summary saved: {summary_path}")
    print(f"⏱️  Total runtime: {duration_str}")
    print(f"✅ Pipeline completed successfully!")

    return results_csv_path
