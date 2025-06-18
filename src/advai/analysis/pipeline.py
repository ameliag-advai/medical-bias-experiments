"""This module provides a pipeline for analyzing bias in AI models for a given dataset."""
import csv
import datetime
import io
import json
import os
import sys
from tqdm import tqdm
from typing import Tuple

from src.advai.analysis.analyse import (
    run_prompt,
    compile_results,
    extract_top_diagnoses,
)
from src.advai.analysis.summary import generate_summary, write_output
from src.advai.data.io import (
    extract_cases_from_dataframe,
    load_conditions_mapping,
    load_patient_data,
)
from src.advai.data.example_templates import TEMPLATE_SETS
from src.advai.data.prompt_builder import PromptBuilder, get_subsets
from src.advai.visuals.plots import visualize_feature_overlaps

# Setup global outputs directory and timestamped CSV path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
#PROJECT_ROOT = "/mnt/advai_scratch/shared/alethia"
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
FIELD_NAMES = [
        "case_id",
        "dataset_age",
        "dataset_sex",
        "dataset_symptoms",
        "diagnosis",
        "prompt",
        "prompt_age",
        "prompt_sex",
        "features_clamped",
        "clamping_levels",
        "diagnosis_1",
        "diagnosis_2",
        "diagnosis_3",
        "diagnosis_4",
        "diagnosis_5",
        "diagnosis_1_logits",
        "diagnosis_2_logits",
        "diagnosis_3_logits",
        "diagnosis_4_logits",
        "diagnosis_5_logits",
        "activations",
        "active_features",
        "n_active_features",
        "top5",
        "top5_logits",
    ]

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
        cases = cases[start_case : num_cases]
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
    demographic_concepts: list[str] = ["age", "sex"],
    concepts_to_test: list[str] = ["age", "sex"],
    save_dir: str = "activations",
    output_name: str = None,
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
    # Get the patient data and conditions mapping
    cases, conditions_mapping, evidences = data_preprocessing(
        patient_data_path,
        conditions_json_path,
        evidences_json_path,
        num_cases=num_cases,
        start_case=start_case,
    )

    # Ask user to enter a prompt structure or use a default one
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

    if len(full_prompt_structure) == 0:
        full_prompt_structure = get_templates(demographic_concepts)[
            0
        ]  # default prompt structure
        baseline_prompt_structure = get_templates([])[0]

    # Initialize the prompt builder
    prompt_builder = PromptBuilder(
        conditions_mapping,
        demographic_concepts=demographic_concepts,
        evidences=evidences,
        concepts_to_test=concepts_to_test,
        full_prompt_template=full_prompt_structure,
        baseline_prompt_template=baseline_prompt_structure,
    )
    all_combinations = get_subsets(concepts_to_test, lower=-1)
    pairs_to_compare = [
        ("_".join(all_combinations[i]), "_".join(all_combinations[-1]))
        for i in range(len(all_combinations) - 1)
    ]

    results = []
    case_summaries = []

    # Setup directory for saving prompts
    prompts_dir = os.path.join(save_dir, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    all_prompts_text = []
    all_prompts_outputs = {}

    # Redirect stdout to capture all debug/print output
    debug_log_path = os.path.join(save_dir, "debug_log.txt")
    # old_stdout = sys.stdout
    # sys.stdout = io.StringIO()
    # print(f"[INFO] Saving all prompts for each case in: {prompts_dir}")
    # print(f"[INFO] Saving master prompt file at: {os.path.join(prompts_dir, 'all_prompts.txt')}")
    # print(f"[INFO] Visualization will be saved as: feature_overlap.html in {os.getcwd()}")

    # Initialize csv dirs
    demos = "_".join(concepts_to_test) if len(concepts_to_test) > 0 else "no_demo"
    os.makedirs(OUTPUTS_DIR + f"/{RUN_TIMESTAMP}_{demos}", exist_ok=True)
    results_csv_base_bath = f"{RUN_TIMESTAMP}_{demos}/results_database.csv"
    activations_base_path = f"{RUN_TIMESTAMP}_{demos}/activations.json"
    results_csv_path = os.path.join(OUTPUTS_DIR, results_csv_base_bath)
    activations_path = os.path.join(OUTPUTS_DIR, activations_base_path)
    write_header = not os.path.exists(results_csv_path) or os.stat(results_csv_path).st_size == 0
    #csv_logger = CSVLogger(concepts_to_test)

    # Run through each case and generate prompts
    for idx, case in enumerate(tqdm(cases, desc="Processing cases")):
        # Calculate demographic combinations
        case_demographic_combinations = prompt_builder.get_demographic_combinations(
            case
        )

        prompt_outputs = {}
        prompts_for_this_case = []
        for demo_combination in all_combinations:
            if demo_combination in case_demographic_combinations:
                group = "_".join(demo_combination)
                prompt, symptoms = prompt_builder.build_prompts(
                    case, idx, demo_combination
                )
                prompts_for_this_case.append(prompt)

                # Get activations and store in a dictionary
                sae_output = run_prompt(prompt, model, sae)
                diagnoses_output = extract_top_diagnoses(
                    prompt, model, demo_combination, case_id=idx
                )

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
                prompt_outputs[group]["prompt_age"] = age if "age" in group else ""
                prompt_outputs[group]["prompt_sex"] = sex if "sex" in group else ""
                prompt_outputs[group]["features_clamped"] = None
                prompt_outputs[group]["clamping_levels"] = None
                for i in range(topk):
                    prompt_outputs[group][f"diagnosis_{i+1}"] = diagnoses_output[
                        "top5"
                    ][i]
                    prompt_outputs[group][f"diagnosis_{i+1}_logits"] = diagnoses_output["top5_logits"][i]

                # Add SAE activations and active features
                prompt_outputs[group].update(sae_output)
                prompt_outputs[group].update({k: v for k, v in diagnoses_output.items() if k != "debug_rows"})

            # If this combination is not in the case, set to None
            else:
                prompt_outputs[group] = None

            # Save to csv here.
            #csv_logger.write_row(prompt_outputs[group])
            # @TODO: Fix csv logger later: csv_logger.write_row(prompt_outputs[group])
            with open(results_csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=FIELD_NAMES)
                if write_header:
                    writer.writeheader()
                    write_header = False
                if prompt_outputs[group] is not None:
                    writer.writerow(prompt_outputs[group])

        #all_prompts_outputs[idx] = prompt_outputs
        
        # Now compare relevant pairs of activations for this case
        # @TODO: Move this to analysis capability
        # case_result = process_case_result(prompt_outputs, pairs_to_compare, case_id=idx, case_info=case)
        # results.append(case_result)
        # case_summaries.append(str(case_result))

    # Close the CSV file after writing all results
    #csv_logger.close()
    #Save all_prompts_outputs to json
    #with open(activations_path, "w", encoding="utf-8") as f:
    #    json.dump(all_prompts_outputs, f, indent=4)

    # Write debug log
    # debug_out = sys.stdout.getvalue()
    # with open(debug_log_path, "w", encoding="utf-8") as dbg:
    #     dbg.write(debug_out)
    # sys.stdout = old_stdout
    # print(f"[INFO] Debug log for this run written to: {debug_log_path}")

    # Generate results summaries and visualizations
    # summary_text = generate_summary(results, pairs_to_compare)
    # visualize_feature_overlaps(results, pairs_to_compare, save_path=feature_path)
    # write_output(analysis_path, case_summaries, summary_text)

    return results_csv_path
