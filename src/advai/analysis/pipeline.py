"""This module provides a pipeline for analyzing bias in AI models for a given dataset."""
import datetime
import json
import os

from tqdm import tqdm

from src.advai.analysis.analyse import run_prompt, compare_activations
from src.advai.analysis.summary import generate_summary, write_output
from src.advai.data.io import (
    extract_cases_from_dataframe,
    load_conditions_mapping,
    load_patient_data,
)
from src.advai.data.example_templates import TEMPLATE_SETS
from src.advai.data.prompt_builder import PromptBuilder, get_subsets
from src.advai.visuals.plots import visualize_feature_overlaps


def get_templates(demographic_concepts: list[str]):
    key = frozenset(demographic_concepts)
    return TEMPLATE_SETS.get(key)


def data_preprocessing(
    patient_data_path: str, conditions_json_path: str, num_cases: int = 1
):
    """Load and preprocess patient data and conditions mapping."""
    df = load_patient_data(patient_data_path)
    cases = extract_cases_from_dataframe(df)
    if num_cases:
        cases = cases[:num_cases]
    conditions_mapping = load_conditions_mapping(conditions_json_path)

    return cases, conditions_mapping


def process_case_result(activations, pairs_to_compare, case_id=None, threshold=1.0):
    """Process the activations and compare them for each pair."""
    case_result = {}
    for pair in pairs_to_compare:
        activations_1 = activations[pair[0]]
        activations_2 = activations[pair[1]]
        if activations_1 is None or activations_2 is None:
            case_result[pair] = None
        else:
            case_result[pair] = compare_activations(activations_1, activations_2, case_id=case_id, threshold=threshold)
    return case_result


def run_analysis_pipeline(
    patient_data_path,
    conditions_json_path,
    model,
    sae,
    num_cases: int = 1,
    demographic_concepts: list[str] = ["age", "sex"],
    concepts_to_test: list[str] = ["age", "sex"],
    save_dir: str = "activations",
    output_name: str = None,
) -> str:
    """Run the full analysis pipeline including loading data, generating prompts,
    analyzing bias, and writing results to disk.
    """
    # Setup outputs directory at project level
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Get the patient data and conditions mapping
    cases, conditions_mapping = data_preprocessing(
        patient_data_path, conditions_json_path, num_cases=num_cases
    )

    # Ask user to enter a prompt structure or use a default one
    user_instruction = (
        "Enter the full prompt template. Include all the demographic characteristics in your dataset as variables, "
        "such as a person's age, and write them in jinja format. Include all other relevant variables, such as the person's symptoms. "
        "For example: 'Patient has the following symptoms: {{ symptoms }}. Age: {{ age }}. Sex: {{ sex }}. Race: {{ race }}'."
        "The prompt should be written such that removing any combination of the demographic attributes leaves the remaining"
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
        concepts_to_test=concepts_to_test,
        full_prompt_template=full_prompt_structure,
        baseline_prompt_template=baseline_prompt_structure,
    )
    all_combinations = get_subsets(concepts_to_test, lower=-1)
    pairs_to_compare = [("_".join(all_combinations[i]), "_".join(all_combinations[-1])) 
                        for i in range(len(all_combinations) - 1)]

    results = []
    case_summaries = []
    for idx, case in enumerate(tqdm(cases, desc="Processing cases")):
        # Calculate demographic combinations
        case_demographic_combinations = prompt_builder.get_demographic_combinations(case)
        activations = {}
        for demo_combination in all_combinations: #demographic_combinations:
            if demo_combination in case_demographic_combinations:
                prompt = prompt_builder.build_prompts(case, demo_combination)
                activation = run_prompt(prompt, model, sae)
                activations["_".join(demo_combination)] = activation
            # If this combination is not in the case, set to None
            else:
                activations["_".join(demo_combination)] = None

        case_result = process_case_result(activations, pairs_to_compare, case_id=idx, threshold=1.0)
        results.append(case_result)
        case_summaries.append(str(case_result))

    # Construct timestamped filenames using output_name
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{output_name or 'analysis'}_{now}"
    feature_path = os.path.join(outputs_dir, f"{base_name}_feature_overlap")
    analysis_path = os.path.join(outputs_dir, f"{base_name}_analysis_output.txt")
    visualize_feature_overlaps(results, pairs_to_compare, save_path=feature_path)

    # Generate results summaries
    summary_text = generate_summary(results, pairs_to_compare)
    write_output(analysis_path, case_summaries, summary_text)

    return analysis_path
