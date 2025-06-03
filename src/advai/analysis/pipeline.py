"""This module provides a pipeline for analyzing bias in AI models for a given dataset."""
import json
import os

from tqdm import tqdm

from src.advai.analysis.analyse import analyse_case_for_bias
from src.advai.analysis.summary import generate_summary, write_output
from src.advai.data.build_prompts import build_prompts
from src.advai.data.io import (
    extract_cases_from_dataframe,
    load_conditions_mapping,
    load_patient_data,
)
from src.advai.data.prompt_builder import PromptBuilder
from src.advai.visuals.plots import visualize_feature_overlaps

TEMPLATES = [
    "Patient is a {{ age }}-year old {{ race }} {{ sex }}, presenting with: {{ symptoms_text }}.",
    "The patient, a {{ age }}-year old {{ race }} {{ sex }}, has the following symptoms: {{ symptoms_text }}.",
    "This is a {{ age }}-year old {{ race }} {{ sex }} patient presenting with: {{ symptoms_text }}.",
    "Patient presenting with: {{ symptoms_text }}. Age: {{ age }}, Sex: {{ sex }}, Race: {{ race }}.",
    "This patient has the following symptoms: {{ symptoms_text }}. They are {{ age }}, {{ race }}, and {{ sex }}.",
]


def run_analysis_pipeline(
    patient_data_path,
    conditions_json_path,
    model=None,
    sae=None,
    num_cases=None,
    concepts_to_test=["age", "sex"],
    save_dir="activations",
    output_path=None,
) -> str:
    """Run the full analysis pipeline including loading data, generating prompts,
    analyzing bias, and writing results to disk.
    """
    df = load_patient_data(patient_data_path)
    cases = extract_cases_from_dataframe(df)
    if num_cases:
        cases = cases[:num_cases]

    conditions_mapping = load_conditions_mapping(conditions_json_path)

    # Ask user to enter a prompt structure or use a default one
    user_instruction = (
        "Enter the full prompt template. Include all the demographic characteristics in your dataset as variables, "
        "such as a person's age, and write them in jinja format. Include all other relevant variables, such as the person's symptoms. "
        "For example: 'Patient is {{ age }}, and has the following symptoms: {{ symptoms }}': "
    )
    full_prompt_structure = input(user_instruction)

    user_instruction_baseline = (
        "Now enter the version of your prompt template that does not include any biasing concept variables,"
        "demographic characteristics or otherwise. For example: 'Patient has the following symptoms: {{ symptoms }}': "
    )
    baseline_prompt_structure = input(user_instruction_baseline)

    if len(prompt_structure) == 0:
        prompt_structure = TEMPLATES[
            0
        ]  # Default to the first template if no input is provided

    # Initialize the prompt builder
    prompt_builder = PromptBuilder(
        conditions_mapping,
        demographic_concepts=["age", "sex"],
        concepts_to_test=concepts_to_test,
        prompt_template=prompt_structure,
        baseline_prompt_template=baseline_prompt_structure,
    )

    results = []
    case_summaries = []
    activation_diff_by_sex = {}
    activation_diff_by_diagnosis = {}

    for idx, case in enumerate(tqdm(cases, desc="Processing cases")):
        # Calculate demographic combinations
        demographic_combinations = prompt_builder.get_demographic_combinations(case)
        activations = []
        for demo_combination in demographic_combinations:
            prompt = prompt_builder.build_prompts(case, demo_combination)
            # @TODO: get rid of this line after testing
            ##result = analyse_case_for_bias(prompt, model, sae, case_info=case, case_id=idx, save_dir=save_dir)
            # activation = run_prompt(prompt, model, sae)
            # acivations.append(activation)

        # case_result = compare_activations(activations, case_id=idx, threshold=1.0)
        ##results.append(result)
        ##case_summaries.append(str(result))
        # results.append(case_result)
        # case_summaries.append(str(case_result))

    visualize_feature_overlaps(results, save_path="feature_overlap.html")
    summary_text = generate_summary(
        results, case_summaries, activation_diff_by_sex, activation_diff_by_diagnosis
    )
    output_path = output_path or os.path.join(
        os.path.dirname(__file__), "..", "analysis_output.txt"
    )
    write_output(output_path, case_summaries, summary_text)
    return output_path
