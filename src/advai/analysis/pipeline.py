"""This module provides a pipeline for analyzing bias in AI models for a given dataset."""
import datetime
import io
import os
import sys

from tqdm import tqdm

from src.advai.analysis.analyse import run_prompt, compile_results, extract_top_diagnoses
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
    """Load and preprocess patient data and conditions mapping.
    
    :param patient_data_path: Path to the patient data CSV file.
    :param conditions_json_path: Path to the conditions mapping JSON file.
    :param num_cases: Number of cases to analyze from the dataset.
    :return: A tuple containing the cases and conditions mapping.
    """
    df = load_patient_data(patient_data_path)
    cases = extract_cases_from_dataframe(df)
    if num_cases:
        cases = cases[:num_cases]
    conditions_mapping = load_conditions_mapping(conditions_json_path)

    return cases, conditions_mapping


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
            case_result[pair] = compile_results(prompt_output_1, prompt_output_2, pair, case_id=case_id, case_info=case_info)
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

    # Setup directory for saving prompts
    prompts_dir = os.path.join(save_dir, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    all_prompts_text = []
  
    # Redirect stdout to capture all debug/print output
    debug_log_path = os.path.join(save_dir, "debug_log.txt")
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    print(f"[INFO] Saving all prompts for each case in: {prompts_dir}")
    print(f"[INFO] Saving master prompt file at: {os.path.join(prompts_dir, 'all_prompts.txt')}")
    print(f"[INFO] Visualization will be saved as: feature_overlap.html in {os.getcwd()}")

    # Run through each case and generate prompts
    for idx, case in enumerate(tqdm(cases, desc="Processing cases")):
        # Calculate demographic combinations
        case_demographic_combinations = prompt_builder.get_demographic_combinations(case)

        # Save prompts for this case
        prompt_file = os.path.join(prompts_dir, f"case_{idx}_prompts.txt")
        
        prompt_outputs = {}
        prompts_for_this_case = []
        for demo_combination in all_combinations:
            if demo_combination in case_demographic_combinations:
                prompt = prompt_builder.build_prompts(case, demo_combination)
                prompts_for_this_case.append(prompt)
                
                # Get activations and store in a dictionary
                sae_output = run_prompt(prompt, model, sae)
                diagnoses_output = extract_top_diagnoses(prompt, model, demo_combination, case_id=idx)
                prompt_outputs["_".join(demo_combination)] = {**sae_output, **diagnoses_output}
            
            # If this combination is not in the case, set to None
            else:
                prompt_outputs["_".join(demo_combination)] = None

        # Now compare relevant pairs of activations for this case
        case_result = process_case_result(prompt_outputs, pairs_to_compare, case_id=idx, case_info=case)
        results.append(case_result)
        case_summaries.append(str(case_result))
        
        # Save all prompts for this case
        case_prompts = f"CASE {idx}\n"
        with open(prompt_file, "w", encoding="utf-8") as f:
            for demo_combination, prompt in zip(case_demographic_combinations, prompts_for_this_case):
                f.write(f"Demographic combination: {demo_combination}\nPrompt: {prompt}\n")
                case_prompts = case_prompts + f"Demographic combination: {demo_combination}\nPrompt: {prompt}\n"
        all_prompts_text.append(case_prompts)

    # Save all prompts as a master text file
    master_prompt_file = os.path.join(prompts_dir, "all_prompts.txt")
    with open(master_prompt_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_prompts_text))
    
    # Write debug log
    debug_out = sys.stdout.getvalue()
    with open(debug_log_path, "w", encoding="utf-8") as dbg:
        dbg.write(debug_out)
    sys.stdout = old_stdout
    print(f"[INFO] Debug log for this run written to: {debug_log_path}")

    # Construct timestamped filenames using output_name
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{output_name or 'analysis'}_{now}"
    feature_path = os.path.join(outputs_dir, f"{base_name}_feature_overlap")
    analysis_path = os.path.join(outputs_dir, f"{base_name}_analysis_output.txt")

    # Generate results summaries and visualizations
    summary_text = generate_summary(results, pairs_to_compare)
    visualize_feature_overlaps(results, pairs_to_compare, save_path=feature_path)
    write_output(analysis_path, case_summaries, summary_text)

    return analysis_path
