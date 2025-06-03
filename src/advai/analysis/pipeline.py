import os
import json
import datetime
from tqdm import tqdm
from src.advai.data.io import load_patient_data, extract_cases_from_dataframe, load_conditions_mapping
from src.advai.data.prompt_builder import PromptBuilder
from src.advai.analysis.analyse import run_prompt, compare_activations
from src.advai.visuals.plots import visualize_feature_overlaps
from src.advai.analysis.summary import generate_summary, write_output

def run_analysis_pipeline(patient_data_path, conditions_json_path, model, sae, num_cases=None, save_dir="activations", output_name=None):
    """Run the full analysis pipeline including loading data, generating prompts,
    analyzing bias, and writing results to disk.
    """
    # Setup outputs directory at project level
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    outputs_dir = os.path.join(project_root, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    df = load_patient_data(patient_data_path)
    cases = extract_cases_from_dataframe(df)
    if num_cases:
        cases = cases[:num_cases]

    conditions_mapping = load_conditions_mapping(conditions_json_path)

    # Initialize the prompt builder
    prompt_builder = PromptBuilder(
        conditions_mapping, 
        demographic_concepts=["age", "sex"],
        concepts_to_test=["sex"]
    )

    results = []
    case_summaries = []
    activation_diff_by_sex = {}
    activation_diff_by_diagnosis = {}

    for idx, case in enumerate(tqdm(cases, desc="Processing cases")):

        text_with_demo, text_without_demo = prompt_builder.build_prompts(case)

        activations_1 = run_prompt(text_with_demo, model, sae)
        activations_2 = run_prompt(text_without_demo, model, sae)
        case_result = compare_activations(activations_1, activations_2,case_id=idx, threshold=1.0)
        
        results.append(case_result)
        case_summaries.append(str(case_result))

    # Construct timestamped filenames using output_name
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{output_name or 'analysis'}_{now}"
    feature_path = os.path.join(outputs_dir, f"{base_name}_feature_overlap.html")
    analysis_path = os.path.join(outputs_dir, f"{base_name}_analysis_output.txt")
    visualize_feature_overlaps(results, save_path=feature_path)
    summary_text = generate_summary(results, case_summaries, activation_diff_by_sex, activation_diff_by_diagnosis)
    write_output(analysis_path, case_summaries, summary_text)
    return analysis_path
