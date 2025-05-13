import os
import json
from tqdm import tqdm
from src.advai.data.io import load_patient_data, extract_cases_from_dataframe, load_conditions_mapping
from src.advai.data.build_prompts import build_prompts
from src.advai.analysis.analyse import analyse_case_for_bias
from src.advai.visuals.plots import visualize_feature_overlaps
from src.advai.analysis.summary import generate_summary, write_output

def run_analysis_pipeline(patient_data_path, conditions_json_path, model, sae, num_cases=None, save_dir="activations", output_path=None):
    """Run the full analysis pipeline including loading data, generating prompts,
    analyzing bias, and writing results to disk.
    """


    df = load_patient_data(patient_data_path)
    cases = extract_cases_from_dataframe(df)
    if num_cases:
        cases = cases[:num_cases]

    conditions_mapping = load_conditions_mapping(conditions_json_path)

    results = []
    case_summaries = []
    activation_diff_by_sex = {}
    activation_diff_by_diagnosis = {}

    for idx, case in enumerate(tqdm(cases, desc="Processing cases")):
        text_with_demo, text_without_demo = build_prompts(case, conditions_mapping)
        result = analyse_case_for_bias(text_with_demo, text_without_demo, model, sae, case_info=case, case_id=idx, save_dir=save_dir)
        results.append(result)
        case_summaries.append(str(result))

    visualize_feature_overlaps(results, save_path="feature_overlap.html")
    summary_text = generate_summary(results, case_summaries, activation_diff_by_sex, activation_diff_by_diagnosis)
    output_path = output_path or os.path.join(os.path.dirname(__file__), "..", "analysis_output.txt")
    write_output(output_path, case_summaries, summary_text)
    return output_path