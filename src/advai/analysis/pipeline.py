import os
import json
import datetime
from tqdm import tqdm
from src.advai.data.io import load_patient_data, extract_cases_from_dataframe, load_conditions_mapping

from src.advai.data.build_prompts import build_prompts
from src.advai.analysis.analyse import analyse_case_for_bias
from src.advai.visuals.plots import visualize_feature_overlaps
from src.advai.analysis.summary import generate_summary, write_output
import sys
import io

def run_analysis_pipeline(
    patient_data_path,
    conditions_json_path,
    model,
    sae,
    num_cases=None,
    save_dir="activations",
    output_path=None
):
    """
    Orchestrate the full analysis pipeline.
    Args:
        patient_data_path: Path to patient data CSV
        conditions_json_path: Path to the conditions JSON
        model: Loaded model
        sae: Loaded SAE
        num_cases: Number of cases to process (optional)
        save_dir: Directory to save activations
        output_path: Path to write output (optional)
    Returns:
        Tuple (output_path, results, case_summaries, summary_text)
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
    overlap_lists = []

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
    for idx, case in enumerate(tqdm(cases, desc="Processing cases")):



        text_with_demo, text_without_demo = prompt_builder.build_prompts(case)

        activations_1 = run_prompt(text_with_demo, model, sae)
        activations_2 = run_prompt(text_without_demo, model, sae)
        case_result = compare_activations(activations_1, activations_2,case_id=idx, threshold=1.0)
        
        results.append(case_result)
        case_summaries.append(str(case_result))


    visualize_feature_overlaps(results, save_path="feature_overlap.html")
    # Write debug log
    debug_out = sys.stdout.getvalue()
    with open(debug_log_path, "w", encoding="utf-8") as dbg:
        dbg.write(debug_out)
    sys.stdout = old_stdout
    print(f"[INFO] Debug log for this run written to: {debug_log_path}")

    summary_text = generate_summary(results, case_summaries, activation_diff_by_sex, activation_diff_by_diagnosis)
    if output_path is None:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(os.path.dirname(__file__), "..", f"analysis_output_{now}.txt")
    write_output(output_path, case_summaries, summary_text)

    return output_path
