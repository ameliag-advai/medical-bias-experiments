import torch
import csv
import os
import json
import logging
from typing import Dict, Any, Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def load_diagnosis_list(json_path) -> List[str]:
    """Load the list of diagnoses from a JSON file.

    :param json_path: Path to the JSON file containing diagnoses.
    :return: List of condition names.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return [v["condition_name"] for v in data.values()]


def debug_info_to_csv(debug_rows):
    """Save debug information to a CSV file for post-analysis."""
    with open("diagnosis_logit_debug.csv", "a", newline="") as csvfile:
        # Include all fields that are being added to the rows
        fieldnames = ["case_id", "group", "candidate", "log_probs", "raw_logits", "correct", "correct_top1", "correct_top5"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is empty
        if csvfile.tell() == 0:
            writer.writeheader()
        
        for row in debug_rows:
            # Convert lists to strings for CSV writing
            row["log_probs"] = str(row["log_probs"])
            row["raw_logits"] = str(row["raw_logits"])
            row["correct"] = str(row.get("correct", False))
            row["correct_top1"] = str(row.get("correct_top1", False))
            row["correct_top5"] = str(row.get("correct_top5", False))
            
            # Filter row to only include fields in fieldnames
            filtered_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(filtered_row)


def summary_info_to_csv(summary_rows):
    """Save summary information to a CSV file for post-analysis."""
    with open("diagnosis_summary.csv", "a", newline="") as csvfile:
        fieldnames = ["case_id", "group", "top1", "top5", "correct_top1", "correct_top5"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is empty
        if csvfile.tell() == 0:
            writer.writeheader()
        
        for row in summary_rows:
            # Convert lists to strings
            row["top5"] = str(row["top5"])
            row["correct_top1"] = str(row.get("correct_top1", False))
            row["correct_top5"] = str(row.get("correct_top5", False))
            
            # Filter row to only include fields in fieldnames
            filtered_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(filtered_row)


def score_candidate(prompt_prefix: str, candidate: str, model, baseline_prefix: str = "", alpha: float = 1.0, length_penalty: float = 0.0) -> Tuple[float, List[float], List[float]]:
    """
    Score a candidate diagnosis by computing its log-probability under the model.

    :param prompt_prefix: Text prompt without diagnosis.
    :param candidate: Diagnosis string.
    :param model: The language model.
    :return: Total log-probability, list of per-token log-probs, list of raw logits.
    """
    # Full prompt with and without diagnosis
    prompt_with_candidate = f"{prompt_prefix} Diagnosis is: {candidate} "
    prompt_without_candidate = f"{prompt_prefix} Diagnosis is: "

    # Tokenise both
    toks_full = model.to_tokens(prompt_with_candidate)
    toks_prefix = model.to_tokens(prompt_without_candidate)

    # Adjust prefix length to exclude trailing shared token (e.g., whitespace or EOS)
    true_prefix_len = toks_prefix.shape[-1] - 1

    # Diagnosis tokens are those after the true prefix
    diagnosis_token_ids = toks_full[0, true_prefix_len:-1]

    # Run model
    logits, _ = model.run_with_cache(toks_full)

    log_probs = []
    raw_logits = []

    for i, token_id in enumerate(diagnosis_token_ids):
        token_id = int(token_id)
        pos = true_prefix_len - 1 + i  # predict token i from this position
        logit = logits[0, pos, :]
        log_prob = torch.nn.functional.log_softmax(logit, dim=-1)[token_id]

        log_probs.append(log_prob.item())
        raw_logits.append(logit[token_id].item())
    
    logger.debug(f"[LOGITS] Candidate: '{candidate}' | Log probs (scalar): {log_probs} | Raw logits: {raw_logits}")
    
    # Compute base average log-prob (average per token)
    score = sum(log_probs) / len(log_probs) if log_probs else float('-inf')

    # Baseline correction (subtract model prior on candidate)
    if baseline_prefix:
        bpref = f"{baseline_prefix} Diagnosis is: "
        bfull = f"{bpref}{candidate} "
        toks_bfull = model.to_tokens(bfull)
        toks_bpref = model.to_tokens(bpref)
        bpref_len = toks_bpref.shape[-1] - 1
        diag_bids = toks_bfull[0, bpref_len:-1]
        logits_b, _ = model.run_with_cache(toks_bfull)
        base_lp = []
        for i, tid in enumerate(diag_bids):
            tid = int(tid)
            pos = bpref_len - 1 + i
            logit_b = logits_b[0, pos, :]
            lp_b = torch.nn.functional.log_softmax(logit_b, dim=-1)[tid]
            base_lp.append(lp_b.item())
        avg_base = sum(base_lp) / len(base_lp)
        score -= avg_base

    # Length normalization and penalty
    if alpha != 1.0 and log_probs:
        score = sum(log_probs) / (len(log_probs) ** alpha)
    if length_penalty and log_probs:
        score -= length_penalty * (1 / len(log_probs))

    return score, log_probs, raw_logits


def score_diagnoses(
    prompt, group, diagnosis_list, model, case_id, debug_rows=None
) -> Tuple[List, List]:
    """Score a list of candidate diagnoses and return their scores along with debug information.

    :param prompt: The prompt to be used for scoring diagnoses.
    :param group: The demographic group for which the diagnoses are being scored.
    :param diagnosis_list: List of candidate diagnoses to score.
    :param model: The model used to score the diagnoses.
    :param case_id: Optional case identifier for logging/debugging.
    :param debug_rows: Optional list to collect debug information.
    :return: A list of tuples containing diagnosis and its score, and the debug rows.
    """
    if debug_rows is None:
        debug_rows = []
    dx_scores = []
    for dx in diagnosis_list:
        score, log_probs, raw_logits = score_candidate(prompt, dx, model)
        dx_scores.append((dx, score, raw_logits))
        debug_rows.append(
            {
                "case_id": case_id,
                "group": group,
                "candidate": dx,
                "log_probs": log_probs,
                "raw_logits": raw_logits,
            }
        )
    return dx_scores, debug_rows


def tensor_to_json(tensor):
    return json.dumps(tensor.cpu().numpy().tolist())


def run_prompt(prompt, model, sae, threshold=1.0) -> Dict[str, Any]:
    """Run a single prompt and return SAE feature activations.

    :param prompt: The prompt to be processed by the model.
    :param model: The model used to generate activations.
    :param sae: The SAE model used to process the activations.
    :param threshold: Threshold for determining active features.
    :return: Dictionary containing SAE activations, active features, and top diagnoses.
    """
    with torch.no_grad():
        tokenised_prompt = model.to_tokens(prompt)
        model_activations = model.run_with_cache(tokenised_prompt, return_type=None)[1][
            sae.cfg.hook_name
        ]
        vectorised = model_activations[0, -1, :].unsqueeze(0)
        sae_activations = sae(vectorised)[0]
        active_features = (sae_activations.abs() > threshold).squeeze(0)
        n_active_features = active_features.sum().item()

        sae_output = {"n_active_features": n_active_features}
        for i in range(sae_activations.shape[0]):
            sae_output[f"activation_{i}"] = sae_activations[i].item()

        for i in range(active_features.shape[0]):
            sae_output[f"active_feature_{i}"] = active_features[i].item()

    return sae_output


def extract_top_diagnoses(prompt, model, demo_combination, case_id, true_dx: str = None) -> Dict[str, Any]:
    """Extract top diagnoses from SAE activations.

    Candidate-based Top-5 Diagnoses Extraction.

    :param prompt: The prompt to be used for diagnosis extraction.
    :param model: The model used for scoring diagnoses.
    :param demo_combination: Demographic combination used for the case.
    :param case_id: Optional case identifier for logging/debugging.
    :param true_dx: True diagnosis for calculating correctness.
    :return: Dictionary containing top diagnoses and debug information.
    """
    with torch.no_grad():
        dx_json_path = os.getcwd() + "/release_conditions.json"
        diagnosis_list = load_diagnosis_list(dx_json_path)
        group = "_".join(demo_combination) if len(demo_combination) > 0 else "no_demo"
        dx_scores, debug_rows = score_diagnoses(
            prompt, group, diagnosis_list, model, case_id
        )
        top5 = []
        top5_logits = []
        sorted_dx_scores = sorted(dx_scores, key=lambda x: x[1], reverse=True)[:5]
        for dx in sorted_dx_scores:
            top5.append(dx[0])
            top5_logits.append(dx[2])

        # Robustly define correctness flags
        if true_dx and top5:
            correct_top1 = top5[0].lower() == true_dx.lower()
            correct_top5 = any(dx.lower() == true_dx.lower() for dx in top5)
        else:
            correct_top1 = False
            correct_top5 = False
        
        # Mark each candidate row
        for row in debug_rows:
            row["correct"] = row["candidate"].lower() == true_dx.lower() if true_dx else False
            row["correct_top1"] = correct_top1
            row["correct_top5"] = correct_top5
        
        # Save debug info to CSV
        debug_info_to_csv(debug_rows)

        diagnoses_output = {
            "top5": top5,
            "top5_logits": top5_logits,
            "correct_top1": correct_top1,
            "correct_top5": correct_top5,
            "debug_rows": debug_rows,
        }

        # Write summary of top1/top5 correctness to CSV
        summary_info_to_csv([
            {
                "case_id": case_id,
                "group": group,
                "top1": top5[0] if top5 else None,
                "top5": top5,
                "correct_top1": correct_top1,
                "correct_top5": correct_top5,
            }
        ])
    
    return diagnoses_output


def compile_results(
    prompt_output_1, prompt_output_2, pair, case_id=None, case_info=None
) -> Dict[str, Any]:
    """Run model and SAE on both prompts, compute active features, activation difference, and overlap.

    Save per-case SAE activations to disk for further analysis.

    :param prompt_output_1: Output results from the first prompt.
    :param prompt_output_2: Output results from the second prompt.
    :param pair: Tuple containing the names of the two groups being compared.
    :param case_id: Optional case identifier for saving results.
    :param case_info: Optional additional information about the case.
    :return: Dictionary containing the results of the comparison.
    """
    group1_name = pair[0]
    group2_name = pair[1] if len(pair[1]) > 0 else "no_demo"

    # Get result of comparing activations from both prompts' outputs
    activations_1 = prompt_output_1["activations"]
    activations_2 = prompt_output_2["activations"]

    n_active_features_1 = prompt_output_1["n_active_features"]
    n_active_features_2 = prompt_output_2["n_active_features"]

    active_features_1 = prompt_output_1["active_features"]
    active_features_2 = prompt_output_2["active_features"]

    activation_difference = torch.norm(activations_1 - activations_2).item()
    overlap = (
        ((active_features_1) & (active_features_2)).nonzero(as_tuple=True)[0].tolist()
    )

    # Optionally, save more details if needed
    result = {
        "1": group1_name,
        "2": group2_name,
        "n_active_1": n_active_features_1,
        "n_active_2": n_active_features_2,
        "activation_difference": activation_difference,
        "overlapping_features": overlap,
        "top_dxs_" + group1_name: prompt_output_1["top_dxs"],
        "top_dxs_" + group2_name: prompt_output_2["top_dxs"],
        f"top_dxs_{group1_name}_candidate": prompt_output_1["top5"],
        f"top_dxs_{group2_name}_candidate": prompt_output_2["top5"],
        "sae_out_with": activations_1.cpu().numpy(),
        "sae_out_without": activations_2.cpu().numpy(),
        "case_id": case_id,
        "case_info": case_info,
        # Top-1/Top-5 correctness flags per group
        f"correct_top1_{group1_name}": prompt_output_1.get("correct_top1"),
        f"correct_top5_{group1_name}": prompt_output_1.get("correct_top5"),
        f"correct_top1_{group2_name}": prompt_output_2.get("correct_top1"),
        f"correct_top5_{group2_name}": prompt_output_2.get("correct_top5"),
    }

    if pair == ("age", ""):
        print(f"Results for pair {pair}: \n{result}\n")

    return result


def run_analysis():
    """Main function to run the analysis."""
    pass


if __name__ == "__main__":
    run_analysis()