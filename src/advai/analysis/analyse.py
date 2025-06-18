import torch
import os
import json
import csv
import logging
from typing import Dict, Any, Tuple, List

logging.basicConfig(level=logging.INFO)  # Set to logging.DEBUG to show debug messages
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
    """Save debug information to a CSV file for post-analysis.

    :param debug_rows: List of dictionaries containing debug information for each candidate diagnosis.
    """
    with open("diagnosis_logit_debug.csv", "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["case_id", "group", "candidate", "log_probs", "raw_logits"],
        )
        if csvfile.tell() == 0:
            writer.writeheader()
        for row in debug_rows:
            # Write as string for list fields
            row["log_probs"] = str(row["log_probs"])
            row["raw_logits"] = str(row["raw_logits"])
            writer.writerow(row)


def score_candidate(
    prompt_prefix, candidate, model
) -> Tuple[float, List[float], List[float]]:
    """Score a candidate diagnosis by computing its log probability in the model's output.

    :param prompt: The prompt to be used for scoring the candidate.
    :param candidate: The candidate diagnosis to be scored.
    :param model: The model used to score the candidate.
    :return: A tuple containing the total log probability, a list of log probabilities for each token, and raw logits.
    """
    # Add the candidate diagnosis to the prompt and run the model.
    prompt = f"{prompt_prefix} Diagnosis is: {candidate}."
    toks = model.to_tokens(prompt)
    candidate_tokens = model.to_tokens(candidate)
    logits, _ = model.run_with_cache(toks)

    log_probs = []
    raw_logits = []
    candidate_tokens = (
        candidate_tokens.flatten().tolist()
        if hasattr(candidate_tokens, "flatten")
        else list(candidate_tokens)
    )
    for i, token_id in enumerate(candidate_tokens):
        idx = int(token_id)
        pos = -len(candidate_tokens) - 1 + i  # position of each candidate token
        logit = logits[0, pos, :]  # logit for each candidate token
        log_prob = torch.log_softmax(logit, dim=-1)[idx]

        # Ensure log_prob is a scalar
        if hasattr(log_prob, "item") and log_prob.numel() == 1:
            log_probs.append(log_prob.item())
        elif hasattr(log_prob, "tolist"):
            val = log_prob.tolist()
            if isinstance(val, list):
                log_probs.append(
                    val[0] if val and isinstance(val[0], (int, float)) else float("nan")
                )
            else:
                log_probs.append(float(val))
        else:
            log_probs.append(float(log_prob))
        raw_logits.append(logit[idx].item())

    logger.debug(
        f"[LOGITS] Candidate: '{candidate}' | Log probs (scalar): {log_probs} | Raw logits: {raw_logits}"
    )

    score = sum(log_probs) if log_probs else float("-inf")
    normalised_score = score / len(candidate_tokens) if candidate_tokens else float("-inf")

    return normalised_score, log_probs, raw_logits


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
    print(f"Symptom prompt: {prompt}\n")
    for dx in diagnosis_list[:49]:  # [:5]:
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
        top_dxs = torch.topk(sae_activations, 5).indices.tolist()

        sae_output = {
            "activations": sae_activations,
            "active_features": active_features,
            "n_active_features": n_active_features,
            "top_dxs": top_dxs,
        }

    return sae_output


def extract_top_diagnoses(prompt, model, demo_combination, case_id) -> Dict[str, Any]:
    """Extract top diagnoses from SAE activations.

    Candidate-based Top-5 Diagnoses Extraction.

    :param prompt: The prompt to be used for diagnosis extraction.
    :param model: The model used for scoring diagnoses.
    :param demo_combination: Demographic combination used for the case.
    :param case_id: Optional case identifier for logging/debugging.
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
        sorted_dx_scores = sorted(dx_scores, key=lambda x: x[1], reverse=True)[10:15]  # 10:15
        for dx in sorted_dx_scores:
            top5.append(dx[0])
            top5_logits.append(dx[2])

        # Save debug info to CSV for post-analysis
        debug_info_to_csv(debug_rows)

        diagnoses_output = {"top5": top5, "top5_logits": top5_logits, "debug_rows": debug_rows}

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
    :param save_dir: Directory where results will be saved.
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
    }

    if pair == ("age", ""):
        print(f"Results for pair {pair}: \n{result}\n")
        # print(f"Top dxs without {result['top_dxs_no_demo_candidate']}\n")

    return result


# @TODO: Implement the main function to run the analysis. Optionally, do this in a notebook.
def run_analysis():
    pass


if __name__ == "__main__":
    run_analysis()
