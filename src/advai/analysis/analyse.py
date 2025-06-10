import torch
import os
import json
import csv
import logging

logging.basicConfig(level=logging.INFO)  # Set to logging.DEBUG to show debug messages
logger = logging.getLogger()


def load_diagnosis_list(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [v['condition_name'] for v in data.values()]


def debug_info_to_csv(debug_rows):
    """Save debug information to a CSV file for post-analysis."""
    with open('diagnosis_logit_debug.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['case_id', 'group', 'candidate', 'log_probs', 'raw_logits'])
        if csvfile.tell() == 0:
            writer.writeheader()
        for row in debug_rows:
            # Write as string for list fields
            row['log_probs'] = str(row['log_probs'])
            row['raw_logits'] = str(row['raw_logits'])
            writer.writerow(row)


def score_candidate(prompt_prefix, candidate, model):
    prompt = f"{prompt_prefix}{candidate}."
    toks = model.to_tokens(prompt)
    candidate_tokens = model.to_tokens(candidate)
    logits, _ = model.run_with_cache(toks)
    log_probs = []
    raw_logits = []
    candidate_tokens = candidate_tokens.flatten().tolist() if hasattr(candidate_tokens, 'flatten') else list(candidate_tokens)
    for i, token_id in enumerate(candidate_tokens):
        idx = int(token_id)
        pos = -len(candidate_tokens) - 1 + i  # position of each candidate token
        logit = logits[0, pos, :]
        log_prob = torch.log_softmax(logit, dim=-1)[idx]
        # Ensure log_prob is a scalar
        if hasattr(log_prob, 'item') and log_prob.numel() == 1:
            log_probs.append(log_prob.item())
        elif hasattr(log_prob, 'tolist'):
            val = log_prob.tolist()
            if isinstance(val, list):
                log_probs.append(val[0] if val and isinstance(val[0], (int, float)) else float('nan'))
            else:
                log_probs.append(float(val))
        else:
            log_probs.append(float(log_prob))
        raw_logits.append(logit[idx].item())
    logger.debug(f"[LOGITS] Candidate: '{candidate}' | Log probs (scalar): {log_probs} | Raw logits: {raw_logits}")
    return sum(log_probs) if log_probs else float('-inf'), log_probs, raw_logits


def score_diagnoses(prefix, group, diagnosis_list, model, case_id, score_candidate_func, debug_rows=None):
    if debug_rows is None:
        debug_rows = []
    dx_scores = []
    for dx in diagnosis_list[:5]:
        score, log_probs, raw_logits = score_candidate_func(prefix, dx, model)
        dx_scores.append((dx, score))
        debug_rows.append({'case_id': case_id, 'group': group, 'candidate': dx, 'log_probs': log_probs, 'raw_logits': raw_logits})
    return dx_scores, debug_rows


def process_activations(act, sae, threshold=1.0):
    vec = act[0, -1, :].unsqueeze(0)
    sae_out = sae(vec)
    active = (sae_out[0].abs() > threshold).squeeze(0)
    n_active = active.sum().item()
    top_dxs = torch.topk(sae_out[0], 5).indices.tolist()
    return sae_out, active, n_active, top_dxs


def run_prompt(prompt, model, sae):
    """Run a single prompt and return SAE feature activations."""
    with torch.no_grad():
        tokenised_prompt = model.to_tokens(prompt)
        model_activations = model.run_with_cache(tokenised_prompt, return_type=None)[1][sae.cfg.hook_name]
        vectorised = model_activations[0, -1, :].unsqueeze(0)
        sae_activations = sae(vectorised)[0]

    return sae_activations


def compare_activations(activations_1, activations_2, case_id=None, threshold=1.0):
    """Compare SAE feature activations for two sets of activations."""
    active_features_1 = (activations_1.abs() > threshold).squeeze(0)
    active_features_2 = (activations_2.abs() > threshold).squeeze(0)

    n_active_features_1 = active_features_1.sum().item()
    n_active_features_2 = active_features_2.sum().item()

    activation_difference = torch.norm(activations_1 - activations_2).item()
    overlap = (
        ((active_features_1) & (active_features_2)).nonzero(as_tuple=True)[0].tolist()
    )

    result = {
        "n_active_1": n_active_features_1,
        "n_active_2": n_active_features_2,
        "activation_difference": activation_difference,
        "overlapping_features": overlap,
        "case_id": case_id,
    }

    return result


def compare_activations(text_with_demo, text_without_demo, model, sae, case_info=None, case_id=None, save_dir="activations"):
    """Run model and SAE on both prompts, compute active features, activation difference, and overlap.
    
    Save per-case SAE activations to disk for further analysis.
    """
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # Use the provided prompts only—NO HARDCODED INFO
        toks_with = model.to_tokens(text_with_demo)
        toks_without = model.to_tokens(text_without_demo)

        act_with = model.run_with_cache(toks_with, return_type=None)[1][sae.cfg.hook_name]
        act_without = model.run_with_cache(toks_without, return_type=None)[1][sae.cfg.hook_name]

        sae_out_with, active_with, n_active_with, top_dxs_with_demo = process_activations(act_with, sae)
        sae_out_without, active_without, n_active_without, top_dxs_without_demo = process_activations(act_without, sae)

        activation_difference = torch.norm(sae_out_with[0] - sae_out_without[0]).item()
        overlap = ((active_with) & (active_without)).nonzero(as_tuple=True)[0].tolist()

        # Save activations
        torch.save({'sae_out_with': sae_out_with[0].cpu(), 'sae_out_without': sae_out_without[0].cpu()}, os.path.join(save_dir, f"case_{case_id}_activations.pt"))

        # --- Candidate-based Top-5 Diagnoses Extraction ---
        dx_json_path = "~/release_conditions.json"
        diagnosis_list = load_diagnosis_list(dx_json_path)

        prefix_with = text_with_demo.rsplit(" ", 1)[0] + " "
        prefix_without = text_without_demo.rsplit(" ", 1)[0] + " "

        dx_scores_with, debug_rows = score_diagnoses(prefix_with, "with_demo", diagnosis_list, model, case_id, score_candidate)
        dx_scores_without, debug_rows = score_diagnoses(prefix_without, "no_demo", diagnosis_list, model, case_id, score_candidate, debug_rows)
        
        top5_with = [dx for dx, _ in sorted(dx_scores_with, key=lambda x: x[1], reverse=True)[:5]]
        top5_without = [dx for dx, _ in sorted(dx_scores_without, key=lambda x: x[1], reverse=True)[:5]]

        # Save debug info to CSV for post-analysis
        debug_info_to_csv(debug_rows)

        # Optionally, save more details if needed
        result = {
            'n_active_with': n_active_with,
            'n_active_without': n_active_without,
            'activation_difference': activation_difference,
            'overlapping_features': overlap,
            'top_dxs_with_demo': top_dxs_with_demo,
            'top_dxs_without_demo': top_dxs_without_demo,
            'top_dxs_with_demo_candidate': top5_with,
            'top_dxs_without_demo_candidate': top5_without,
            'sae_out_with': sae_out_with[0].cpu().numpy(),
            'sae_out_without': sae_out_without[0].cpu().numpy(),
            'case_id': case_id,
            'case_info': case_info
        }
        
        return result
