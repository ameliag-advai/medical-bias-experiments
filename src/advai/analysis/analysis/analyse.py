import torch
import os
import json

def analyse_case_for_bias(text_with_demo, text_without_demo, model, sae, case_info=None, case_id=None, save_dir="activations"):
    """
    Run model and SAE on both prompts, compute active features, activation difference, and overlap.
    Save per-case SAE activations to disk for further analysis.
    Returns a dict with keys: n_active_with, n_active_without, activation_difference, overlapping_features, top_dxs_with_demo, top_dxs_without_demo, sae_out_with, sae_out_without
    """
    os.makedirs(save_dir, exist_ok=True)
    import torch
    with torch.no_grad():
        # Use the provided prompts only—NO HARDCODED INFO
        print(f'[DEBUG] Using provided prompt with demo: {text_with_demo}')
        print('[DEBUG] Before to_tokens (with demo)')
        toks_with = model.to_tokens(text_with_demo)
        print('[DEBUG] After to_tokens (with demo)')
        print('[DEBUG] Before to_tokens (without demo)')
        toks_without = model.to_tokens(text_without_demo)
        print('[DEBUG] After to_tokens (without demo)')
        print('[DEBUG] Before run_with_cache (with demo)')
        act_with = model.run_with_cache(toks_with, return_type=None)[1][sae.cfg.hook_name]
        print('[DEBUG] After run_with_cache (with demo)')
        print('[DEBUG] Before run_with_cache (without demo)')
        act_without = model.run_with_cache(toks_without, return_type=None)[1][sae.cfg.hook_name]
        print('[DEBUG] After run_with_cache (without demo)')
        print('[DEBUG] Before SAE (with demo)')
        # --- Candidate-based Top-5 Diagnoses Extraction ---
        def load_diagnosis_list(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            return [v['condition_name'] for v in data.values()]

        def score_candidate(prompt_prefix, candidate, model):
            prompt = f"{prompt_prefix}{candidate}."
            toks = model.to_tokens(prompt)
            candidate_tokens = model.to_tokens(candidate)
            logits, _ = model.run_with_cache(toks)
            log_probs = []
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
            return sum(log_probs) if log_probs else float('-inf')

        dx_json_path = "/Users/ameliag/Downloads/alethia-main/release_conditions.json"
        print('[DEBUG] Before load_diagnosis_list')
        diagnosis_list = load_diagnosis_list(dx_json_path)
        print(f'[DEBUG] After load_diagnosis_list: {len(diagnosis_list)} diagnoses loaded')
        prefix_with = text_with_demo.rsplit(" ", 1)[0] + " "
        prefix_without = text_without_demo.rsplit(" ", 1)[0] + " "
        print('[DEBUG] Before dx_scores_with loop')
        dx_scores_with = []
        dx_scores_without = []
        for dx in diagnosis_list[:5]:
            print(f'[DEBUG] Scoring WITH demo: {dx}')
            dx_scores_with.append((dx, score_candidate(prefix_with, dx, model)))
        print('[DEBUG] After dx_scores_with loop')
        for dx in diagnosis_list[:5]:
            print(f'[DEBUG] Scoring WITHOUT demo: {dx}')
            dx_scores_without.append((dx, score_candidate(prefix_without, dx, model)))
        print('[DEBUG] After dx_scores_without loop')
        top5_with = [dx for dx, _ in sorted(dx_scores_with, key=lambda x: x[1], reverse=True)[:5]]
        top5_without = [dx for dx, _ in sorted(dx_scores_without, key=lambda x: x[1], reverse=True)[:5]]
        vec_with = act_with[0, -1, :].unsqueeze(0)
        sae_out_with = sae(vec_with)
        print('[DEBUG] After SAE (with demo)')
        print('[DEBUG] Before SAE (without demo)')
        vec_without = act_without[0, -1, :].unsqueeze(0)
        sae_out_without = sae(vec_without)
        print('[DEBUG] After SAE (without demo)')
        threshold = 1.0
        active_with = (sae_out_with[0].abs() > threshold).squeeze(0)
        active_without = (sae_out_without[0].abs() > threshold).squeeze(0)
        n_active_with = active_with.sum().item()
        n_active_without = active_without.sum().item()
        activation_difference = torch.norm(sae_out_with[0] - sae_out_without[0]).item()
        overlap = ((active_with) & (active_without)).nonzero(as_tuple=True)[0].tolist()
        top_dxs_with_demo = torch.topk(sae_out_with[0], 5).indices.tolist()
        top_dxs_without_demo = torch.topk(sae_out_without[0], 5).indices.tolist()
        
        # Save activations
        torch.save({'sae_out_with': sae_out_with[0].cpu(), 'sae_out_without': sae_out_without[0].cpu()}, os.path.join(save_dir, f"case_{case_id}_activations.pt"))
        # Optionally, save more details if needed
        return {
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