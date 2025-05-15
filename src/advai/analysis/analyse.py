import torch

def analyse_case_for_bias(text_with_demo, text_without_demo, model, sae, case_info=None, case_id=None, save_dir="activations"):
    """Analyze a single case to compare SAE feature activations with/without demographics."""
    with torch.no_grad():
        toks_with = model.to_tokens(text_with_demo)
        toks_without = model.to_tokens(text_without_demo)

        act_with = model.run_with_cache(toks_with, return_type=None)[1][sae.cfg.hook_name]
        act_without = model.run_with_cache(toks_without, return_type=None)[1][sae.cfg.hook_name]

        vec_with = act_with[0, -1, :].unsqueeze(0)
        vec_without = act_without[0, -1, :].unsqueeze(0)

        sae_out_with = sae(vec_with)[0]
        sae_out_without = sae(vec_without)[0]

        threshold = 1.0
        active_with = (sae_out_with.abs() > threshold).squeeze(0)
        active_without = (sae_out_without.abs() > threshold).squeeze(0)

        n_active_with = active_with.sum().item()
        n_active_without = active_without.sum().item()
        activation_difference = torch.norm(sae_out_with - sae_out_without).item()
        overlap = ((active_with) & (active_without)).nonzero(as_tuple=True)[0].tolist()

        return {
            'n_active_with': n_active_with,
            'n_active_without': n_active_without,
            'activation_difference': activation_difference,
            'overlapping_features': overlap,
            'case_id': case_id
        }