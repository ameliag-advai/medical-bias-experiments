import torch

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
    overlap = ((active_features_1) & (active_features_2)).nonzero(as_tuple=True)[0].tolist()
    return {
        'n_active_1': n_active_features_1,
        'n_active_2': n_active_features_2,
        'activation_difference': activation_difference,
        'overlapping_features': overlap,
        'case_id': case_id
    }