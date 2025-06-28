from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

def load_model_and_sae(model_scope="gemma", device=None):


    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained("google/gemma-2b-it", device=device, move_to_device=True)
    sae, *_ = SAE.from_pretrained(
        release="gemma-2b-it-res-jb",
        sae_id="blocks.12.hook_resid_post",
        device=device
    )
    model.to(device)

    return model, sae
