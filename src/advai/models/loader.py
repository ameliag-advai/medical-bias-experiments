from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

def load_model_and_sae(model_scope="gemma", device=None):


    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_scope == "gpt2":
        model = HookedTransformer.from_pretrained("gpt2-small", device=device)
        sae, *_ = SAE.from_pretrained(
            release="jbloom/GPT2-Small-SAEs",
            sae_id="blocks.6.hook_mlp_out",
            device=device
        )
    elif model_scope == "gemma":
        model = HookedTransformer.from_pretrained("google/gemma-2b-it", device=device)
        sae, *_ = SAE.from_pretrained(
            release="jbloom/Gemma-2b-IT-Residual-Stream-SAEs",
            sae_id="gemma_2b_it_blocks.12.hook_resid_post_16384",
            device=device
        )
    else:
        raise NotImplementedError()

    return model, sae