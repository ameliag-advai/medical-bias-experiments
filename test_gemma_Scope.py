import torch
from functools import partial
from transformer_lens import HookedTransformer, utils
from sae_lens import SAE, SAEConfig
from sae_lens.toolkit.pretrained_sae_loaders import gemma_2_sae_huggingface_loader


# === Load Model ===
model = HookedTransformer.from_pretrained(
    "google/gemma-2-2b",
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device="cuda",
    move_to_device=True
)

# === Load SAE ===
cfg_dict, state_dict, log_sparsity = gemma_2_sae_huggingface_loader(
    repo_id="google/gemma-scope-2b-pt-res",
    folder_name="layer_20/width_16k/average_l0_71",
    device="cuda"
)
cfg = SAEConfig.from_dict(cfg_dict)
sae = SAE(cfg=cfg)#.load_state_dict(state_dict)

# === Setup Prompt ===
example_prompt = "When John and Mary went to the shops, John gave the bag to"
example_answer = " Mary"

tokens = model.to_tokens(example_prompt).to("cuda")
hook_name = sae.cfg.hook_name


# === Run baseline forward pass ===
print("\n=== 🔍 Original Model ===")
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)
orig_loss = model(tokens, return_type="loss").item()
print(f"Original Loss: {orig_loss:.4f}")


# === Capture activations ===
_, cache = model.run_with_cache(example_prompt, prepend_bos=True)
sae_out = sae(cache[hook_name])


# === Hook Functions ===
def reconstr_hook(activations, hook, sae_out):
    return sae_out


def zero_abl_hook(activations, hook):
    return torch.zeros_like(activations)


# === SAE Reconstruction Forward ===
print("\n=== 🔍 SAE Reconstruction ===")
reconstr_loss = model.run_with_hooks(
    tokens,
    fwd_hooks=[(hook_name, partial(reconstr_hook, sae_out=sae_out))],
    return_type="loss",
).item()
print(f"SAE Reconstruction Loss: {reconstr_loss:.4f}")

with model.hooks(fwd_hooks=[(hook_name, partial(reconstr_hook, sae_out=sae_out))]):
    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# === Zero Ablation Forward ===
print("\n=== 🔍 Zero Ablation ===")
zero_loss = model.run_with_hooks(
    tokens,
    fwd_hooks=[(hook_name, zero_abl_hook)],
    return_type="loss",
).item()
print(f"Zero Ablation Loss: {zero_loss:.4f}")

with model.hooks(fwd_hooks=[(hook_name, zero_abl_hook)]):
    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


# === Summary ===
print("\n=== 📊 Loss Summary ===")
print(f"Original Loss:       {orig_loss:.4f}")
print(f"SAE Reconstruction:  {reconstr_loss:.4f}")
print(f"Zero Ablation:       {zero_loss:.4f}")
