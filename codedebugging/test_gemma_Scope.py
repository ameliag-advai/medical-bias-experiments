import torch
from functools import partial
from transformer_lens import HookedTransformer, utils
from sae_lens import SAE


# === Load Model ===
print("🤖 Loading Gemma-2b-IT model...")
device = "cpu"  # Use CPU since CUDA not available
model = HookedTransformer.from_pretrained(
    "google/gemma-2b-it",  # ✅ Use instruction-tuned version
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device=device,
    move_to_device=True
)

# === Load SAE ===
print("🧠 Loading matching SAE for Gemma-2b-IT...")
sae, *_ = SAE.from_pretrained(
    release="jbloom/Gemma-2b-IT-Residual-Stream-SAEs",  # ✅ IT-specific SAE
    sae_id="gemma_2b_it_blocks.12.hook_resid_post_16384",  # ✅ Layer 12, correct hook
    device=device
)
print(f"SAE hook name: {sae.cfg.hook_name}")

# === Setup Prompt ===
example_prompt = "When John and Mary went to the shops, John gave the bag to"
example_answer = " Mary"

tokens = model.to_tokens(example_prompt).to(device)
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

# === Reconstruction Quality Analysis ===
print("\n=== 🔍 Reconstruction Quality ===")
reconstruction_error = abs(reconstr_loss - orig_loss)
zero_ablation_error = abs(zero_loss - orig_loss)

print(f"Reconstruction Error: {reconstruction_error:.4f}")
print(f"Zero Ablation Error:  {zero_ablation_error:.4f}")

if reconstruction_error < 0.1:
    print("✅ EXCELLENT reconstruction quality (error < 0.1)")
elif reconstruction_error < 0.5:
    print("✅ GOOD reconstruction quality (error < 0.5)")
elif reconstruction_error < 1.0:
    print("⚠️  FAIR reconstruction quality (error < 1.0)")
else:
    print("❌ POOR reconstruction quality (error >= 1.0)")

reconstruction_ratio = reconstruction_error / zero_ablation_error if zero_ablation_error > 0 else float('inf')
print(f"\nReconstruction preserves {(1-reconstruction_ratio)*100:.1f}% of original behavior")
print(f"(Compared to zero ablation baseline)")
