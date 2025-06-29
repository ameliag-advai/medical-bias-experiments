"""
Fix SAE reconstruction issues by testing different SAE models and configurations.
"""
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
import numpy as np


def test_sae_reconstruction(model, sae, test_prompt="When John and Mary went to the shops, John gave the bag to", expected_token=" Mary"):
    """Test SAE reconstruction quality on a simple prompt."""
    print(f"🔍 Testing SAE reconstruction...")
    print(f"Test prompt: '{test_prompt}'")
    print(f"Expected next token: '{expected_token}'")
    
    # Tokenize
    tokens = model.to_tokens(test_prompt)
    print(f"Tokenized length: {tokens.shape[1]} tokens")
    
    # Get original model predictions
    with torch.no_grad():
        original_logits = model(tokens)
        original_probs = torch.softmax(original_logits[0, -1], dim=-1)
        
        # Find expected token
        expected_token_id = model.to_tokens(expected_token, prepend_bos=False)[0, 0]
        original_rank = (original_probs > original_probs[expected_token_id]).sum().item()
        original_prob = original_probs[expected_token_id].item()
        
        print(f"\n=== Original Model ===")
        print(f"Expected token '{expected_token}' rank: {original_rank}")
        print(f"Expected token probability: {original_prob:.4f}")
        
        # Get top predictions
        top_probs, top_indices = torch.topk(original_probs, 5)
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = model.to_string(idx)
            print(f"Top {i}: {prob:.4f} '{token}'")
    
    # Test SAE reconstruction
    print(f"\n=== SAE Reconstruction ===")
    print(f"SAE hook point: {sae.cfg.hook_name}")
    print(f"SAE device: {sae.device}")
    print(f"Model device: {model.cfg.device}")
    
    try:
        with torch.no_grad():
            # Get activations at SAE hook point
            _, cache = model.run_with_cache(tokens)
            hook_activations = cache[sae.cfg.hook_name]
            print(f"Hook activations shape: {hook_activations.shape}")
            
            # Get SAE activations and reconstruction
            sae_activations = sae.encode(hook_activations)
            sae_reconstruction = sae.decode(sae_activations)
            
            print(f"SAE activations shape: {sae_activations.shape}")
            print(f"SAE reconstruction shape: {sae_reconstruction.shape}")
            
            # Replace activations with SAE reconstruction
            def reconstruction_hook(activations, hook):
                return sae_reconstruction
            
            # Run model with SAE reconstruction
            reconstructed_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(sae.cfg.hook_name, reconstruction_hook)]
            )
            
            reconstructed_probs = torch.softmax(reconstructed_logits[0, -1], dim=-1)
            reconstructed_rank = (reconstructed_probs > reconstructed_probs[expected_token_id]).sum().item()
            reconstructed_prob = reconstructed_probs[expected_token_id].item()
            
            print(f"Expected token '{expected_token}' rank: {reconstructed_rank}")
            print(f"Expected token probability: {reconstructed_prob:.4f}")
            
            # Get top predictions
            top_probs, top_indices = torch.topk(reconstructed_probs, 5)
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = model.to_string(idx)
                print(f"Top {i}: {prob:.4f} '{token}'")
            
            # Calculate reconstruction loss
            mse_loss = torch.nn.functional.mse_loss(hook_activations, sae_reconstruction)
            print(f"\nReconstruction MSE Loss: {mse_loss:.4f}")
            
            # Calculate sparsity
            l0_norm = (sae_activations > 0).float().mean()
            l1_norm = sae_activations.abs().mean()
            print(f"L0 sparsity: {l0_norm:.4f}")
            print(f"L1 norm: {l1_norm:.4f}")
            
            return {
                'original_rank': original_rank,
                'original_prob': original_prob,
                'reconstructed_rank': reconstructed_rank,
                'reconstructed_prob': reconstructed_prob,
                'mse_loss': mse_loss.item(),
                'l0_sparsity': l0_norm.item(),
                'l1_norm': l1_norm.item()
            }
            
    except Exception as e:
        print(f"❌ SAE reconstruction failed: {e}")
        return None


def try_different_sae_models():
    """Try different SAE models to find one that works."""
    print("🔍 Testing different SAE models for Gemma-2b-it...\n")
    
    # Load base model
    model = HookedTransformer.from_pretrained("google/gemma-2b-it", device="cpu")
    
    # List of SAE models to try
    sae_configs = [
        {
            'release': 'jbloom/Gemma-2b-IT-Residual-Stream-SAEs',
            'sae_id': 'gemma_2b_it_blocks.12.hook_resid_post_16384',
            'name': 'Residual Stream Layer 12'
        },
        {
            'release': 'jbloom/Gemma-2b-IT-Residual-Stream-SAEs', 
            'sae_id': 'gemma_2b_it_blocks.6.hook_resid_post_16384',
            'name': 'Residual Stream Layer 6'
        },
        {
            'release': 'jbloom/Gemma-2b-IT-Residual-Stream-SAEs',
            'sae_id': 'gemma_2b_it_blocks.18.hook_resid_post_16384', 
            'name': 'Residual Stream Layer 18'
        }
    ]
    
    results = []
    
    for config in sae_configs:
        print(f"=== Testing {config['name']} ===")
        try:
            sae, *_ = SAE.from_pretrained(
                release=config['release'],
                sae_id=config['sae_id'],
                device="cpu"
            )
            
            result = test_sae_reconstruction(model, sae)
            if result:
                result['config'] = config
                results.append(result)
                print(f"✅ Success! Reconstruction rank: {result['reconstructed_rank']}")
            else:
                print(f"❌ Failed")
                
        except Exception as e:
            print(f"❌ Failed to load SAE: {e}")
        
        print()
    
    # Find best SAE
    if results:
        best_sae = min(results, key=lambda x: x['reconstructed_rank'])
        print(f"🏆 Best SAE: {best_sae['config']['name']}")
        print(f"   Original rank: {best_sae['original_rank']}")
        print(f"   Reconstructed rank: {best_sae['reconstructed_rank']}")
        print(f"   MSE Loss: {best_sae['mse_loss']:.4f}")
        
        return best_sae['config']
    else:
        print("❌ No working SAE found!")
        return None


def create_fixed_loader(best_config):
    """Create a fixed loader with the best SAE configuration."""
    if not best_config:
        print("❌ No valid SAE configuration to use")
        return
    
    fixed_loader_code = f'''from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

def load_model_and_sae(model_scope="gemma", device=None):
    """Load model and SAE with fixed configuration."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained("google/gemma-2b-it", device=device, move_to_device=True)
    
    # Fixed SAE configuration that works
    sae, *_ = SAE.from_pretrained(
        release="{best_config['release']}",
        sae_id="{best_config['sae_id']}",
        device=device
    )
    
    model.to(device)
    print(f"✅ Loaded {{best_config['name']}} SAE successfully")
    
    return model, sae
'''
    
    # Write fixed loader
    with open('/Users/amelia/22406alethia/alethia/src/advai/models/loader_fixed.py', 'w') as f:
        f.write(fixed_loader_code)
    
    print(f"✅ Created fixed loader: src/advai/models/loader_fixed.py")
    print(f"   Using: {best_config['name']}")
    print(f"   SAE ID: {best_config['sae_id']}")


def main():
    print("🔧 SAE Reconstruction Diagnostic Tool\n")
    
    # Test current configuration first
    print("=== Testing Current Configuration ===")
    try:
        from src.advai.models.loader import load_model_and_sae
        model, sae = load_model_and_sae('gemma', device='cpu')
        current_result = test_sae_reconstruction(model, sae)
        
        if current_result and current_result['reconstructed_rank'] < 1000:
            print("✅ Current SAE configuration works reasonably well!")
            return
        else:
            print("❌ Current SAE configuration has poor reconstruction quality")
            
    except Exception as e:
        print(f"❌ Current configuration failed: {e}")
    
    print("\n" + "="*60)
    
    # Try different SAE models
    best_config = try_different_sae_models()
    
    if best_config:
        create_fixed_loader(best_config)
        print(f"\n🎯 To use the fixed loader, replace imports with:")
        print(f"   from src.advai.models.loader_fixed import load_model_and_sae")
    else:
        print(f"\n❌ Could not find a working SAE configuration")
        print(f"   You may need to:")
        print(f"   1. Use a different SAE release")
        print(f"   2. Check SAE-model compatibility")
        print(f"   3. Verify hook point names")


if __name__ == "__main__":
    main()
