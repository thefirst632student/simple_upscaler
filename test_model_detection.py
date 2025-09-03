#!/usr/bin/env python3

import safetensors.torch
import torch
from pathlib import Path

def test_model_detection(model_path):
    """Test the model detection logic for a given model"""
    print(f"Testing model: {model_path}")
    
    # Load state dict
    if Path(model_path).suffix.lower() == '.safetensors':
        state_dict = safetensors.torch.load_file(model_path)
    else:
        state_dict = torch.load(model_path, weights_only=False, map_location="cpu")
    
    print(f"Number of keys: {len(state_dict.keys())}")
    
    # Test current detection logic
    print("\n=== Current Detection Logic ===")
    
    # SRVGGNet Real-ESRGAN (v2)
    if ("params" in state_dict.keys() and "body.0.weight" in state_dict["params"].keys()):
        print("✓ Detected as: RealESRGAN-v2")
        return
    
    # FDAT (Fast Dual Aggregation Transformer) - CURRENT LOGIC
    elif "conv_first.weight" in state_dict and "groups.0.blocks.0.n1.weight" in state_dict:
        print("✓ Detected as: FDAT")
        return
    
    # SPSR 
    elif "f_HR_conv1.0.weight" in state_dict:
        print("✓ Detected as: SPSR")
        return
    
    # DAT detection
    elif is_dat_architecture(state_dict):
        print("✓ Detected as: DAT")
        return
    
    # Default to ESRGAN
    else:
        print("✓ Detected as: ESRGAN (default)")
    
    print("\n=== New Detection Analysis ===")
    
    # Analyze the structure
    has_conv_first = 'conv_first.weight' in state_dict
    has_groups = any('groups.' in key for key in state_dict.keys())
    has_layers = any('layers.' in key for key in state_dict.keys())
    has_blocks = any('blocks.' in key for key in state_dict.keys())
    has_attn = any('attn.' in key for key in state_dict.keys())
    
    print(f"Has conv_first: {has_conv_first}")
    print(f"Has groups: {has_groups}")
    print(f"Has layers: {has_layers}")
    print(f"Has blocks: {has_blocks}")
    print(f"Has attn: {has_attn}")
    
    # Show some sample keys
    print(f"\nFirst 10 keys:")
    for i, key in enumerate(sorted(state_dict.keys())[:10]):
        print(f"  {i+1:2}: {key}")
    
    # Show FDAT pattern
    fdat_keys = [k for k in state_dict.keys() if 'groups.' in k and 'blocks.' in k][:5]
    if fdat_keys:
        print(f"\nFDAT pattern keys (groups.X.blocks.Y):")
        for key in fdat_keys:
            print(f"  {key}")
    
    # Show DAT pattern
    dat_keys = [k for k in state_dict.keys() if 'layers.' in k and 'blocks.' in k][:5]
    if dat_keys:
        print(f"\nDAT pattern keys (layers.X.blocks.Y):")
        for key in dat_keys:
            print(f"  {key}")
    
    # Propose correct detection
    if has_conv_first and has_groups and has_blocks and has_attn:
        # Check for FDAT pattern (groups.X.blocks.Y.attn)
        has_fdat_pattern = any('groups.' in key and 'blocks.' in key and 'attn.' in key for key in state_dict.keys())
        if has_fdat_pattern:
            print("\n🔧 Should be detected as: FDAT")
            return
    
    if has_conv_first and has_layers and has_blocks and has_attn:
        # Check for DAT pattern (layers.X.blocks.Y.attn)
        has_dat_pattern = any('layers.' in key and 'blocks.' in key and 'attn.' in key for key in state_dict.keys())
        if has_dat_pattern:
            print("\n🔧 Should be detected as: DAT")
            return

def is_dat_architecture(state_dict):
    """Current DAT detection logic"""
    has_conv_first = 'conv_first.weight' in state_dict
    has_layers = any('layers.' in key for key in state_dict.keys())
    has_blocks = any('blocks.' in key for key in state_dict.keys()) 
    has_before_rg = any('before_RG' in key for key in state_dict.keys())
    has_attn = any('attn.' in key for key in state_dict.keys())
    
    # Check for the specific DAT pattern: layers.X.blocks.Y.attn structure
    has_dat_pattern = False
    for key in state_dict.keys():
        if 'layers.' in key and 'blocks.' in key and 'attn.' in key:
            has_dat_pattern = True
            break
    
    # DAT should have conv_first, layers with blocks, and attention modules
    return has_conv_first and has_layers and has_blocks and has_attn and has_dat_pattern

if __name__ == "__main__":
    models = [
        "models/4x_IllustrationJaNai_V2standard_FDAT_XL_18k.safetensors",
        "models/4x_IllustrationJaNai_V2standard_FDAT_M_52k.safetensors", 
        "models/4x_IllustrationJaNai_V2standard_DAT2_27k.safetensors",
        "models/4x-AnimeSharp.pth",
        "models/4x_eula_digimanga_bw_v2_nc1_307k.pth"
    ]
    
    for model in models:
        if Path(model).exists():
            print("="*60)
            test_model_detection(model)
            print()