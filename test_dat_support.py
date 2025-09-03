#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for DAT architecture support in the simple_manga_upscaler.
"""

import torch
import tempfile
import os
from pathlib import Path
from utils.architecture.DAT import DAT
from upscale import Upscale
import numpy as np


def test_dat_model_saving_and_loading():
    """Test creating, saving, and loading a DAT model"""
    print("Testing DAT model creation, saving, and loading...")
    
    # Create a small DAT model for testing
    model = DAT(
        upscale=2,
        in_chans=3,
        embed_dim=60,
        depth=[2, 2],
        num_heads=[2, 2],
        expansion_factor=2.0,
        img_size=64
    )
    model.eval()
    
    # Create a temporary file to save the model
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
        temp_model_path = tmp_file.name
    
    try:
        # Save the model
        torch.save(model.state_dict(), temp_model_path)
        print(f"✓ DAT model saved to {temp_model_path}")
        
        # Test upscaler loading
        upscaler = Upscale(
            model=temp_model_path,
            input=Path("./input"),
            output=Path("./output"),
            cpu=True
        )
        upscaler.load_model(temp_model_path)
        
        print(f"✓ Model loaded successfully")
        print(f"  - Architecture: {upscaler.last_kind}")
        print(f"  - Scale: {upscaler.last_scale}")
        print(f"  - Input channels: {upscaler.last_in_nc}")
        print(f"  - Output channels: {upscaler.last_out_nc}")
        print(f"  - Features: {upscaler.last_nf}")
        
        # Test image upscaling
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = upscaler.upscale(test_image)
        
        print(f"✓ Image upscaling successful")
        print(f"  - Input shape: {test_image.shape}")
        print(f"  - Output shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if os.path.exists(temp_model_path):
            os.unlink(temp_model_path)


def test_dat_detection():
    """Test DAT architecture detection"""
    print("\nTesting DAT architecture detection...")
    
    # Create a sample DAT state dict
    sample_state = {
        'conv_first.weight': torch.randn(180, 3, 3, 3),
        'layers.0.blocks.0.attn.qkv.weight': torch.randn(540, 180),
        'before_RG.1.weight': torch.randn(180),
        'layers.0.conv.weight': torch.randn(180, 180, 3, 3),
        'layers.0.blocks.0.norm1.weight': torch.randn(180),
        'layers.1.blocks.0.attn.qkv.weight': torch.randn(540, 180),
    }
    
    # Create upscaler with dummy paths
    upscaler = Upscale(
        model="dummy.pth",
        input=Path("./input"),
        output=Path("./output"),
        cpu=True
    )
    is_dat = upscaler._is_dat_architecture(sample_state)
    
    print(f"✓ DAT detection result: {is_dat}")
    
    if is_dat:
        from utils.architecture.DAT_variants import detect_dat_variant
        variant = detect_dat_variant(sample_state)
        print(f"✓ Detected DAT variant: {variant}")
    
    return is_dat


if __name__ == "__main__":
    print("=" * 60)
    print("DAT Architecture Support Test")
    print("=" * 60)
    
    # Test 1: DAT detection
    detection_success = test_dat_detection()
    
    # Test 2: Full model test
    model_success = test_dat_model_saving_and_loading()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  DAT Detection: {'✓ PASS' if detection_success else '✗ FAIL'}")
    print(f"  Model Loading: {'✓ PASS' if model_success else '✗ FAIL'}")
    print("=" * 60)
    
    if detection_success and model_success:
        print("🎉 All tests passed! DAT support is working correctly.")
        exit(0)
    else:
        print("❌ Some tests failed. Check the output above.")
        exit(1)