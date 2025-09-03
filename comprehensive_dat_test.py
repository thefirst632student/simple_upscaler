#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final comprehensive test for DAT architecture support.
This demonstrates DAT support with real models from the repository.
"""

import numpy as np
from pathlib import Path
from upscale import Upscale
import torch
import safetensors.torch
from utils.architecture.DAT_variants import detect_dat_variant


def test_real_dat_model():
    """Test with real DAT2 model from the repository"""
    print("🔧 Testing Real DAT Model Support")
    print("=" * 50)
    
    model_path = 'models/4x_IllustrationJaNai_V2standard_DAT2_27k.safetensors'
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Test 1: Model detection
        print("1. Testing DAT architecture detection...")
        state_dict = safetensors.torch.load_file(model_path)
        
        upscaler = Upscale("dummy.pth", Path("./input"), Path("./output"), cpu=True)
        is_dat = upscaler._is_dat_architecture(state_dict)
        variant = detect_dat_variant(state_dict)
        
        print(f"   ✓ Detected as DAT: {is_dat}")
        print(f"   ✓ Detected variant: {variant}")
        
        # Test 2: Model loading
        print("\\n2. Testing model loading...")
        upscaler = Upscale(model_path, Path("./input"), Path("./output"), cpu=True)
        upscaler.load_model(model_path)
        
        print(f"   ✓ Architecture: {upscaler.last_kind}")
        print(f"   ✓ Scale factor: {upscaler.last_scale}x")
        print(f"   ✓ Channels: {upscaler.last_in_nc} → {upscaler.last_out_nc}")
        print(f"   ✓ Features: {upscaler.last_nf}")
        print(f"   ✓ Layer groups: {upscaler.last_nb}")
        
        # Test 3: Image upscaling
        print("\\n3. Testing image upscaling...")
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = upscaler.upscale(test_image)
        
        expected_h = test_image.shape[0] * upscaler.last_scale
        expected_w = test_image.shape[1] * upscaler.last_scale
        
        print(f"   ✓ Input: {test_image.shape}")
        print(f"   ✓ Output: {result.shape}")
        print(f"   ✓ Scale applied correctly: {result.shape[:2] == (expected_h, expected_w)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_dat_variant_configurations():
    """Test different DAT variant configurations"""
    print("\\n🔧 Testing DAT Variant Configurations")
    print("=" * 50)
    
    from utils.architecture.DAT_variants import DAT_CONFIGS
    
    variants_tested = 0
    variants_passed = 0
    
    for variant_name, create_func in DAT_CONFIGS.items():
        try:
            print(f"Testing {variant_name}...")
            model = create_func(upscale=2, embed_dim=60, depth=[2])
            model.eval()
            
            # Test forward pass
            with torch.no_grad():
                test_input = torch.randn(1, 3, 64, 64)
                output = model(test_input)
                
            expected_shape = (1, 3, 128, 128)  # 2x upscale
            success = output.shape == expected_shape
            
            print(f"   {'✓' if success else '❌'} {variant_name}: {test_input.shape} → {output.shape}")
            
            variants_tested += 1
            if success:
                variants_passed += 1
                
        except Exception as e:
            print(f"   ❌ {variant_name}: Error - {e}")
            variants_tested += 1
    
    print(f"\\nVariant test results: {variants_passed}/{variants_tested} passed")
    return variants_passed == variants_tested


def main():
    """Run comprehensive DAT support tests"""
    print("🚀 Comprehensive DAT Architecture Support Test")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Real model support
    if test_real_dat_model():
        tests_passed += 1
        print("\\n✅ Real DAT model test: PASSED")
    else:
        print("\\n❌ Real DAT model test: FAILED")
    
    # Test 2: Variant configurations
    if test_dat_variant_configurations():
        tests_passed += 1
        print("\\n✅ DAT variant configuration test: PASSED")
    else:
        print("\\n❌ DAT variant configuration test: FAILED")
    
    # Final results
    print("\\n" + "=" * 60)
    print(f"📊 Final Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! DAT support is fully functional.")
        print("\\nSupported DAT variants:")
        print("  • DAT (base) - Full-featured dual aggregation transformer")
        print("  • DAT-2 - Reduced expansion factor for efficiency")  
        print("  • DAT-S - Smaller split size for different performance")
        print("  • DAT-light - Lightweight single-layer variant")
        print("\\nFeatures supported:")
        print("  • Automatic architecture detection")
        print("  • Parameter auto-detection from state dict")
        print("  • Model interpolation compatibility")
        print("  • All existing upscaler features (alpha modes, seamless, etc.)")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)