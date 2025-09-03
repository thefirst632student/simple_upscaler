# Safetensors Support

This document describes the safetensors support added to simple_manga_upscaler.

## Overview

The upscaler now supports loading models in both `.pth` (PyTorch) and `.safetensors` formats. This enhancement allows the use of newer model formats while maintaining full backward compatibility.

## Supported Formats

- **`.pth`** - Traditional PyTorch model files (existing support)
- **`.safetensors`** - Safer, faster model format with memory mapping support (newly added)

## Supported Architectures

The following model architectures are supported in both formats:

1. **ESRGAN** - Regular ESRGAN, "new-arch" ESRGAN, Real-ESRGAN v1
2. **RealESRGAN-v2** - SRVGGNet-based models
3. **SPSR** - ESRGAN with additional layers
4. **FDAT** - Fast Dual Aggregation Transformer (with dynamic parameter detection)

### FDAT Architecture Enhancements

The FDAT architecture support has been enhanced to automatically detect and handle different model variants:

- **FDAT M** - Medium variant with 120 embedding dimensions and 4 attention heads
- **FDAT XL** - Extra-large variant with 180 embedding dimensions and 6 attention heads
- Automatic parameter detection based on model state dictionary

## Unsupported Architectures

The following architectures are currently not supported:

- **DAT2** - Dual Attention Transformer v2 (requires different implementation)

When encountering unsupported architectures, the upscaler will display informative error messages.

## Usage

No changes to command-line usage are required. The upscaler automatically detects the file format:

```bash
# Using .pth models (existing)
python upscale.py 4x-AnimeSharp.pth

# Using .safetensors models (new)
python upscale.py 4x_IllustrationJaNai_V2standard_FDAT_M_52k.safetensors
python upscale.py 4x_IllustrationJaNai_V2standard_FDAT_XL_18k.safetensors

# Model interpolation works with both formats
python upscale.py "4x-AnimeSharp.pth@50&4x_eula_digimanga_bw_v2_nc1_307k.pth@50"
```

## Model Files in Repository

The repository includes several example models:

### .pth models:
- `4x-AnimeSharp.pth` - ESRGAN model for anime upscaling
- `4x_eula_digimanga_bw_v2_nc1_307k.pth` - ESRGAN model for manga upscaling

### .safetensors models:
- `4x_IllustrationJaNai_V2standard_FDAT_M_52k.safetensors` - FDAT Medium model
- `4x_IllustrationJaNai_V2standard_FDAT_XL_18k.safetensors` - FDAT XL model
- `4x_IllustrationJaNai_V2standard_DAT2_27k.safetensors` - DAT2 model (unsupported)

## Technical Implementation

### File Format Detection

The upscaler detects the file format based on the file extension:

- `.safetensors` files are loaded using `safetensors.torch.load_file()`
- All other files (`.pth`, `.pt`, etc.) are loaded using `torch.load()`

### Architecture Detection

Architecture detection remains the same, based on keys in the state dictionary:

- **RealESRGAN-v2**: `"params"` key and `"body.0.weight"` in params
- **FDAT**: `"conv_first.weight"` and `"groups.0.blocks.0.n1.weight"`
- **SPSR**: `"f_HR_conv1.0.weight"`
- **ESRGAN**: Default fallback for compatible models

### Dynamic Parameter Detection (FDAT)

For FDAT models, the following parameters are automatically detected:

- **Number of heads**: Determined from attention bias tensor dimensions
- **Embedding dimensions**: From `conv_first.weight` tensor shape
- **Scale factor**: From upsampler metadata or structure analysis
- **Input/output channels**: From first/last layer tensors

## Dependencies

The safetensors support requires the `safetensors` library:

```bash
pip install safetensors
```

This dependency has been added to `requirements.txt`.

## Testing

Comprehensive tests are available in `tests/test_safetensors_support.py`:

```bash
python tests/test_safetensors_support.py
```

The test suite validates:
- File format detection
- .pth compatibility (regression testing)
- .safetensors loading and processing
- Model interpolation functionality
- Error handling for unsupported architectures

## Migration Guide

Existing users can continue using `.pth` models without any changes. To use `.safetensors` models:

1. Ensure `safetensors` is installed (`pip install safetensors`)
2. Use `.safetensors` model files with the same command syntax
3. Supported architectures will be automatically detected and loaded

## Performance Notes

`.safetensors` format offers several advantages:

- **Faster loading**: Memory mapping reduces load times
- **Memory efficiency**: Better memory usage patterns
- **Safety**: Protection against pickle-based attacks
- **Cross-platform**: Better portability between systems

For large models like FDAT XL, the performance improvements can be significant.