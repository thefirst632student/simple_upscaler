# FDAT (Fast Dual Aggregation Transformer) Architecture

## Overview

This repository now includes support for the Fast Dual Aggregation Transformer (FDAT) architecture, a state-of-the-art transformer-based model for image super-resolution. FDAT combines spatial and channel attention mechanisms to achieve high-quality upscaling results.

## Features

- **Multiple Upsampling Methods**: Supports various upsampling techniques including:
  - `transpose+conv` - Transposed convolution with convolution
  - `pixelshuffle` - Pixel shuffle upsampling
  - `pixelshuffledirect` - Direct pixel shuffle
  - `nearest+conv` - Nearest neighbor interpolation with convolution
  - `dysample` - Dynamic sampling (experimental)
  - `lda` - Learned deformable attention upsampling
  - `pa_up` - Progressive attention upsampling

- **Flexible Architecture**: Configurable parameters including:
  - Embedding dimensions
  - Number of transformer groups
  - Attention heads and window sizes
  - Scale factors (2x, 3x, 4x, etc.)

- **Efficient Implementation**: Optimized transformer blocks with simplified attention mechanisms

## Usage

### Basic Usage

The FDAT architecture integrates seamlessly with the existing ESRGAN upscaler. Simply use a FDAT model file:

```bash
python upscale.py path/to/fdat_model.pth --input input_folder --output output_folder
```

### Model Detection

The upscaler automatically detects FDAT models based on their state dictionary structure. Models with the following keys are identified as FDAT:
- `conv_first.weight` (first convolution layer)
- `groups.0.blocks.0.n1.weight` (transformer block layer norm)

### Creating FDAT Models

```python
from utils.architecture.FDAT import FDAT

# Create a 4x upscaling model
model = FDAT(
    num_in_ch=3,          # Input channels (RGB)
    num_out_ch=3,         # Output channels (RGB)
    scale=4,              # Upscaling factor
    embed_dim=120,        # Embedding dimension
    num_groups=4,         # Number of transformer groups
    depth_per_group=3,    # Blocks per group
    upsampler_type="transpose+conv"  # Upsampling method
)

# Save the model
torch.save(model.state_dict(), "my_fdat_4x.pth")
```

## Technical Details

### Architecture Components

1. **FastSpatialWindowAttention**: Efficient window-based spatial attention
2. **FastChannelAttention**: Channel-wise attention mechanism
3. **SimplifiedAIM**: Attention interaction module
4. **SimplifiedFFN**: Feed-forward network with spatial mixing
5. **UniUpsampleV3**: Unified upsampling module with multiple methods

### Dependencies

- `torch` - PyTorch framework
- `einops` - Tensor operations (automatically installed)
- Standard Python libraries (numpy, math, typing)

### Model Parameters

FDAT models automatically expose the following properties for compatibility:
- `scale` - Upscaling factor
- `in_nc` - Input channels
- `out_nc` - Output channels
- `num_feat` - Number of features
- `num_blocks` - Number of transformer groups

## Compatibility

FDAT models are fully compatible with all existing ESRGAN features:
- Model interpolation
- Alpha channel processing
- Seamless tiling
- CPU/GPU inference
- FP16 support

## Performance Notes

- FDAT models may require more memory than traditional ESRGAN models due to transformer operations
- Performance scales with embedding dimension and number of groups
- Use smaller configurations for faster inference or larger for better quality

## Examples

### Small, Fast Model (2x upscaling)
```python
FDAT(scale=2, embed_dim=48, num_groups=2, depth_per_group=2)
```

### High Quality Model (4x upscaling)
```python
FDAT(scale=4, embed_dim=180, num_groups=6, depth_per_group=4)
```

### Custom Upsampler
```python
FDAT(scale=4, embed_dim=120, upsampler_type="pixelshuffle")
```