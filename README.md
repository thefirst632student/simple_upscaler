# A fork of [joeyballentine's fork](https://github.com/joeyballentine/ESRGAN) of [BlueAmulet's fork](https://github.com/BlueAmulet/ESRGAN) of [ESRGAN by Xinntao](https://github.com/xinntao/ESRGAN)

This fork was created for using in personal projects on Google Colab. It includes some specific fixes related to Google Colab. It also includes some AI models.

It also includes a model "4x_eula_digimanga_bw_v2_nc1_307k" by "end user license agreement#9756" and "4x-AnimeSharp.pth" by Kim2091, both licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)

## Supported Architectures

This upscaler supports multiple model architectures:

- **ESRGAN**: Classic ESRGAN and Real-ESRGAN v1 models (.pth files)
- **Real-ESRGAN v2**: SRVGGNet-based models (includes "params" structure)
- **SPSR**: Enhanced ESRGAN with additional layers
- **FDAT**: Fast Dual Aggregation Transformer models (.safetensors)
- **DAT**: Dual Aggregation Transformer models (.safetensors)

## Dependencies and Installation

```
pip install --user -r requirements.txt
```
