# ComfyUI HDR VAE Decode Node

[![GitHub](https://img.shields.io/badge/GitHub-sumitchatterjee13%2Fvae--decode--hdr-blue?logo=github)](https://github.com/sumitchatterjee13/vae-decode-hdr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange)](https://github.com/comfyanonymous/ComfyUI)

A custom ComfyUI node that intelligently preserves HDR data from VAE models for professional VFX workflows.

## 📋 Table of Contents

- [Overview](#overview)
- [Technical Innovation](#technical-innovation)  
- [Installation](#installation)
- [Usage](#usage)
- [Built-in HDR Export](#built-in-hdr-export-solution)
- [Results & Performance](#results--performance)
- [Technical Achievements](#technical-achievements)
- [Development Notes](#development-notes)
- [Contributing](#contributing)
- [License](#license)

## Overview

After extensive research and development, I've created a node that solves the fundamental HDR preservation problem in VAE decoding. Instead of blindly bypassing layers, this implementation uses a **scientific approach** to understand and work with the VAE's natural behavior:

- **Intelligent Analysis**: Automatically analyzes the VAE's `conv_out` transformation to understand how HDR data is processed
- **Smart HDR Expansion**: Preserves the VAE's excellent tone mapping while selectively expanding highlight regions  
- **Multiple HDR Modes**: Conservative, Moderate, Exposure, and Aggressive modes for different workflows
- **Professional Quality**: Maintains perceptual image quality while extending dynamic range where needed
- **VFX Ready**: 32-bit float pipeline with proper HDR preservation for compositing workflows

## Problem Statement

ComfyUI's default VAE Decode node applies several limitations:
1. **Range Clamping**: EXR outputs are constrained to 0-1 pixel values
2. **Lost Dynamic Range**: High and low luminance information is compressed/clipped  
3. **8-bit Pipeline**: Effective output is 8-bit despite higher precision formats
4. **Color Space Constraints**: Limited to sRGB-like color spaces

## Technical Innovation

### Core Breakthrough
Through systematic analysis, I discovered that the VAE's `conv_out` layer applies sigmoid-like normalization that clamps HDR values to [0,1]. Rather than simply bypassing this layer, I developed an intelligent approach that:

1. **Analyzes the transformation** using forward hooks to capture pre/post conv_out data
2. **Detects the normalization pattern** (sigmoid, tanh, or custom)  
3. **Preserves base image quality** by using the VAE's excellent tone mapping as foundation
4. **Selectively expands highlights** only in regions that exceed the standard range

### Technical Features

- **Scientific Analysis**: Real-time conv_out transformation analysis with statistical profiling
- **MAX Pooling Channel Conversion**: Preserves HDR peaks when converting 128→3 channels (vs averaging which destroys brightness)
- **Smart Device Management**: Automatic CUDA/CPU synchronization across all processing stages
- **Multiple HDR Modes**: 
  - **Conservative**: Gentle 1.5x expansion, safest for general use (default)
  - **Moderate**: 3x smart expansion, balanced quality/range  
  - **Exposure**: Natural exposure-based HDR for compositing workflows
  - **Aggressive**: Full mathematical recovery for maximum range
- **Robust Fallback System**: Smart bypass → Simple bypass if intelligent methods fail
- **Professional Pipeline**: Float32 throughout with proper tensor formatting for ComfyUI

## Installation

### Method 1: Git Clone (Recommended)
1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sumitchatterjee13/vae-decode-hdr.git
```

2. Install minimal additional dependencies (ComfyUI already provides torch, torchvision, numpy):
```bash
cd vae-decode-hdr
pip install -r requirements.txt
```

3. Restart ComfyUI

### Method 2: Direct Download
1. [Download ZIP](https://github.com/sumitchatterjee13/vae-decode-hdr/archive/refs/heads/main.zip) from GitHub
2. Extract to `ComfyUI/custom_nodes/vae-decode-hdr/`
3. Install dependencies: `pip install -r requirements.txt`
4. Restart ComfyUI

The nodes should appear in the **latent** category as "HDR VAE Decode" and in the **image** category as "Linear EXR Export".

## Usage

The **HDR VAE Decode** node will appear in the "latent" category in ComfyUI.

### Parameters:
- **samples**: Latent samples from your pipeline
- **vae**: The VAE model (tested with Flux.1 Dev, compatible with other VAE models)
- **hdr_mode** *(optional)*: HDR processing mode:
  - `conservative`: Gentle 1.5x expansion, maintains natural appearance (default)
  - `moderate`: 3x smart expansion, balanced for most VFX work
  - `exposure`: Natural exposure-based HDR using exposure stops  
  - `aggressive`: Full mathematical recovery, maximum dynamic range
- **max_range** *(optional)*: Final output clamp range (1-1000, default: 50)
- **scale_factor** *(optional)*: Additional scaling multiplier (0.1-10x, default: 1.0)
- **enable_negatives** *(optional)*: Allow negative values for advanced workflows (default: false)

### Expected Results:
The node first attempts **intelligent HDR decode** using scientific analysis of the VAE's behavior. If this succeeds, you'll get natural-looking images with selectively expanded highlights. If it fails, the system falls back to the robust bypass method that directly processes VAE decoder layers.

### Output:
- **image**: Professional-quality HDR image tensor with preserved dynamic range, ready for EXR export or further processing

## Example Workflow

For typical VFX work:

1. **Connect your Flux.1 latents** to the HDR VAE Decode node
2. **Use default "conservative" mode** for natural results, or choose "moderate" for higher dynamic range
3. **Leave other parameters default** for most use cases
4. **Connect output to the Linear EXR Export node** to save professional EXR files
5. **Use in your compositing software** (Nuke, After Effects, etc.) with proper HDR handling

The node will automatically:
- Analyze the VAE's behavior
- Apply intelligent HDR expansion to highlight regions  
- Preserve natural image appearance
- Output HDR data for professional workflows

## Built-in HDR Export Solution

This package now includes a **Linear EXR Export** node for professional HDR output:

### Linear EXR Export Node Features:
- **True 32-bit EXR export** with preserved HDR values above 1.0
- **Professional VFX quality** - maintains linear color space
- **Smart path handling**: 
  - Empty path → saves to `ComfyUI/output/`
  - `/subfolder` → saves to `ComfyUI/output/subfolder/` 
  - Full path → uses custom absolute/relative path
- **Multiple bit depths**: 32-bit (maximum quality) or 16-bit (smaller files)
- **Compression options**: ZIP, PIZ, RLE, PXR24, or none
- **Clean file naming** with customizable prefixes and counters
- **HDR verification**: Automatically verifies HDR values are preserved in saved files
- **Seamless integration**: Designed specifically for HDR VAE Decode output

### Complete HDR Workflow:
**HDR VAE Decode** → **Linear EXR Export** → **Professional EXR files** ready for compositing in Nuke, After Effects, or other VFX software.

The Linear EXR Export node will appear in the **image** category in ComfyUI.

### Additional HDR Export Option

For advanced HDR processing workflows with multi-exposure fusion, you can also use the **HDR Export node** from the [Luminance Stack Processor](https://github.com/sumitchatterjee13/Luminance-Stack-Processor) package available through ComfyUI Manager.

## Results & Performance

The intelligent HDR approach successfully preserves HDR data while maintaining professional image quality:

- **HDR Preservation**: Maintains hundreds of thousands of HDR pixels (>1.0 values) through the entire pipeline
- **Dynamic Range**: Conservative mode (default): 1.5x range, Moderate: up to 9.0x, Exposure/Aggressive: 10+ range
- **Image Quality**: Natural appearance with selective highlight expansion, not false color artifacts
- **Processing Speed**: ~40-42 seconds for 752×1328 images (similar to standard VAE decode)
- **Memory Efficiency**: Float32 pipeline with smart device management (CUDA/CPU synchronization)

## Technical Achievements  

Through this project, I solved several challenging problems:

1. **Identified the root cause**: VAE's `conv_out` layer applies sigmoid normalization, not simple clamping
2. **Developed MAX pooling**: Preserves HDR peaks during 128→3 channel conversion (vs averaging)
3. **Created intelligent expansion**: Uses VAE's tone mapping + selective highlight extension  
4. **Solved device synchronization**: Proper CUDA/CPU tensor management across processing stages
5. **Built robust fallbacks**: Multiple processing paths ensure reliability

## Compatibility

- **Primary Target**: Flux.1 Dev VAE model
- **Tested With**: ComfyUI stable versions
- **Requirements**: Python 3.8+, PyTorch, CUDA-capable GPU recommended
- **Formats**: Outputs compatible with EXR export nodes for professional workflows

## Development Notes

This project represents months of research into VAE architecture and HDR processing. The approach evolved from simple layer bypassing to sophisticated analysis-based processing. While the current implementation works well with Flux.1 Dev, different VAE models may require adjustments to the analysis logic.

Key learnings:
- **Simple bypassing often breaks image quality** - the VAE's processing pipeline exists for good reasons
- **Averaging destroys HDR data** - MAX pooling is essential for preserving brightness peaks
- **Device synchronization is critical** - mixed CPU/CUDA operations cause subtle failures
- **Multiple fallback paths improve reliability** - complex pipelines need robust error handling

## Contributing

This project welcomes contributions! If you encounter issues or have improvements:

- 🐛 **Report bugs**: [Open an issue](https://github.com/sumitchatterjee13/vae-decode-hdr/issues) 
- 💡 **Feature requests**: [Suggest enhancements](https://github.com/sumitchatterjee13/vae-decode-hdr/issues)
- 🔧 **Code contributions**: [Submit a pull request](https://github.com/sumitchatterjee13/vae-decode-hdr/pulls)
- 📖 **Documentation**: Help improve docs and examples

### How to Contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

This is an experimental node that pushes the boundaries of what's possible with VAE decoding - your contributions help advance HDR processing in ComfyUI!

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/sumitchatterjee13/vae-decode-hdr/blob/main/LICENSE) file for details.

### MIT License Summary:
✅ Commercial use  
✅ Modification  
✅ Distribution  
✅ Private use  

---

**Repository**: [https://github.com/sumitchatterjee13/vae-decode-hdr](https://github.com/sumitchatterjee13/vae-decode-hdr)  
**Author**: Sumit Chatterjee  
**Created**: 2024  

⭐ If this project helps you, please consider giving it a star on GitHub!
