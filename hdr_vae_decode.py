"""
Simple HDR VAE Decode

Simple VAE decode that preserves wider dynamic range (0-50) without normalization to 0-1.
Author: Sumit Chatterjee
Contributor: Antonio Neto
  Version: 1.1.5
Semantic Versioning: MAJOR.MINOR.PATCH
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple
import logging
from kornia.core import ImageModule as Module
from kornia.core import Tensor

# --- Global/Class-level Logger Setup ---
# Configure the logger once at the class definition level or module level
logger = logging.getLogger("HDRVAEDecode")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[HDR VAE] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class HDRVAEDecode:
    """
    Advanced HDR VAE Decode node for professional VFX workflows.
    
    Features:
    - Scientific conv_out analysis with intelligent HDR recovery
    - Multiple HDR modes: Conservative, Exposure, mathematical_recovery
    - Smart highlight expansion preserving base image quality
    - Exposure-based HDR for natural compositing workflows
    - Smart bypass fallback for maximum compatibility
    - Float32 pipeline throughout for maximum precision
    """
    
    def __init__(self):
        self.logger = logger
        self.NORMALIZATION_FUNCTION = str()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
            "optional": {
                "hdr_mode": (["conservative", "exposure", "adaptive_recovery", "mathematical_recovery"],
                             {"default": "mathematical_recovery",
                              "tooltip": "conservative: Gentle conservative_ev_multiplier expansion, safest for general use \n "
                                         "exposure: Natural exposure-based HDR for compositing workflows \n "
                                         "mathematical_recovery: Full mathematical recovery, maximum range"}),
                "conservative_ev_multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Expansion multiplier for the conservative mode."}),
                "enable_negatives": ("BOOLEAN", {"default": False, "tooltip": "Will display any value that is below 0.0"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",) 
    FUNCTION = "simple_hdr_decode"
    CATEGORY = "latent"

    def simple_hdr_decode(
        self,
        samples: Dict[str, torch.Tensor],
        vae: Any,
        hdr_mode: str = "mathematical_recovery",
        conservative_ev_multiplier: float = 1.0,
        enable_negatives: bool = False,
    ) -> Tuple[torch.Tensor]:
        """
        HDR VAE decode with intelligent conv_out analysis and multiple HDR modes.
        HDR Modes:
        - conservative: Gentle conservative_ev_multiplier expansion, safest for general use
        - exposure: Natural exposure-based HDR for compositing workflows
        - mathematical_recovery: Full mathematical recovery, maximum range
        Features smart highlight expansion preserving base image perceptual quality.
        """
        
        latent = samples["samples"]
        
        # Log input stats
        latent_min = float(torch.min(latent))
        latent_max = float(torch.max(latent))
        
        self.logger.info(f"INPUT LATENT: range=[{latent_min:.3f}, {latent_max:.3f}]")
        
        # FIRST: Analyze what conv_out actually does (NEW SCIENTIFIC APPROACH)
        self.logger.info("🔬 STEP 1: Analyzing conv_out transformation...")
        analysis_result = self.analyze_conv_out(vae, latent)
        
        if analysis_result is not None:
            self.logger.info("✅ Analysis complete! Now we understand the transformation...")
            
            # Try intelligent HDR decode first based on analysis
            try:
                self.logger.info("🧠 STEP 2: Attempting INTELLIGENT HDR decode based on conv_out analysis...")
                decoded = self.intelligent_hdr_decode(vae, latent, analysis_result, hdr_mode)
                
                # Check if we got HDR values
                hdr_pixels = int(torch.sum(decoded > 1.0))
                decode_min = float(torch.min(decoded))
                decode_max = float(torch.max(decoded))
                
                self.logger.info(f"🎯 INTELLIGENT DECODE: range=[{decode_min:.3f}, {decode_max:.3f}], HDR pixels: {hdr_pixels}")
                
                if hdr_pixels > 0 or decode_max > 1.1:  # Lower threshold - be more accepting
                    self.logger.info("✅ INTELLIGENT decode succeeded! Using this result.")
                    # Skip bypass and use intelligent result
                    use_bypass = False
                else:
                    self.logger.info("⚠️ INTELLIGENT decode didn't produce significant HDR values, trying bypass...")
                    use_bypass = True
                    
            except Exception as e:
                self.logger.error(f"❌ INTELLIGENT decode failed: {str(e)}")
                import traceback
                self.logger.error(f"🔍 ERROR DETAILS: {traceback.format_exc()}")
                self.logger.info("🔄 Falling back to bypass...")
                use_bypass = True
        else:
            self.logger.warning("❌ Analysis failed - falling back to bypass")
            use_bypass = True
            
        # Try custom decoder bypass only if intelligent decode didn't work
        if use_bypass:
            self.logger.info("🔄 STEP 3: Falling back to bypass decode approach...")
            try:
                import threading
                import time
                
                # Use threading to implement timeout for bypass decode
                result = [None]
                exception = [None]
                
                def bypass_worker():
                    try:
                        result[0] = self.bypass_conv_out_decode(vae, latent)
                    except Exception as e:
                        exception[0] = e
                
                self.logger.info("🕒 Starting bypass decode with 30s timeout...")
                thread = threading.Thread(target=bypass_worker)
                thread.daemon = True
                thread.start()
                thread.join(timeout=30)  # 30 second timeout
                
                if thread.is_alive():
                    self.logger.warning("⏰ Bypass decode timed out after 30s, falling back to simple bypass")
                    raise RuntimeError("Bypass decode timeout")
                elif exception[0]:
                    raise exception[0]
                else:
                    decoded = result[0]
                
                decode_min = float(torch.min(decoded))
                decode_max = float(torch.max(decoded))
                hdr_pixels = int(torch.sum(decoded > 1.0))
                self.logger.info(f"🎯 BYPASS DECODE: range=[{decode_min:.3f}, {decode_max:.3f}], HDR pixels: {hdr_pixels}")
                
            except Exception as e:
                self.logger.error(f"Full bypass failed: {str(e)}, trying simple bypass...")
                # Try simple bypass first
                try:
                    decoded = self.simple_bypass_decode(vae, latent)
                    
                    decode_min = float(torch.min(decoded))
                    decode_max = float(torch.max(decoded))
                    hdr_pixels = int(torch.sum(decoded > 1.0))
                    self.logger.info(f"🚀 SIMPLE BYPASS: range=[{decode_min:.3f}, {decode_max:.3f}], HDR pixels: {hdr_pixels}")
                    
                except Exception as e2:
                    self.logger.error(f"Simple bypass failed: {str(e2)}")
                    self.logger.error("🚨 CRITICAL: Both smart and simple bypass failed - this indicates a fundamental issue!")
                    raise RuntimeError(f"HDR bypass failed. Smart bypass error: {str(e)}, Simple bypass error: {str(e2)}")
        else:
            # We already have the decoded result from intelligent decode
            self.logger.info("✅ Using INTELLIGENT decode result (skipped bypass)")
        
        # Apply scale factor if specified
        if conservative_ev_multiplier != 1.0:
            decoded = decoded * conservative_ev_multiplier
            self.logger.info(f"Applied ev multiplication of: {conservative_ev_multiplier}")
        
        # Clamp to wider range, allowing negatives if enabled
        if enable_negatives:
            hdr_image = torch.clamp(decoded, analysis_result['pre_stats']['min'], analysis_result['pre_stats']['max'])
            self.logger.info(f"Clamping to range: [{analysis_result['pre_stats']['min']:.3f}, {analysis_result['pre_stats']['max']:.3f}] (negatives enabled)")
        else:
            hdr_image = torch.clamp(decoded, 0, analysis_result['pre_stats']['max'])
            self.logger.info(f"Clamping to range: [0.000, {analysis_result['pre_stats']['max']:.3f}]")
        
        # Format tensor for ComfyUI
        formatted = self._format_tensor(hdr_image)
        
        # Final stats
        final_min = float(torch.min(formatted))
        final_max = float(torch.max(formatted))
        hdr_pixels = int(torch.sum(formatted > 1.0))
        negative_pixels = int(torch.sum(formatted < 0.0))

        self.logger.info(f"OUTPUT: range=[{final_min:.3f}, {final_max:.3f}], HDR pixels: {hdr_pixels}, Negative pixels: {negative_pixels}")
        
        return (formatted,)

    def _format_tensor(self, tensor):
        """Format tensor for ComfyUI with enhanced debugging."""
        self.logger.info(f"🔧 FORMATTING: Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        
        # Handle different tensor formats from bypass
        if tensor.dim() == 4:
            # CRITICAL: Detect if tensor is already in ComfyUI format [batch, height, width, channels]
            # vs PyTorch format [batch, channels, height, width]
            shape = tensor.shape
            self.logger.info(f"📐 4D tensor shape: {shape}")
            
            # Check if last dimension is 3 (likely already in ComfyUI format)
            if shape[-1] == 3:
                # Already in [batch, height, width, 3] format
                formatted = tensor
                self.logger.info(f"✅ ALREADY ComfyUI FORMAT: {shape} (batch, height, width, 3)")
            # Check if second dimension is 3 (PyTorch format with RGB)
            elif shape[1] == 3:
                # [batch, 3, height, width] -> [batch, height, width, 3]
                formatted = tensor.permute(0, 2, 3, 1)
                self.logger.info(f"✅ CONVERTED PyTorch->ComfyUI: {shape} -> {formatted.shape}")
            # Check if second dimension > 3 (PyTorch format with many channels)
            elif shape[1] > 3 and shape[1] < 2000:  # Reasonable channel count
                channels = shape[1]
                self.logger.info(f"🔧 PyTorch format with {channels} channels, converting to RGB PRESERVING HDR")
                
                # 🎯 CRITICAL HDR PRESERVATION: Don't just take first 3 channels!
                if channels == 128:
                    # Special handling for VAE decoder output (128 channels)
                    # Use weighted combination to preserve HDR information
                    self.logger.info("🎯 APPLYING HDR-PRESERVING 128->3 CONVERSION")
                    
                    # 🎯 NEW METHOD: MAX POOLING to preserve HDR peaks instead of averaging
                    # Averaging destroys HDR range - use MAX to preserve bright values!
                    channels_per_rgb = channels // 3  # 42-43 channels per RGB channel
                    r_channels = tensor[:, 0:42, :, :]  # Red from channels 0-41
                    g_channels = tensor[:, 42:84, :, :]  # Green from channels 42-83  
                    b_channels = tensor[:, 84:126, :, :] # Blue from channels 84-125
                    
                    # Use MAX pooling to preserve HDR peaks (bright values)
                    r, _ = torch.max(r_channels, dim=1, keepdim=True)
                    g, _ = torch.max(g_channels, dim=1, keepdim=True)
                    b, _ = torch.max(b_channels, dim=1, keepdim=True)
                    
                    # Combine to RGB
                    rgb_tensor = torch.cat([r, g, b], dim=1)
                    
                    # Log the preservation
                    orig_min = float(torch.min(tensor))
                    orig_max = float(torch.max(tensor))
                    rgb_min = float(torch.min(rgb_tensor))
                    rgb_max = float(torch.max(rgb_tensor))
                    orig_hdr = int(torch.sum(tensor > 1.0))
                    rgb_hdr = int(torch.sum(rgb_tensor > 1.0))
                    
                    self.logger.info(f"📊 BEFORE CONVERSION: range=[{orig_min:.3f}, {orig_max:.3f}], HDR pixels: {orig_hdr}")
                    self.logger.info(f"📊 AFTER CONVERSION:  range=[{rgb_min:.3f}, {rgb_max:.3f}], HDR pixels: {rgb_hdr}")
                    
                    formatted = rgb_tensor.permute(0, 2, 3, 1)
                    self.logger.info(f"✅ HDR-PRESERVED CONVERSION: {channels}ch->3ch: {shape} -> {formatted.shape}")
                else:
                    # For other channel counts, use MAX pooling approach
                    self.logger.info(f"🔧 APPLYING HDR-PRESERVING conversion for {channels} channels")
                    step = channels // 3
                    
                    # Group channels and use MAX pooling to preserve HDR
                    r_group = tensor[:, 0:step, :, :]
                    g_group = tensor[:, step:step*2, :, :]  
                    b_group = tensor[:, step*2:step*3, :, :]
                    
                    # Use MAX pooling instead of simple selection
                    r, _ = torch.max(r_group, dim=1, keepdim=True)
                    g, _ = torch.max(g_group, dim=1, keepdim=True)
                    b, _ = torch.max(b_group, dim=1, keepdim=True)
                    
                    rgb_tensor = torch.cat([r, g, b], dim=1)
                    
                    # Log HDR preservation stats
                    orig_min = float(torch.min(tensor))
                    orig_max = float(torch.max(tensor))
                    rgb_min = float(torch.min(rgb_tensor))
                    rgb_max = float(torch.max(rgb_tensor))
                    orig_hdr = int(torch.sum(tensor > 1.0))
                    rgb_hdr = int(torch.sum(rgb_tensor > 1.0))
                    
                    self.logger.info(f"📊 BEFORE CONVERSION: range=[{orig_min:.3f}, {orig_max:.3f}], HDR pixels: {orig_hdr}")
                    self.logger.info(f"📊 AFTER CONVERSION:  range=[{rgb_min:.3f}, {rgb_max:.3f}], HDR pixels: {rgb_hdr}")
                    
                    formatted = rgb_tensor.permute(0, 2, 3, 1)
                    self.logger.info(f"✅ DISTRIBUTED CONVERSION: {channels}ch->3ch: {shape} -> {formatted.shape}")
            # Check if first dimension > 3 (likely misinterpreted format)
            elif shape[0] == 1 and shape[1] > shape[3] and shape[3] == 3:
                # This is [batch=1, height, width, 3] - already correct!
                formatted = tensor
                self.logger.info(f"✅ CONFIRMED ComfyUI FORMAT: {shape} (batch=1, height, width, 3)")
            else:
                # Fallback: assume it's in PyTorch format and needs conversion
                self.logger.warning(f"⚠️ UNKNOWN 4D format: {shape}, assuming PyTorch format")
                if shape[1] >= 3:
                    # Use safer distributed channel selection instead of just first 3
                    channels = shape[1]
                    if channels >= 128:
                        # Large channel count - use MAX pooling approach for HDR preservation
                        self.logger.info(f"🔧 FALLBACK HDR-preserving conversion for {channels} channels")
                        step = channels // 3
                        
                        # Group and use MAX pooling
                        r_group = tensor[:, 0:step, :, :]
                        g_group = tensor[:, step:step*2, :, :]
                        b_group = tensor[:, step*2:step*3, :, :]
                        
                        r, _ = torch.max(r_group, dim=1, keepdim=True)
                        g, _ = torch.max(g_group, dim=1, keepdim=True)
                        b, _ = torch.max(b_group, dim=1, keepdim=True)
                        
                        rgb_tensor = torch.cat([r, g, b], dim=1)
                        
                        # Log HDR preservation stats
                        orig_min = float(torch.min(tensor))
                        orig_max = float(torch.max(tensor))
                        rgb_min = float(torch.min(rgb_tensor))
                        rgb_max = float(torch.max(rgb_tensor))
                        orig_hdr = int(torch.sum(tensor > 1.0))
                        rgb_hdr = int(torch.sum(rgb_tensor > 1.0))
                        
                        self.logger.info(f"📊 BEFORE CONVERSION: range=[{orig_min:.3f}, {orig_max:.3f}], HDR pixels: {orig_hdr}")
                        self.logger.info(f"📊 AFTER CONVERSION:  range=[{rgb_min:.3f}, {rgb_max:.3f}], HDR pixels: {rgb_hdr}")
                        
                        formatted = rgb_tensor.permute(0, 2, 3, 1)
                        self.logger.info(f"🔧 FALLBACK HDR-safe conversion: {shape} -> {formatted.shape}")
                    else:
                        # Small channel count - safer to use first 3
                        formatted = tensor[:, :3, :, :].permute(0, 2, 3, 1)
                        self.logger.info(f"🔧 FALLBACK simple conversion: {shape} -> {formatted.shape}")
                else:
                    formatted = tensor
                    self.logger.warning(f"⚠️ FALLBACK: keeping original format {shape}")
        elif tensor.dim() == 3:
            # [height, width, channels] - already in ComfyUI format
            formatted = tensor.unsqueeze(0)  # Add batch dimension
            self.logger.info(f"✅ 3D->4D: {tensor.shape} -> {formatted.shape}")
        else:
            self.logger.error(f"❌ UNEXPECTED tensor dimensions: {tensor.shape}")
            formatted = tensor

        # Ensure final format is correct (only if not already 3 channels)
        if formatted.dim() == 4 and formatted.shape[-1] != 3:
            self.logger.warning(f"⚠️ Final tensor channels != 3: {formatted.shape}")
            # Only force to 3 channels if it's not already in the right format
            if formatted.shape[-1] > 3:
                formatted = formatted[..., :3]  # Take first 3 channels
                self.logger.info(f"🔧 Trimmed to 3 channels: {formatted.shape}")
            elif formatted.shape[-1] == 1:
                formatted = formatted.repeat(1, 1, 1, 3)  # Expand to RGB
                self.logger.info(f"🔧 Expanded 1->3 channels: {formatted.shape}")
        elif formatted.dim() == 4 and formatted.shape[-1] == 3:
            self.logger.info(f"✅ Perfect format: {formatted.shape} (batch, height, width, 3)")
            
        result = formatted.contiguous().float()
        self.logger.info(f"🎯 FINAL FORMAT: {result.shape}, dtype: {result.dtype}, range: [{float(torch.min(result)):.3f}, {float(torch.max(result)):.3f}]")
        
        return result

    def inspect_vae_architecture(self, vae):
        """Inspect VAE structure to find where normalization happens."""
        
        self.logger.info("=== VAE ARCHITECTURE INSPECTION ===")
        
        # Print overall structure
        self.logger.info(f"VAE type: {type(vae)}")
        self.logger.info(f"VAE attributes: {[attr for attr in dir(vae) if not attr.startswith('_')]}")
        
        # Look for the actual model
        if hasattr(vae, 'first_stage_model'):
            model = vae.first_stage_model
            self.logger.info(f"First stage model type: {type(model)}")
            self.logger.info(f"First stage attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
            
            # Look for decoder
            if hasattr(model, 'decoder'):
                decoder = model.decoder
                self.logger.info(f"Decoder type: {type(decoder)}")
                self.logger.info("Decoder structure:")
                for name, module in decoder.named_modules():
                    if name:  # Skip empty names
                        self.logger.info(f"  {name}: {type(module).__name__}")
                        
                        # Check for activation functions that might clamp values
                        if 'sigmoid' in str(type(module)).lower():
                            self.logger.info(f"    ⚠️  FOUND SIGMOID - this clamps to 0-1!")
                        elif 'tanh' in str(type(module)).lower():
                            self.logger.info(f"    ⚠️  FOUND TANH - this clamps to -1,1!")
                        elif 'conv' in str(type(module)).lower():
                            self.logger.info(f"    Conv layer - check final layer")
                
            # Look for decode method
            if hasattr(model, 'decode'):
                import inspect
                try:
                    sig = inspect.signature(model.decode)
                    self.logger.info(f"Decode method signature: {sig}")
                except:
                    self.logger.info("Could not get decode method signature")
                    
            # Print full decoder structure
            if hasattr(model, 'decoder'):
                self.print_model_structure(model.decoder, max_depth=4)
        
        # Check for other common VAE structures
        for attr in ['model', 'vae', 'autoencoder', 'ae']:
            if hasattr(vae, attr):
                self.logger.info(f"Found {attr}: {type(getattr(vae, attr))}")

    def print_model_structure(self, model, max_depth=3):
        """Print detailed model structure."""
        
        def print_modules(module, prefix="", depth=0):
            if depth > max_depth:
                return
                
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                module_type = type(child).__name__
                self.logger.info(f"{'  ' * depth}{full_name}: {module_type}")
                
                # Highlight potential clamping layers
                if 'sigmoid' in module_type.lower():
                    self.logger.info(f"{'  ' * (depth+1)}⚠️  SIGMOID - clamps output to 0-1!")
                elif 'tanh' in module_type.lower():
                    self.logger.info(f"{'  ' * (depth+1)}⚠️  TANH - clamps output to -1,1!")
                elif 'conv' in module_type.lower() and len(list(child.children())) == 0:
                    # Final conv layer might be the output layer
                    self.logger.info(f"{'  ' * (depth+1)}🔍 Final conv layer - potential output layer")
                
                # Print parameters info if it's a leaf module
                if len(list(child.children())) == 0:
                    try:
                        params = sum(p.numel() for p in child.parameters())
                        self.logger.info(f"{'  ' * (depth+1)}Parameters: {params}")
                    except:
                        pass
                    
                print_modules(child, full_name, depth + 1)
        
        self.logger.info("=== MODEL STRUCTURE ===")
        print_modules(model)

    def smart_bypass_decode(self, vae, latent):
        """UPDATED: Smart bypass that properly handles channel reduction first."""
        
        self.logger.info("🧠 SMART bypass v2 - handle channel reduction correctly")
        
        decoder = vae.first_stage_model.decoder
        
        with torch.inference_mode():
            # Force CUDA device detection and debugging
            if hasattr(decoder, 'conv_in'):
                detected_device = next(decoder.conv_in.parameters()).device
                detected_dtype = next(decoder.conv_in.parameters()).dtype
                
                # Debug device detection
                self.logger.info(f"🔍 DETECTED VAE device: {detected_device}, dtype: {detected_dtype}")
                
                # Force CUDA if available and VAE is on CPU (ComfyUI usually has VAE on CUDA)
                if detected_device.type == 'cpu' and torch.cuda.is_available():
                    self.logger.warning(f"⚠️ VAE detected on CPU but CUDA available - checking for CUDA VAE...")
                    # Try to find CUDA parameters in the VAE
                    cuda_device = None
                    for name, param in decoder.named_parameters():
                        if param.device.type == 'cuda':
                            cuda_device = param.device
                            self.logger.info(f"✅ FOUND CUDA parameter: {name} on {cuda_device}")
                            break
                    
                    if cuda_device is not None:
                        device = cuda_device
                        dtype = detected_dtype
                        self.logger.info(f"🚀 FORCING CUDA: Using {device} instead of CPU")
                    else:
                        # Fallback to CUDA:0 if available
                        device = torch.device('cuda:0')
                        dtype = detected_dtype
                        self.logger.info(f"🚀 CUDA FALLBACK: Using {device}")
                else:
                    device = detected_device
                    dtype = detected_dtype
                
                # Move latent to the correct device
                original_device = latent.device
                latent = latent.to(device=device, dtype=dtype)
                self.logger.info(f"🔧 Latent: {original_device} → {device}, {dtype}")
                
                # Verify the decoder is on the same device
                conv_device = next(decoder.conv_in.parameters()).device
                if conv_device != device:
                    self.logger.error(f"❌ DEVICE MISMATCH: decoder on {conv_device}, latent on {device}")
                    # Try to move decoder to match latent
                    try:
                        self.logger.info(f"🔧 MOVING decoder from {conv_device} to {device}")
                        decoder = decoder.to(device)
                        conv_device_after = next(decoder.conv_in.parameters()).device
                        self.logger.info(f"✅ MOVED decoder to {conv_device_after}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to move decoder: {str(e)}")
                        raise RuntimeError(f"Device mismatch: decoder on {conv_device}, latent on {device}")
                else:
                    self.logger.info(f"✅ DEVICE MATCH: Both on {device}")
            
            # Input conv
            h = decoder.conv_in(latent)
            self.logger.info(f"After conv_in: range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
            
            # Process middle blocks (skip only attention)
            if hasattr(decoder, 'mid'):
                h = decoder.mid.block_1(h)
                self.logger.info(f"After mid.block_1: range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                
                # Skip attention to avoid hangs
                self.logger.info("🚫 SKIPPING mid.attn_1 (attention) to avoid hangs")
                
                h = decoder.mid.block_2(h)
                self.logger.info(f"After mid.block_2: range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
            
            # NEW APPROACH: Find and apply channel reduction FIRST
            if hasattr(decoder, 'up') and len(decoder.up) > 0:
                first_up = decoder.up[0]
                
                # Try to find the module that handles 512 → 256 reduction
                channel_reducer = None
                for name, module in first_up.named_children():
                    if hasattr(module, 'weight') and len(module.weight.shape) >= 2:
                        if module.weight.shape[1] == 512:  # Input channels = 512
                            channel_reducer = module
                            self.logger.info(f"🎯 FOUND channel reducer: {name} - {module.weight.shape}")
                            break
                
                if channel_reducer is not None:
                    # Apply channel reduction first
                    h = channel_reducer(h)
                    self.logger.info(f"After channel reduction: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                    
                    # Now apply ResNet blocks (they should work with reduced channels)
                    if hasattr(first_up, 'block'):
                        for j, block in enumerate(first_up.block):
                            h = block(h)
                            self.logger.info(f"  up[0].block[{j}]: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                    
                    # CRITICAL DEBUG: Check what components up[0] actually has
                    self.logger.info("🔍 INSPECTING up[0] components:")
                    up0_components = list(first_up.named_children())
                    for comp_name, comp_module in up0_components:
                        self.logger.info(f"  - {comp_name}: {type(comp_module)}")
                    
                    # Apply UPSAMPLING components (critical for spatial resolution)  
                    self.logger.info("🚀 STARTING up[0] upsampling processing...")
                    if hasattr(first_up, 'upsample'):
                        if first_up.upsample is not None:
                            self.logger.info(f"✅ up[0] HAS upsample: {type(first_up.upsample)}")
                            try:
                                h = first_up.upsample(h)
                                self.logger.info(f"🔧 up[0].upsample SUCCESS: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                            except Exception as e:
                                self.logger.error(f"❌ up[0].upsample FAILED: {str(e)}")
                                raise e
                        else:
                            self.logger.warning(f"⚠️ up[0].upsample is None")
                    else:
                        self.logger.warning(f"⚠️ up[0] has no 'upsample' attribute")
                    
                    # Apply attention if present (but skip if it causes hangs)
                    self.logger.info("🎯 CHECKING up[0] attention...")
                    if hasattr(first_up, 'attn'):
                        if first_up.attn is not None:
                            self.logger.info(f"✅ up[0] HAS attention: {type(first_up.attn)}")
                            try:
                                h = first_up.attn(h)
                                self.logger.info(f"✅ up[0].attn SUCCESS: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                            except Exception as e:
                                self.logger.warning(f"⚠️ Skipping up[0].attn due to error: {str(e)}")
                        else:
                            self.logger.info(f"ℹ️ up[0].attn is None")
                    else:
                        self.logger.info(f"ℹ️ up[0] has no 'attn' attribute")
                    
                    # Apply any other remaining components
                    self.logger.info("🔍 PROCESSING remaining up[0] components...")
                    for name, module in first_up.named_children():
                        if name not in ['block', 'upsample', 'attn'] and module != channel_reducer:
                            self.logger.info(f"🔧 Processing up[0].{name}: {type(module)}")
                            if hasattr(module, 'forward'):
                                try:
                                    h = module(h)
                                    self.logger.info(f"✅ up[0].{name} SUCCESS: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                                except Exception as e:
                                    self.logger.warning(f"⚠️ Skipping up[0].{name} due to error: {str(e)}")
                            else:
                                self.logger.warning(f"⚠️ up[0].{name} has no forward method")
                    
                    self.logger.info(f"✅ COMPLETED up[0]: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                    
                    # DEBUG: Check total number of up blocks
                    total_up_blocks = len(decoder.up)
                    self.logger.info(f"🔍 TOTAL UP BLOCKS TO PROCESS: {total_up_blocks}")
                    
                    # Continue with remaining up blocks
                    for i in range(1, total_up_blocks):
                        up_block = decoder.up[i]
                        self.logger.info(f"🔧 Processing up[{i}]...")
                        
                        # Process ResNet blocks
                        if hasattr(up_block, 'block'):
                            for j, block in enumerate(up_block.block):
                                h = block(h)
                                self.logger.info(f"  up[{i}].block[{j}]: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                        
                        # Apply upsampling (critical for spatial resolution!)
                        if hasattr(up_block, 'upsample') and up_block.upsample is not None:
                            h = up_block.upsample(h)
                            self.logger.info(f"🔧 up[{i}].upsample: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}] 🚀 SPATIAL UPSAMPLING!")
                        
                        # Apply attention if present
                        if hasattr(up_block, 'attn') and up_block.attn is not None:
                            try:
                                h = up_block.attn(h)
                                self.logger.info(f"✅ up[{i}].attn: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                            except Exception as e:
                                self.logger.warning(f"⚠️ Skipping up[{i}].attn due to error: {str(e)}")
                        
                        # Apply other components
                        for name, module in up_block.named_children():
                            if name not in ['block', 'upsample', 'attn'] and hasattr(module, 'forward'):
                                try:
                                    h = module(h)
                                    self.logger.info(f"  up[{i}].{name}: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                                except Exception as e:
                                    self.logger.warning(f"⚠️ Skipping up[{i}].{name} due to error: {str(e)}")
                        
                        self.logger.info(f"✅ Completed up[{i}]: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                else:
                    # Fallback: manually create channel reducer
                    self.logger.info("🔧 No channel reducer found, creating manual 512→256 reduction")
                    channel_reducer = torch.nn.Conv2d(512, 256, kernel_size=1, device=h.device, dtype=h.dtype)
                    with torch.no_grad():
                        # Initialize to preserve intensity
                        channel_reducer.weight.fill_(1.0 / 2.0)  # Average two 512 channels into each 256
                    
                    h = channel_reducer(h)
                    self.logger.info(f"After manual channel reduction: shape={h.shape}")
                    
                    # Now try ResNet blocks
                    self.logger.info("🔧 PROCESSING up[0] ResNet blocks...")
                    if hasattr(first_up, 'block'):
                        for j, block in enumerate(first_up.block):
                            h = block(h)
                            self.logger.info(f"  up[0].block[{j}]: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                    
                    self.logger.info("✅ FINISHED up[0] ResNet blocks, starting upsampling...")
                    
                    # CRITICAL DEBUG: Check what components up[0] actually has  
                    self.logger.info("🔍 INSPECTING up[0] components:")
                    up0_components = list(first_up.named_children())
                    for comp_name, comp_module in up0_components:
                        self.logger.info(f"  - {comp_name}: {type(comp_module)}")
                    
                    # Apply UPSAMPLING components (critical for spatial resolution)
                    self.logger.info("🚀 CHECKING up[0] upsampling...")
                    if hasattr(first_up, 'upsample'):
                        if first_up.upsample is not None:
                            self.logger.info(f"✅ up[0] HAS upsample: {type(first_up.upsample)}")
                            try:
                                h_before = h.shape
                                h = first_up.upsample(h)
                                self.logger.info(f"🚀 up[0].upsample: {h_before} -> {h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                            except Exception as e:
                                self.logger.error(f"❌ up[0].upsample FAILED: {str(e)}")
                                # Don't raise, continue processing
                        else:
                            self.logger.warning(f"⚠️ up[0].upsample is None")
                    else:
                        self.logger.warning(f"⚠️ up[0] has no 'upsample' attribute")
                    
                    self.logger.info(f"✅ COMPLETED MANUAL up[0]: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                    
                    # DEBUG: Check total number of up blocks
                    total_up_blocks = len(decoder.up)
                    self.logger.info(f"🔍 TOTAL UP BLOCKS TO PROCESS: {total_up_blocks}")
                    
                    # Continue with remaining up blocks with smart channel adaptation
                    for i in range(1, total_up_blocks):
                        up_block = decoder.up[i]
                        self.logger.info(f"🔧 Processing up[{i}]...")
                        
                        # SMART CHANNEL ADAPTATION: Check if next up block expects different channel count
                        current_channels = h.shape[1]
                        
                        # Inspect first block of this up layer to see what channels it expects
                        expected_channels = None
                        if hasattr(up_block, 'block') and len(up_block.block) > 0:
                            first_block = up_block.block[0]
                            # Look for normalization layer that indicates expected input channels
                            for name, module in first_block.named_children():
                                if 'norm' in name.lower() and hasattr(module, 'weight') and module.weight is not None:
                                    weight_size = module.weight.shape[0]
                                    expected_channels = weight_size
                                    self.logger.info(f"🔍 up[{i}] expects {expected_channels} channels (from {name})")
                                    break
                                elif hasattr(module, 'weight') and len(module.weight.shape) >= 2:
                                    expected_channels = module.weight.shape[1] 
                                    self.logger.info(f"🔍 up[{i}] expects {expected_channels} channels (from {name}: {module.weight.shape})")
                                    break
                        
                        # Apply channel adaptation if needed
                        if expected_channels is not None and expected_channels != current_channels:
                            self.logger.info(f"🔧 CHANNEL ADAPTATION NEEDED: {current_channels} -> {expected_channels}")
                            
                            if expected_channels > current_channels:
                                # Expand channels using 1x1 conv
                                channel_expander = torch.nn.Conv2d(
                                    current_channels, expected_channels, 
                                    kernel_size=1, device=h.device, dtype=h.dtype
                                )
                                with torch.no_grad():
                                    # Smart initialization - repeat and scale channels
                                    for out_ch in range(expected_channels):
                                        src_ch = out_ch % current_channels
                                        scale = 1.0 / (expected_channels // current_channels + (1 if expected_channels % current_channels > 0 else 0))
                                        channel_expander.weight[out_ch, src_ch] = scale
                                
                                h = channel_expander(h)
                                self.logger.info(f"✅ EXPANDED: {current_channels} -> {h.shape[1]} channels")
                                
                            else:
                                # Reduce channels using 1x1 conv  
                                channel_reducer = torch.nn.Conv2d(
                                    current_channels, expected_channels,
                                    kernel_size=1, device=h.device, dtype=h.dtype
                                )
                                with torch.no_grad():
                                    # Average groups of channels
                                    channels_per_output = current_channels // expected_channels
                                    for out_ch in range(expected_channels):
                                        start_ch = out_ch * channels_per_output
                                        end_ch = min(start_ch + channels_per_output, current_channels)
                                        weight_val = 1.0 / (end_ch - start_ch)
                                        for in_ch in range(start_ch, end_ch):
                                            channel_reducer.weight[out_ch, in_ch] = weight_val
                                
                                h = channel_reducer(h)
                                self.logger.info(f"✅ REDUCED: {current_channels} -> {h.shape[1]} channels")
                        else:
                            self.logger.info(f"✅ CHANNELS OK: {current_channels} matches expected")
                        
                        # Now process ResNet blocks with correct channels
                        if hasattr(up_block, 'block'):
                            for j, block in enumerate(up_block.block):
                                try:
                                    h = block(h)
                                    self.logger.info(f"  ✅ up[{i}].block[{j}]: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                                except Exception as e:
                                    self.logger.error(f"❌ up[{i}].block[{j}] FAILED: {str(e)}")
                                    self.logger.info(f"   Input shape: {h.shape}")
                                    raise e
                        
                        # Apply upsampling (critical for spatial resolution!)
                        if hasattr(up_block, 'upsample') and up_block.upsample is not None:
                            h_before = h.shape
                            h = up_block.upsample(h)
                            self.logger.info(f"🚀 up[{i}].upsample: {h_before} -> {h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}] 🚀 SPATIAL UPSAMPLING!")
                        
                        # Apply attention if present
                        if hasattr(up_block, 'attn') and up_block.attn is not None:
                            try:
                                h = up_block.attn(h)
                                self.logger.info(f"✅ up[{i}].attn: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
                            except Exception as e:
                                self.logger.warning(f"⚠️ Skipping up[{i}].attn due to error: {str(e)}")
                        
                        self.logger.info(f"✅ Completed up[{i}]: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
            
            # CRITICAL: Reduce channels to what norm_out expects (usually 128)
            current_channels = h.shape[1]
            self.logger.info(f"🔧 Before norm_out: shape={h.shape}")
            
            # Check what norm_out expects
            if hasattr(decoder, 'norm_out') and hasattr(decoder.norm_out, 'weight'):
                expected_channels = decoder.norm_out.weight.shape[0] if decoder.norm_out.weight is not None else 128
                self.logger.info(f"🔍 norm_out expects {expected_channels} channels")
                
                if current_channels != expected_channels:
                    self.logger.info(f"🔧 FINAL CHANNEL REDUCTION NEEDED: {current_channels} -> {expected_channels}")
                    
                    # Create final channel reducer
                    final_channel_reducer = torch.nn.Conv2d(
                        current_channels, expected_channels,
                        kernel_size=1, device=h.device, dtype=h.dtype
                    )
                    with torch.no_grad():
                        # Smart averaging of channels
                        channels_per_output = current_channels // expected_channels
                        for out_ch in range(expected_channels):
                            start_ch = out_ch * channels_per_output
                            end_ch = min(start_ch + channels_per_output, current_channels)
                            weight_val = 1.0 / (end_ch - start_ch)
                            for in_ch in range(start_ch, end_ch):
                                final_channel_reducer.weight[out_ch, in_ch] = weight_val
                    
                    h = final_channel_reducer(h)
                    self.logger.info(f"✅ FINAL REDUCTION: {current_channels} -> {h.shape[1]} channels")
                
                # Now apply normalization
                h = decoder.norm_out(h)
                self.logger.info(f"✅ After norm_out: shape={h.shape}, range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
            else:
                self.logger.warning("⚠️ No norm_out found or no weight - skipping normalization")
            
            h = torch.nn.functional.silu(h)
            self.logger.info(f"After SiLU: range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")
            
            # SKIP ONLY conv_out - this is what clamps to 0-1!
            self.logger.info("🎯 SKIPPING conv_out (the HDR killer) - preserving full range!")
            
            # Stats
            final_min = float(torch.min(h))
            final_max = float(torch.max(h))
            hdr_pixels = int(torch.sum(h > 1.0))
            negative_pixels = int(torch.sum(h < 0.0))
            self.logger.info(f"🧠 SMART BYPASS v2 OUTPUT: range=[{final_min:.3f}, {final_max:.3f}]")
            self.logger.info(f"   📊 HDR pixels (>1.0): {hdr_pixels}, Negative pixels: {negative_pixels}")
            
            # Convert to float32 for ComfyUI
            if h.dtype != torch.float32:
                h = h.to(dtype=torch.float32)
                self.logger.info(f"🔧 Converted output to float32")
            
            return h

    def bypass_conv_out_decode(self, vae, latent):
        """Original bypass - kept for fallback."""
        # Delegate to smart bypass for now
        return self.smart_bypass_decode(vae, latent)

    def analyze_conv_out(self, vae, latent):
        """Analyze exactly what conv_out does to our values."""
        
        self.logger.info("🔬 ANALYZING conv_out transformation...")
        
        decoder = vae.first_stage_model.decoder
        
        # Get the pre-conv_out features by running everything except the last layer
        with torch.no_grad():
            # Run the full decode to get the input to conv_out
            # We can hook into this
            pre_conv_out = None
            
            def capture_hook(module, input, output):
                nonlocal pre_conv_out
                pre_conv_out = input[0].clone()  # Capture input to conv_out
            
            # Register hook on conv_out
            hook = decoder.conv_out.register_forward_hook(capture_hook)
            
            try:
                # Run standard decode
                result = vae.decode(latent)
                
                # Analyze what conv_out did
                pre_min = float(torch.min(pre_conv_out))
                pre_max = float(torch.max(pre_conv_out))
                pre_mean = float(torch.mean(pre_conv_out))
                pre_std = float(torch.std(pre_conv_out))
                
                post_min = float(torch.min(result))
                post_max = float(torch.max(result))
                post_mean = float(torch.mean(result))
                post_std = float(torch.std(result))
                
                self.logger.info(f"📊 PRE-conv_out:  range=[{pre_min:.6f}, {pre_max:.6f}], mean={pre_mean:.6f}, std={pre_std:.6f}")
                self.logger.info(f"📊 POST-conv_out: range=[{post_min:.6f}, {post_max:.6f}], mean={post_mean:.6f}, std={post_std:.6f}")
                
                # Check if there's a learnable activation or just the convolution
                conv_only = decoder.conv_out(pre_conv_out)
                conv_min = float(torch.min(conv_only))
                conv_max = float(torch.max(conv_only))
                conv_mean = float(torch.mean(conv_only))
                
                self.logger.info(f"📊 CONV-only result: range=[{conv_min:.6f}, {conv_max:.6f}], mean={conv_mean:.6f}")
                
                # Check what transformation is applied
                ratio_min = post_min / pre_min if abs(pre_min) > 1e-6 else 0
                ratio_max = post_max / pre_max if abs(pre_max) > 1e-6 else 0
                
                self.logger.info(f"🔍 TRANSFORMATION RATIOS: min_ratio={ratio_min:.6f}, max_ratio={ratio_max:.6f}")
                
                # Analyze the transformation pattern
                if abs(post_max - 1.0) < 1e-3 and abs(post_min - 0.0) < 1e-3:
                    self.logger.info("🎯 DETECTED: conv_out appears to apply SIGMOID-like normalization to [0,1]")
                    self.NORMALIZATION_FUNCTION = "SIGMOID"
                elif abs(post_max - 1.0) < 1e-3 and abs(post_min + 1.0) < 1e-3:
                    self.logger.info("🎯 DETECTED: conv_out appears to apply TANH-like normalization to [-1,1]")
                    self.NORMALIZATION_FUNCTION = "TANH"
                else:
                    self.logger.info("🤔 DETECTED: Custom transformation pattern")
                
                # Check if conv_out has bias
                if hasattr(decoder.conv_out, 'bias') and decoder.conv_out.bias is not None:
                    bias = decoder.conv_out.bias.clone()
                    self.logger.info(f"🔧 conv_out bias: {bias.flatten()[:10].tolist()}")
                else:
                    self.logger.info("🔧 conv_out has NO bias")
                
                # Check conv_out weights range  
                if hasattr(decoder.conv_out, 'weight'):
                    weight_min = float(torch.min(decoder.conv_out.weight))
                    weight_max = float(torch.max(decoder.conv_out.weight))
                    self.logger.info(f"🔧 conv_out weight range: [{weight_min:.6f}, {weight_max:.6f}]")
                
                return {
                    'pre_conv_out': pre_conv_out,
                    'conv_only_result': conv_only,
                    'final_result': result,
                    'pre_stats': {'min': pre_min, 'max': pre_max, 'mean': pre_mean, 'std': pre_std},
                    'post_stats': {'min': post_min, 'max': post_max, 'mean': post_mean, 'std': post_std},
                    'conv_stats': {'min': conv_min, 'max': conv_max, 'mean': conv_mean}
                }
                
            except Exception as e:
                self.logger.error(f"❌ Analysis failed: {str(e)}")
                return None
            finally:
                hook.remove()

    def inverse_sigmoid(self, clamped_output):
        """Apply inverse sigmoid to recover wider range values."""
        # Avoid edge cases
        epsilon = 1e-7
        clamped = torch.clamp(clamped_output, epsilon, 1 - epsilon)
        return torch.logit(clamped)
    
    def inverse_tanh(self, clamped_output):
        """Apply inverse tanh to recover wider range values.""" 
        # Avoid edge cases
        epsilon = 1e-6
        clamped = torch.clamp(clamped_output, -1 + epsilon, 1 - epsilon)
        return torch.atanh(clamped)
    
    def smart_hdr_expansion(self, standard_output, pre_conv_out_values, expansion_factor=1.0):
        """
        Smart HDR expansion that preserves base image quality while extending highlights.
        """
        self.logger.info(f"🎯 SMART HDR EXPANSION: Using expansion factor {expansion_factor:.1f}x")

        # 🔧 CRITICAL: Ensure all tensors are on the same device
        target_device = standard_output.device
        self.logger.info(f"🔧 EXPANSION DEVICE SYNC: standard_output on {standard_output.device}, pre_conv_out on {pre_conv_out_values.device}")

        # Move pre_conv_out_values to match standard_output device if needed
        if pre_conv_out_values.device != target_device:
            pre_conv_out_values = pre_conv_out_values.to(target_device)
            self.logger.info(f"✅ MOVED pre_conv_out_values: → {target_device}")

        # Use standard output as base (perceptually correct)
        base = standard_output.clone()

        # Find highlight regions (bright areas that got clamped)
        highlight_mask = pre_conv_out_values > 1.0
        highlight_count = int(torch.sum(highlight_mask))

        self.logger.info(f"📍 HIGHLIGHT REGIONS: {highlight_count} pixels detected for expansion")

        if highlight_count > 0:
            # Apply controlled expansion only to highlights
            # Formula: base + (pre_conv_out - 1.0) * expansion_factor * base
            expansion_amount = (pre_conv_out_values - 1.0) * expansion_factor * base
            expanded = torch.where(highlight_mask, base + expansion_amount, base)

            expanded_min = float(torch.min(expanded))
            expanded_max = float(torch.max(expanded))
            hdr_pixels = int(torch.sum(expanded > 1.0))

            self.logger.info(f"✨ SMART EXPANSION: range=[{expanded_min:.3f}, {expanded_max:.3f}], HDR pixels: {hdr_pixels}")

            return expanded
        else:
            self.logger.info("⚠️ No highlights detected - returning standard output")
            return base

    def exposure_based_hdr(self, standard_output, pre_conv_out_values, max_stops=20.0):
        """
        Convert extended range to exposure stops for more natural HDR.
        """
        self.logger.info(f"📸 EXPOSURE-BASED HDR: Max {max_stops:.1f} stops extension")
        
        # 🔧 CRITICAL: Ensure all tensors are on the same device
        target_device = standard_output.device
        if pre_conv_out_values.device != target_device:
            pre_conv_out_values = pre_conv_out_values.to(target_device)
            self.logger.info(f"✅ MOVED pre_conv_out_values for exposure: → {target_device}")
        
        # Map pre-conv_out range to exposure stops
        # Anything above 1.0 in pre_conv_out becomes +EV
        exposure_map = torch.log2(torch.clamp(pre_conv_out_values, min=0.001))
        
        # Apply exposure compensation to standard output
        hdr_output = standard_output * torch.pow(2.0, exposure_map)

        # Reasonable HDR range for compositing
        final_result = torch.clamp(hdr_output, min=0, max=max_stops)

        final_min = float(torch.min(final_result))
        final_max = float(torch.max(final_result))
        hdr_pixels = int(torch.sum(final_result > 1.0))
        
        self.logger.info(f"📸 EXPOSURE HDR: range=[{final_min:.3f}, {final_max:.3f}], HDR pixels: {hdr_pixels}")
        
        return final_result

    def intelligent_hdr_decode(self, vae, latent, analysis_result, hdr_mode="mathematical_recovery", conservative_ev_multiplier=1.0):
        """
        Intelligent HDR decode with multiple modes for different use cases.

        Modes:
        - conservative: conservative_ev_multiplier, gentle expansion
        - exposure: Exposure-based natural HDR
        - mathematical recovery: Full mathematical recovery
        """
        
        self.logger.info(f"🧠 INTELLIGENT HDR DECODE: Mode = {hdr_mode.upper()}")
        
        # Get standard decode result (perceptually correct base)
        standard_result = vae.decode(latent)

        # Get pre-conv_out values for analysis
        pre_conv_out_raw = analysis_result['pre_conv_out']
        
        # 🔧 CRITICAL: Ensure both tensors are on the same device
        target_device = standard_result.device
        self.logger.info(f"🔧 DEVICE SYNC: standard_result on {standard_result.device}, pre_conv_out_raw on {pre_conv_out_raw.device}")
        
        # Move pre_conv_out_raw to match standard_result device
        if pre_conv_out_raw.device != target_device:
            pre_conv_out_raw = pre_conv_out_raw.to(target_device)
            self.logger.info(f"✅ MOVED pre_conv_out_raw: {pre_conv_out_raw.device} → {target_device}")
        
        # 🔧 CRITICAL FIX: Format pre_conv_out to match standard_result dimensions
        # pre_conv_out is raw (128ch), standard_result is RGB (3ch) - need to convert
        self.logger.info(f"🔧 TENSOR SHAPES: standard_result={standard_result.shape}, pre_conv_out_raw={pre_conv_out_raw.shape}")
        
        # Convert pre_conv_out from 128ch to 3ch RGB using same logic as main formatter
        pre_conv_out = pre_conv_out_raw
        if pre_conv_out_raw.dim() == 4 and pre_conv_out_raw.shape[1] == 128:
            # Apply the same HDR-preserving conversion
            r_channels = pre_conv_out_raw[:, 0:42, :, :]
            g_channels = pre_conv_out_raw[:, 42:84, :, :]
            b_channels = pre_conv_out_raw[:, 84:126, :, :]
            
            # Use MAX pooling to preserve HDR peaks
            r, _ = torch.max(r_channels, dim=1, keepdim=True)
            g, _ = torch.max(g_channels, dim=1, keepdim=True)
            b, _ = torch.max(b_channels, dim=1, keepdim=True)
            
            pre_conv_out_rgb = torch.cat([r, g, b], dim=1)  # [1, 3, H, W]
            
            # Convert to ComfyUI format to match standard_result
            pre_conv_out = pre_conv_out_rgb.permute(0, 2, 3, 1)  # [1, H, W, 3]
            
            self.logger.info(f"✅ CONVERTED pre_conv_out: {pre_conv_out_raw.shape} -> {pre_conv_out.shape}")
        else:
            self.logger.warning(f"⚠️ Unexpected pre_conv_out format: {pre_conv_out_raw.shape}")

        # Pre-calculated statistics:
        base_min = float(torch.min(standard_result))
        base_max = float(torch.max(standard_result))
        pre_min = float(torch.min(pre_conv_out))
        pre_max = float(torch.max(pre_conv_out))
        # pre_conv_out seems to have gradients that are not perfectly smooth!
        pre_stats = analysis_result['pre_stats']

        self.logger.info(f"📊 BASE (standard): range=[{base_min:.3f}, {base_max:.3f}]")
        self.logger.info(f"📊 PRE-CONV_OUT: range=[{pre_min:.3f}, {pre_max:.3f}]")

        # 1. Convert standard_result (sRGB) to linear light [0, 1] for RGB operations
        ldr_linear_image = self.srgb_to_linear(standard_result)
        # Define a small tolerance for floating point comparison/HDR detection
        TOL = 1e-3
        # Check if the internal, pre-activated data contained recoverable HDR information
        has_hdr_data = pre_max > (1.0 + TOL)
        # Only execute the aggressive mathematical recovery if HDR data was detected
        map_recovered = pre_conv_out
        map_recovered_aligned = 1.0
        if has_hdr_data:
            self.logger.info("✅ HDR Data Detected: Internal Max > 1.0. Enabling full mathematical recovery modes.")
            # 2. Apply adaptive inverse function to recover raw features
            if self.NORMALIZATION_FUNCTION == "TANH":
                recovered = self.inverse_tanh(standard_result)
            elif self.NORMALIZATION_FUNCTION == "SIGMOID":
                # Original VAEs use this, which requires inverse_sigmoid
                recovered = self.inverse_sigmoid(standard_result)
            else:
                # Fallback: assume standard_result is already the raw feature map
                self.logger.warning("Unknown NORMALIZATION_FUNCTION. Assuming standard_result is raw feature data.")
                recovered = standard_result

            # 3. Scale back to approximate original range
            # the math below produces smoother gradients compared to pre_conv_out!
            original_range = pre_stats['max'] - pre_stats['min']
            recovered_normalized = (recovered - torch.min(recovered)) / (torch.max(recovered) - torch.min(recovered))
            map_recovered = recovered_normalized * original_range + pre_stats['min']
            # --- CRUCIAL MIDTONE CORRECTION (Aligns mean to 1.0, establishing correct exposure) ---
            # This shifts the VAE's output so its mean corresponds to a neutral multiplier of 1.0.
            map_recovered_aligned = map_recovered - pre_stats['mean'] + 1.0

        # default result as a fallback
        final_result = ldr_linear_image
        if hdr_mode == "conservative":
            final_result = self.smart_hdr_expansion(ldr_linear_image, pre_conv_out, expansion_factor=conservative_ev_multiplier)
            self.logger.info(f"🟢 CONSERVATIVE mode: Gentle expansion of: {conservative_ev_multiplier}")

        elif hdr_mode == "exposure":
            final_result = self.exposure_based_hdr(ldr_linear_image, map_recovered, pre_stats["max"])
            self.logger.info("📸 EXPOSURE mode: Natural exposure-based HDR")

        elif hdr_mode == "adaptive_recovery":
            # Calculate the new max value after alignment for reference
            aligned_max_ev = torch.max(map_recovered_aligned)

            # Calculate compression factor for values > 1.0
            compression_factor = 1.0  # No compression needed
            if aligned_max_ev > 1.0 and aligned_max_ev > pre_stats['max']:
                # How much EV range we have above 1.0:
                excess_ev_range = aligned_max_ev - 1.0
                # How much EV range we want to compress it into:
                target_ev_range = pre_stats['max'] - 1.0

                # Compression ratio to apply to EV values > 1.0
                compression_factor = target_ev_range / excess_ev_range
                self.logger.info(
                    f"✨ Highlight Compression Factor: {compression_factor:.3f} (Max EV reduced to {pre_stats['max']:.3f})")
            else:
                self.logger.info(f"✨ No highlight compression needed. Max EV: {aligned_max_ev:.3f}")

            # Find the EV values greater than the midtone (1.0)
            highlight_mask = (map_recovered_aligned > 1.0).float()

            # Calculate the compressed highlight values
            highlight_values = map_recovered_aligned - 1.0  # Values start at 0
            compressed_highlights = highlight_values * compression_factor + 1.0

            # 2. Combine: Use the original map where mask is 0 (midtone/shadows),
            # and the compressed map where mask is 1 (highlights)
            map_compressed = (map_recovered_aligned * (1.0 - highlight_mask)) + (compressed_highlights * highlight_mask)

            # 3. Convert the compressed map to the final multiplier map
            compressed_exposure_map = torch.log2(torch.clamp(map_compressed, min=0.001))
            compressed_multiplier = torch.pow(2.0, compressed_exposure_map)
            final_result = ldr_linear_image * compressed_multiplier

        elif hdr_mode == "mathematical_recovery":
            # 4. Convert corrected map to Exposure Value (EV) in stops
            ev_target = torch.log2(torch.clamp(map_recovered_aligned, min=0.001))
            # Pre-calculate common components for all modes:
            # ev_target is the VAE's L_ratio (full dynamic range extension)
            L_ratio = torch.pow(2.0, ev_target)
            # --- END OF HDR DATA PRE-CALCULATION ---

            # Aggressive: Full mathematical recovery
            final_result = ldr_linear_image * L_ratio
            self.logger.info("🔴 Full mathematical_recovery mode")

        return final_result

    def srgb_to_linear(self, srgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts an sRGB tensor (typically from a VAE output, range [0, 1] or [-1, 1])
        to a linear light RGB tensor by applying the inverse sRGB gamma curve.

        The sRGB standard assumes values are in the range [0, 1]. If your VAE output
        is in [-1, 1], we first remap it to [0, 1] (or just work with the positive channels).

        Args:
            srgb_tensor: The aggressive tensor, assumed to be in the sRGB color space.

        Returns:
            The tensor in the linear light color space.
        """
        # 1. Handle VAE's [-1, 1] range: For color calculations, we often shift
        #    the [-1, 1] range to [0, 1] first, as sRGB is defined for positive values.
        #    Since 'aggressive' often contains negative values in shadows, we perform
        #    the sRGB conversion only on the positive components.

        # We must operate only on absolute, positive values for the sRGB curve.
        srgb_abs = torch.abs(srgb_tensor)

        # Standard sRGB linearization formula
        # For values < 0.04045 (linear segment)
        linear_part = srgb_abs / 12.92

        # For values >= 0.04045 (gamma segment)
        gamma_part = torch.pow((srgb_abs + 0.055) / 1.055, 2.4)

        # Combine the two parts based on the threshold
        linear_tensor_positive = torch.where(
            srgb_abs <= 0.04045,
            linear_part,
            gamma_part
        )

        # Apply the original sign back (since we used torch.abs earlier)
        # This maintains negative values for shadows/compression artifacts if they exist.
        linear_tensor = torch.sign(srgb_tensor) * linear_tensor_positive

        return linear_tensor

    def simple_bypass_decode(self, vae, latent):
        """Simpler bypass that skips attention to avoid hangs."""

        self.logger.info("🚀 Attempting SIMPLE bypass (skipping attention)")

        decoder = vae.first_stage_model.decoder

        with torch.inference_mode():
            # Force CUDA device detection (same as smart bypass)
            if hasattr(decoder, 'conv_in'):
                detected_device = next(decoder.conv_in.parameters()).device
                detected_dtype = next(decoder.conv_in.parameters()).dtype

                self.logger.info(f"🔍 SIMPLE BYPASS detected: {detected_device}, {detected_dtype}")

                # Force CUDA if available
                if detected_device.type == 'cpu' and torch.cuda.is_available():
                    self.logger.warning(f"⚠️ Simple bypass forcing CUDA over CPU")
                    device = torch.device('cuda:0')
                    dtype = detected_dtype
                else:
                    device = detected_device
                    dtype = detected_dtype

                original_device = latent.device
                latent = latent.to(device=device, dtype=dtype)
                self.logger.info(f"🔧 SIMPLE: {original_device} → {device}, {dtype}")

                # Ensure decoder is on the same device
                conv_device = next(decoder.conv_in.parameters()).device
                if conv_device != device:
                    self.logger.error(f"❌ SIMPLE BYPASS DEVICE MISMATCH: decoder on {conv_device}, latent on {device}")
                    try:
                        self.logger.info(f"🔧 MOVING decoder to {device} for simple bypass")
                        decoder = decoder.to(device)
                        self.logger.info(f"✅ MOVED decoder for simple bypass")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to move decoder in simple bypass: {str(e)}")
                        raise RuntimeError(f"Simple bypass device mismatch: decoder on {conv_device}, latent on {device}")
                else:
                    self.logger.info(f"✅ SIMPLE BYPASS: Both on {device}")

            # Input conv
            h = decoder.conv_in(latent)
            self.logger.info(f"After conv_in: range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")

            # MINIMAL MIDDLE BLOCKS with individual block timeouts
            self.logger.info("⚡ Doing MINIMAL middle blocks (skipping attention)")

            if hasattr(decoder, 'mid'):
                import threading
                import time

                def safe_block_execution(block, input_tensor, block_name):
                    result = [None]
                    exception = [None]

                    def block_worker():
                        try:
                            result[0] = block(input_tensor)
                        except Exception as e:
                            exception[0] = e

                    thread = threading.Thread(target=block_worker)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=10)  # 10 second timeout per block

                    if thread.is_alive():
                        raise RuntimeError(f"{block_name} timed out after 10s")
                    elif exception[0]:
                        raise exception[0]
                    else:
                        return result[0]

                try:
                    # Do block_1 with timeout (essential for channel reduction)
                    self.logger.info("🕒 Starting mid.block_1 with 10s timeout...")
                    h_cloned = h.detach().clone()  # Fix inference tensor error
                    h = safe_block_execution(decoder.mid.block_1, h_cloned, "mid.block_1")
                    self.logger.info(f"✅ mid.block_1 completed: range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")

                    # SKIP attention (decoder.mid.attn_1) - this is what causes hangs
                    self.logger.info("🚫 SKIPPING mid.attn_1 (attention) to avoid hangs")

                    # Do block_2 with timeout (complete the middle processing)
                    self.logger.info("🕒 Starting mid.block_2 with 10s timeout...")
                    h_cloned = h.detach().clone()  # Fix inference tensor error
                    h = safe_block_execution(decoder.mid.block_2, h_cloned, "mid.block_2")
                    self.logger.info(f"✅ mid.block_2 completed: range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")

                except Exception as e:
                    self.logger.error(f"❌ Error in minimal mid blocks: {str(e)}")
                    raise e

            # Now do up blocks
            if hasattr(decoder, 'up'):
                for i, up_block in enumerate(decoder.up):
                    if hasattr(up_block, 'block'):
                        for block in up_block.block:
                            h = block(h)

                    if hasattr(up_block, 'upsample') and up_block.upsample is not None:
                        h = up_block.upsample(h)

                    self.logger.info(f"After up[{i}]: range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")

            # Apply final processing BEFORE conv_out
            if hasattr(decoder, 'norm_out'):
                h = decoder.norm_out(h)
                self.logger.info(f"After norm_out: range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")

            h = torch.nn.functional.silu(h)
            self.logger.info(f"After SiLU: range=[{float(torch.min(h)):.3f}, {float(torch.max(h)):.3f}]")

            # SKIP conv_out - this preserves HDR!
            self.logger.info("🎯 SKIPPING conv_out - preserving HDR!")

            # Convert to RGB if needed
            if h.shape[1] != 3:
                self.logger.info(f"Converting {h.shape[1]} -> 3 channels")
                in_channels = h.shape[1]
                h_flat = h.view(h.shape[0], in_channels, -1).permute(0, 2, 1)
                h_rgb = torch.nn.functional.linear(h_flat, torch.eye(3, in_channels, device=h.device, dtype=h.dtype)[:, :in_channels])
                h = h_rgb.permute(0, 2, 1).view(h.shape[0], 3, h.shape[2], h.shape[3])

            # Stats
            final_min = float(torch.min(h))
            final_max = float(torch.max(h))
            hdr_pixels = int(torch.sum(h > 1.0))
            self.logger.info(f"🚀 SIMPLE BYPASS OUTPUT: range=[{final_min:.3f}, {final_max:.3f}], HDR pixels: {hdr_pixels}")

            # Convert to float32 for ComfyUI
            if h.dtype != torch.float32:
                h = h.to(dtype=torch.float32)

            return h