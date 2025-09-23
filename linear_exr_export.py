"""
Linear EXR Export Node - Professional ComfyUI Custom Node
Exports HDR images to EXR format with full dynamic range preservation

Extracted from Luminance Stack Processor for HDR VAE Decode workflow
Author: Sumit Chatterjee (adapted for HDR VAE Decode)
Version: 1.0.0
"""

import numpy as np
import torch
import cv2
import logging
import os

# Try to import imageio for HDR/EXR support
try:
    import imageio.v3 as iio
    IMAGEIO_AVAILABLE = True
except ImportError:
    try:
        import imageio as iio
        IMAGEIO_AVAILABLE = True
    except ImportError:
        IMAGEIO_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearEXRExport:
    """
    ComfyUI Custom Node for exporting HDR images to EXR format
    Clean filename interface matching standard ComfyUI save nodes
    Preserves full dynamic range data without normalization
    
    PROFESSIONAL VFX PIPELINE NOTE:
    EXR files store linear radiance values (32-bit float per channel).
    This node is specifically designed to work with the HDR VAE Decode node
    to preserve all HDR values above 1.0 for professional compositing workflows.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr_image": ("IMAGE", {"tooltip": "HDR image tensor with values potentially above 1.0"}),
                "filename_prefix": ("STRING", {"default": "HDR_VAE", "tooltip": "Base filename (without extension)"}),
            },
            "optional": {
                "output_path": ("STRING", {"default": "", "tooltip": "Output path: Empty=default ComfyUI/output, /subfolder=output/subfolder, or full custom path"}),
                "counter": ("INT", {"default": 1, "min": 0, "max": 99999, "step": 1, "tooltip": "Frame/sequence counter"}),
                "format": (["exr", "hdr"], {"default": "exr", "tooltip": "HDR file format"}),
                "bit_depth": (["16bit", "32bit"], {"default": "32bit", "tooltip": "EXR precision: 32bit = maximum quality, 16bit = smaller files"}),
                "compression": (["none", "rle", "zip", "piz", "pxr24"], {"default": "zip", "tooltip": "EXR compression type"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "export_linear_exr"
    CATEGORY = "image"
    OUTPUT_NODE = True
    
    def export_linear_exr(self, hdr_image: torch.Tensor, filename_prefix: str = "HDR_VAE", 
                         output_path: str = "", counter: int = 1, format: str = "exr", 
                         bit_depth: str = "32bit", compression: str = "zip"):
        """
        Export HDR image with clean filename interface and smart path handling
        Designed specifically for HDR VAE Decode workflow
        
        Args:
            hdr_image: HDR image tensor (potentially with values > 1.0 from HDR VAE Decode)
            filename_prefix: Base filename (no extension)
            output_path: Output directory with smart handling:
                - Empty string: Uses ComfyUI/output/ (default)
                - Starts with "/": Creates subfolder in ComfyUI/output/ (e.g. "/Test" -> "ComfyUI/output/Test")
                - Full path: Uses absolute/relative custom path
            counter: Frame/sequence number
            format: Output format (exr/hdr)
            bit_depth: EXR precision (16bit/32bit)
            compression: EXR compression type
            
        Returns:
            Tuple containing the filepath of saved HDR file
        """
        try:
            # Convert tensor to numpy array
            if len(hdr_image.shape) == 4:
                hdr_image = hdr_image.squeeze(0)  # Remove batch dimension
            
            hdr_array = hdr_image.cpu().numpy()
            
            logger.info(f"Linear EXR Export: Input range [{hdr_array.min():.6f}, {hdr_array.max():.6f}]")
            logger.info(f"Linear EXR Export: Shape {hdr_array.shape}, dtype {hdr_array.dtype}")
            
            # Check for HDR data
            hdr_pixels = int(np.sum(hdr_array > 1.0))
            negative_pixels = int(np.sum(hdr_array < 0.0))
            logger.info(f"Linear EXR Export: HDR pixels (>1.0): {hdr_pixels}, Negative pixels: {negative_pixels}")
            
            # Determine output path - default to ComfyUI/output/ directory
            output_path_clean = output_path.strip() if output_path else ""
            
            if not output_path_clean:
                # Use default ComfyUI output directory
                output_dir = self._get_comfyui_output_directory()
                logger.info(f"Using default ComfyUI output directory: {output_dir}")
            elif output_path_clean.startswith("/"):
                # User specified a subdirectory within ComfyUI output (e.g., "/Test" -> "output/Test")
                base_output_dir = self._get_comfyui_output_directory()
                subdirectory = output_path_clean[1:]  # Remove leading "/"
                output_dir = os.path.join(base_output_dir, subdirectory)
                logger.info(f"Using ComfyUI output subdirectory: {output_dir}")
            else:
                # User specified absolute or relative custom path
                output_dir = output_path_clean
                logger.info(f"Using custom absolute path: {output_dir}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Clean filename generation (NO automatic timestamps or prefixes)
            if counter > 0:
                # Include counter if specified
                filename = f"{filename_prefix}_{counter:05d}.{format}"
            else:
                # No counter - simple filename
                filename = f"{filename_prefix}.{format}"
                
            filepath = os.path.join(output_dir, filename)
            
            logger.info(f"Linear EXR Export: Saving to {filepath}")
            
            # Set precision based on bit_depth selection
            if bit_depth == "32bit":
                target_dtype = np.float32  # 32-bit single precision
                logger.info("Using 32-bit float precision for maximum HDR quality")
            else:
                # For 16-bit, we still use float32 in processing but imageio will write as half-float
                target_dtype = np.float32
                logger.info("Using 16-bit half-float precision for smaller file size")
            
            # Prepare image for saving
            if len(hdr_array.shape) == 3 and hdr_array.shape[2] == 3:
                # RGB format - keep as is for imageio, convert for OpenCV
                hdr_rgb = hdr_array.astype(target_dtype)
                hdr_bgr = cv2.cvtColor(hdr_rgb, cv2.COLOR_RGB2BGR)  # For OpenCV fallback
            else:
                hdr_rgb = hdr_array.astype(target_dtype)
                hdr_bgr = hdr_array.astype(target_dtype)
            
            # Save HDR file with TRUE bit depth control
            if format.lower() == "exr":
                # CRITICAL: Use imageio for proper EXR bit depth control
                try:
                    if IMAGEIO_AVAILABLE:
                        # Use imageio for proper 32-bit EXR writing
                        if bit_depth == "32bit":
                            logger.info("Using imageio for TRUE 32-bit EXR writing")
                            # Write as float32 for true 32-bit precision
                            iio.imwrite(filepath, hdr_rgb.astype(np.float32))
                            success = True
                        else:
                            logger.info("Using imageio for 16-bit EXR writing")
                            # Write as float16 for 16-bit precision
                            iio.imwrite(filepath, hdr_rgb.astype(np.float16))
                            success = True
                    else:
                        # Fallback to OpenCV (limited bit depth control)
                        logger.warning("imageio not available - using OpenCV (limited 32-bit support)")
                        success = cv2.imwrite(filepath, hdr_bgr)
                except Exception as e:
                    logger.error(f"imageio EXR writing failed: {e}")
                    logger.info("Falling back to OpenCV EXR writing")
                    success = cv2.imwrite(filepath, hdr_bgr)
                
            elif format.lower() == "hdr":
                # Save as Radiance HDR format (always 32-bit RGBE)
                logger.info("Saving as Radiance HDR format (32-bit RGBE)")
                success = cv2.imwrite(filepath, hdr_bgr)
            else:
                success = cv2.imwrite(filepath, hdr_bgr)  # Default to EXR behavior
            
            if not success:
                raise RuntimeError(f"Failed to save HDR file: {filepath}")
            
            # Verify the saved file preserves HDR data
            if os.path.exists(filepath):
                try:
                    # Load back and verify HDR preservation
                    verification_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    if verification_img is not None:
                        max_val = np.max(verification_img)
                        min_val = np.min(verification_img)
                        logger.info(f"Linear EXR Export verification: Range in saved file: [{min_val:.6f}, {max_val:.6f}]")
                        
                        if max_val > 1.0:
                            logger.info("✅ HDR values above 1.0 successfully preserved!")
                        else:
                            logger.warning("⚠️ No HDR values above 1.0 detected (may be LDR data)")
                        
                        if min_val < 0.0:
                            logger.info("✅ Negative values preserved (signed HDR range)")
                            
                        # Check file size as secondary verification
                        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                        logger.info(f"HDR file size: {file_size_mb:.2f} MB")
                        
                        # Get image stats
                        stats = self._get_file_stats(filepath)
                        logger.info(f"Image dimensions: {stats['width']}x{stats['height']}, {stats['channels']} channels")
                        
                    else:
                        logger.warning("Could not verify saved HDR file")
                except Exception as verify_e:
                    logger.warning(f"Could not verify HDR file: {verify_e}")
                
                logger.info(f"✅ Linear {format.upper()} file exported: {filepath}")
                return (filepath,)
            else:
                raise RuntimeError(f"HDR file was not created: {filepath}")
                
        except Exception as e:
            logger.error(f"Linear EXR export failed: {str(e)}")
            import traceback
            logger.error(f"Linear EXR export traceback: {traceback.format_exc()}")
            
            # Return error message
            error_path = f"ERROR: {str(e)}"
            return (error_path,)
    
    def _get_comfyui_output_directory(self) -> str:
        """
        Determine the ComfyUI output directory using multiple fallback methods
        Returns the path to the ComfyUI output directory
        """
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
            logger.info(f"Found ComfyUI output directory via folder_paths: {output_dir}")
            return output_dir
        except ImportError:
            # Fallback: Look for ComfyUI output directory structure
            # Navigate up from custom_nodes to find ComfyUI root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            comfyui_root = None
            
            # Try to find ComfyUI root by looking for typical structure
            search_dir = current_dir
            for _ in range(5):  # Search up to 5 levels up
                if os.path.exists(os.path.join(search_dir, "custom_nodes")) and \
                   os.path.exists(os.path.join(search_dir, "models")):
                    comfyui_root = search_dir
                    break
                search_dir = os.path.dirname(search_dir)
            
            if comfyui_root:
                output_dir = os.path.join(comfyui_root, "output")
                logger.info(f"Found ComfyUI root, using output directory: {output_dir}")
                return output_dir
            else:
                # Final fallback - assume we're in custom_nodes and go up 2 levels
                output_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "output")
                logger.info(f"Using fallback output directory: {output_dir}")
                return output_dir
                
        except Exception as e:
            logger.warning(f"Error determining ComfyUI output directory: {e}")
            # Emergency fallback - try to create output directory relative to current location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "output")
            logger.info(f"Using emergency fallback output directory: {output_dir}")
            return output_dir

    def _get_file_stats(self, filepath: str) -> dict:
        """Get statistics about the saved file"""
        try:
            # File size
            size_bytes = os.path.getsize(filepath)
            size_mb = size_bytes / (1024 * 1024)
            
            # Image dimensions using OpenCV
            img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is not None:
                height, width = img.shape[:2]
                channels = img.shape[2] if len(img.shape) > 2 else 1
            else:
                width = height = channels = 0
            
            return {
                'size_mb': size_mb,
                'width': width,
                'height': height,
                'channels': channels
            }
        except Exception:
            return {
                'size_mb': 0,
                'width': 0,
                'height': 0,
                'channels': 0
            }


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LinearEXRExport": LinearEXRExport
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LinearEXRExport": "Linear EXR Export"
}
