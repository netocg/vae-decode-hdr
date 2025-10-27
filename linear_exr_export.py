"""
Linear HDR Export Node - Professional ComfyUI Custom Node
Exports HDR images to EXR/HDR format with full dynamic range preservation

Extracted from Luminance Stack Processor for HDR VAE Decode workflow
Author: Sumit Chatterjee (adapted for HDR VAE Decode)
Contributor: Antonio Neto (adapted for HDR VAE Decode)
Version: 1.1.4 (Improved verification by prioritizing pyexr for reading)
"""

import numpy as np
import json
import torch
import cv2
import logging
import os
import re
from glob import glob
import traceback # Ensure traceback is available for error logging

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

# Try to import pyexr for dedicated EXR support (Recommended)
try:
    import pyexr
    PYEXR_AVAILABLE = True
except ImportError:
    PYEXR_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


def get_highest_numbered_file(directory, prefix):
    """
    Finds the highest existing version number (e.g., '012' in 'prefix_v012_...')
    to determine the next version number.

    Args:
        directory (str): The directory to search.
        prefix (str): The base filename prefix (e.g., 'HDR_VAE').

    Returns:
        int: The highest number found, or 0.
    """
    pattern = os.path.join(directory, f"{prefix}*")
    files = glob(pattern)
    max_num = 0

    # Robust regex pattern: looks for the prefix, followed by '_v',
    # and then captures one or more digits (\d+), ignoring any suffix.
    # This handles padded versions like _v001, _v010, and non-padded versions.
    regex = re.compile(r'^' + re.escape(prefix) + r'_v(\d+).*$')

    if files:
        for file_path in files:
            filename = os.path.basename(file_path)
            match = regex.match(filename)

            if match:
                # The version number (including padding) is captured in group 1
                num_str = match.group(1)
                # Convert to integer, which naturally ignores leading zeros (e.g., '001' -> 1)
                num = int(num_str)

                if num > max_num:
                    max_num = num

    return max_num


class LinearEXRExport:
    """
    ComfyUI Custom Node for exporting HDR images to EXR format
    Clean filename interface matching standard ComfyUI save nodes
    Preserves full dynamic range data without normalization
    """
    # Dictionary to track the next version number for each filename_prefix
    VERSION_TRACKER = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr_image": ("IMAGE", {"tooltip": "HDR image tensor with values potentially above 1.0"}),
                "filename_prefix": ("STRING", {"default": "comfyUI", "tooltip": "Base filename (without extension)"}),
            },
            "optional": {
                "versioning": ("BOOLEAN", {"default": False, "tooltip": "Incremental versioning save. adding v001, v002... to it's file name"}),
                "frame_sequence": ("BOOLEAN", {"default": False, "tooltip": "Save animation into multiple frames 1001, 1002..."}),
                "start_frame": ("INT", {"default": 1001, "min": 0, "max": 99999999}),
                "frame_pad": ("INT", {"default": 4, "min": 1, "max": 8}),
                "output_path": ("STRING", {"default": "/HDR", "tooltip": "Output path: Empty=default ComfyUI/output, /subfolder=output/subfolder, or full custom path"}),
                "format": (["exr", "hdr"], {"default": "exr", "tooltip": "file format"}),
                "bit_depth": (["16bit", "32bit"], {"default": "16bit", "tooltip": "EXR precision: 32bit = maximum quality, 16bit = smaller files"}),
                "compression": (["none", "rle", "zip", "piz", "pxr24"], {"default": "zip", "tooltip": "EXR compression type"}),
                "save_workflow": ("BOOLEAN", {"default": False, "tooltip": "Saves the workflow JSON to a sidecar file next to the HDR image"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO", # ADDED: Necessary for UI workflow JSON
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "export_linear_exr"
    CATEGORY = "image"
    OUTPUT_NODE = True

    def _write_sidecar_workflow(self, filepath: str, api_json: dict, ui_json: dict):
        """Writes the workflow JSON data to a sidecar file next to the EXR/HDR file."""

        # Change the extension (e.g., .exr) to .json
        base_name, _ = os.path.splitext(filepath)
        json_filepath = base_name + ".json"

        workflow_dict = {
            # 'prompt' is the API JSON
            "prompt": api_json or {},
            # 'extra_pnginfo' is the UI JSON
            "extra_pnginfo": ui_json or {}
        }

        try:
            # Only save if there is content in the prompt or UI info
            if workflow_dict["prompt"] or workflow_dict["extra_pnginfo"]:
                with open(json_filepath, 'w') as f:
                    json.dump(workflow_dict, f, indent=4)
                logger.info(f"✅ Workflow metadata saved to sidecar file: {json_filepath}")
            else:
                 logger.warning(f"Workflow save skipped: No prompt or UI info found.")
        except Exception as e:
            logger.error(f"Failed to write sidecar workflow file: {e}")

    def _save_file(self, filepath: str, hdr_rgb: np.ndarray, hdr_bgr: np.ndarray, format: str, bit_depth: str,
                   compression: str, logger) -> bool:
        """
        Handles saving a single HDR image (frame) to EXR or HDR format.
        Now prioritizes pyexr, then imageio, then cv2.
        """
        success = False

        if format.lower() == "exr":
            # Determine the target numpy dtype based on bit depth
            target_dtype = np.float32 if bit_depth == "32bit" else np.float16

            # --- 1. Attempt using pyexr (Most Robust) ---
            if PYEXR_AVAILABLE:
                try:
                    logger.info(f"Attempting pyexr EXR write with compression='{compression}' and dtype='{target_dtype}'.")

                    # pyexr expects the data to be in the (H, W, C) format, which hdr_rgb is.
                    pyexr.write(
                        filepath,
                        hdr_rgb.astype(target_dtype),
                        compression=compression
                    )
                    success = True
                    return success
                except Exception as e:
                    logger.error(f"pyexr EXR write failed: {e}. Falling back.")

            # --- 2. Attempt using imageio (Failing in user's environment) ---
            if IMAGEIO_AVAILABLE:
                try:
                    logger.info(f"Attempting imageio EXR write with compression='{compression}' and dtype='{target_dtype}'.")

                    # Try writing with all requested arguments
                    iio.imwrite(
                        filepath,
                        hdr_rgb.astype(target_dtype),
                        compression=compression
                    )
                    success = True
                    return success

                except (TypeError, ValueError) as e:
                    # Catch the specific error about incompatible arguments (like PyAVPlugin)
                    logger.warning(f"imageio EXR write failed (Error: {e}). Retrying without 'compression' argument.")
                    try:
                        # Attempt to save without the compression argument
                        iio.imwrite(filepath, hdr_rgb.astype(target_dtype))
                        success = True
                        return success
                    except Exception as e2:
                        logger.error(f"imageio retry without compression failed: {e2}. Falling back.")

                except Exception as e:
                    logger.error(f"imageio EXR write failed (General Error): {e}. Falling back.")

            # --- 3. Fallback to OpenCV (cv2) ---
            logger.warning("Falling back to OpenCV (cv2) for EXR save.")
            try:
                # OpenCV handles saving 3-channel EXR (requires BGR for cv2)
                success = cv2.imwrite(filepath, hdr_bgr)

                # Check explicitly if cv2.imwrite failed (returns False)
                if not success:
                    logger.error("OpenCV cv2.imwrite returned False. This commonly indicates that the required OpenEXR codec is missing or incompatible in your OpenCV build.")

            except Exception as e_cv:
                logger.error(f"OpenCV EXR writing failed (Exception raised): {e_cv}")

        elif format.lower() == "hdr":
            # --- 1. Radiance HDR (Always cv2) ---
            logger.info("Saving as Radiance HDR format (32-bit RGBE) via OpenCV.")
            success = cv2.imwrite(filepath, hdr_bgr)

        else:
            logger.error(f"Unsupported format: {format}")

        if success:
            logger.info(f"Successfully saved file: {filepath}")
        else:
            logger.error(f"Final save failed for file: {filepath}")

        return success


    def export_linear_exr(self, hdr_image: torch.Tensor, filename_prefix: str = "HDR_VAE",
                          output_path: str = "", start_frame: int = 1, frame_pad: int = 4, versioning: bool = True,
                          frame_sequence: bool = False, format: str = "hdr",  bit_depth: str = "16bit",
                          compression: str = "zip", save_workflow: bool = False,
                          prompt: dict = None, extra_pnginfo: dict = None):
        """
        Export HDR image with clean filename interface and smart path handling
        Designed specifically for HDR VAE Decode workflow
        """
        try:
            # Ensure the input is always 4D (B, H, W, C). This must happen first.
            if len(hdr_image.shape) == 3:
                hdr_image = hdr_image.unsqueeze(0)

            # Now the tensor is guaranteed to be 4D
            batch_size = hdr_image.shape[0]

            # Convert tensor to numpy array (B, H, W, C) for initial logging
            hdr_array_initial = hdr_image.cpu().numpy()

            logger.info(f"Linear EXR Export: Input range [{hdr_array_initial.min():.6f}, {hdr_array_initial.max():.6f}]")
            logger.info(f"Linear EXR Export: Shape {hdr_array_initial.shape}, dtype {hdr_array_initial.dtype}")

            if not PYEXR_AVAILABLE:
                logger.warning("pyexr not found. Install with 'pip install pyexr' for the most reliable EXR export.")

            # Check for HDR data
            hdr_pixels = int(np.sum(hdr_array_initial > 1.0))
            negative_pixels = int(np.sum(hdr_array_initial < 0.0))
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

            # 1. Cleanly split the prefix into a relative path and the actual filename base
            prefix_parts = filename_prefix.replace("/", os.sep).replace("\\", os.sep).split(os.sep)
            if len(prefix_parts) > 1:
                prefix_sub_dir = os.path.join(*prefix_parts[:-1])
                base_filename_prefix = prefix_parts[-1]
                output_dir = os.path.join(output_dir, prefix_sub_dir)
            else:
                base_filename_prefix = filename_prefix

            os.makedirs(output_dir, exist_ok=True)

            # Calculate Version (once)
            filename_parts = [base_filename_prefix]
            if versioning:
                max_fs_num = get_highest_numbered_file(os.path.normpath(output_dir), base_filename_prefix)
                current_version = max_fs_num + 1
                filename_parts.append(f"_v{current_version:03d}")

            # Add Frame Placeholder
            if batch_size > 1 or frame_sequence:
                filename_parts.append(f"_frame_%0{frame_pad}d")

            # Final filename template (e.g., "HDR_VAE_v001_frame_%04d.exr")
            base_filename = "".join(filename_parts) + f".{format}"

            # --- 3. DATA PREPARATION ---
            target_dtype = np.float32
            # The array you will iterate over for the RGB data
            hdr_array_rgb = hdr_image.cpu().numpy().astype(target_dtype).copy()

            # --- 4. BATCH LOOP & SAVE ---
            final_filepaths = []

            # Progress Bar Setup
            pbar = None
            if batch_size > 1:
                try:
                    from comfy.utils import ProgressBar
                    pbar = ProgressBar(batch_size)
                except ImportError:
                    pass

            for i in range(batch_size):
                # Get the current frame in RGB
                current_hdr_rgb = hdr_array_rgb[i]

                # --- NEW STEP: Convert the single frame (H, W, 3) to BGR for cv2 ---
                if current_hdr_rgb.shape[-1] == 3:
                    current_hdr_bgr = cv2.cvtColor(current_hdr_rgb, cv2.COLOR_RGB2BGR)
                else:
                    current_hdr_bgr = current_hdr_rgb

                # Calculate the frame number and final filename
                frame_number = start_frame + i
                filename = base_filename % frame_number if (batch_size > 1 or frame_sequence) else base_filename
                filepath = os.path.join(output_dir, filename)

                # Call the helper method for saving the current frame
                success = self._save_file(
                    filepath,
                    current_hdr_rgb,  # Current frame RGB (for pyexr/imageio)
                    current_hdr_bgr,  # Current frame BGR (for cv2)
                    format, bit_depth, compression, logger
                )

                if not success:
                    # Reraise the exception for ComfyUI to catch
                    raise RuntimeError(f"Failed to save {format} file: {filepath}")

                # --- WORKFLOW SAVE (Only for the first frame) ---
                if i == 0 and save_workflow:
                    self._write_sidecar_workflow(filepath, prompt, extra_pnginfo)

                final_filepaths.append(filepath)
                if pbar: pbar.update(1)

            # --- 5. VERIFICATION & RETURN (Modified for batch) ---

            # We will only verify the *last* saved file
            if final_filepaths:
                last_filepath = final_filepaths[-1]
                self._verify_save(last_filepath, logger)
                logger.info(f"✅ Linear {format.upper()} exported: {batch_size} frames.")
                return (last_filepath,)

            raise RuntimeError("Export completed, but no file paths were recorded.")

        except Exception as e:
            logger.error(f"Linear EXR export failed: {str(e)}")
            logger.error(f"Linear EXR export traceback: {traceback.format_exc()}")
            return (f"ERROR: {str(e)}",)

    def _verify_save(self, filepath: str, logger):
        """
        Verifies that the saved file exists, preserves HDR data (values > 1.0),
        and reports image dimensions and file size.
        This function now prioritizes pyexr for reading/verification.
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Could not verify save: File not found at {filepath}")
                return

            verification_img = None

            # --- Attempt 1: Verify using pyexr (Most Robust) ---
            if PYEXR_AVAILABLE:
                try:
                    verification_img = pyexr.read(filepath)
                    logger.info("Verification succeeded using pyexr.")
                except Exception as e:
                    logger.warning(f"pyexr verification failed ({e}). Falling back to cv2.")

            # --- Attempt 2: Fallback to OpenCV (cv2) ---
            if verification_img is None:
                try:
                    # Load back the saved file preserving all color and depth information
                    verification_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    logger.info("Verification succeeded using cv2.")
                except Exception as e:
                    # This catches the OpenEXR codec disabled error
                    logger.warning(f"cv2 verification failed ({e}). Skipping data check.")


            if verification_img is not None:
                # Calculate min/max values
                max_val = np.max(verification_img)
                min_val = np.min(verification_img)

                logger.info(f"Linear EXR Export verification: Range in saved file: [{min_val:.6f}, {max_val:.6f}]")

                # Check for HDR preservation
                if max_val > 1.0:
                    logger.info("✅ HDR values above 1.0 successfully preserved!")
                else:
                    logger.warning("⚠️ No HDR values above 1.0 detected (may be LDR data).")

                # Check for signed range
                if min_val < 0.0:
                    logger.info("✅ Negative values preserved (signed HDR range).")

                # Get and log file statistics
                stats = self._get_file_stats(filepath)
                logger.info(f"Image dimensions: {stats['width']}x{stats['height']}, {stats['channels']} channels")
                logger.info(f"HDR file size: {stats['size_mb']:.2f} MB")

            else:
                logger.warning("Could not verify saved HDR file: Failed to read file with any available method.")

        except Exception as verify_e:
            logger.warning(f"Error during HDR file verification: {verify_e}")

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
            
            # Since cv2 can't read the EXR, we rely on pyexr's successful read for dimensions if available
            width = height = channels = 0
            
            if PYEXR_AVAILABLE:
                try:
                    # Attempt to get dimensions using pyexr just for logging if cv2 failed
                    exr_file = pyexr.open(filepath)
                    width = exr_file.width
                    height = exr_file.height
                    # This is a simplification; channel count is complex in EXR but 3 is typical
                    channels = 3
                except Exception:
                    pass # Keep dimensions at 0 if pyexr failed to open the file object

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
