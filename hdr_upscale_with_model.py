import torch
import logging

import folder_paths
from comfy import model_management, utils
from spandrel import ModelLoader, ImageModelDescriptor
from torchvision.transforms.functional import gaussian_blur

# --- KORNIA SETUP FOR COLOR SPACE CONVERSION (Necessary for Hue Copy) ---
KORNIA_AVAILABLE = False
try:
    from kornia.color import rgb_to_ycbcr
    from kornia.filters import median_blur
    from kornia.core import Tensor
    KORNIA_AVAILABLE = True
except ImportError:
    logging.warning("Kornia not found. Hue-copy will fall back to using only the unclamped result.")

# implemented my own version of the kornia.color method that avoids returning a clamped result
def ycbcr_to_rgb(image: Tensor) -> Tensor:
    r"""Convert an YCbCr image to RGB.
    The image data is assumed to be in the range of (0, 1).
    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.
    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.
    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: Tensor = image[..., 0, :, :]
    cb: Tensor = image[..., 1, :, :]
    cr: Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: Tensor = cb - delta
    cr_shifted: Tensor = cr - delta

    r: Tensor = y + 1.403 * cr_shifted
    g: Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)

class HDRUpscaleWithModel:
    """
    A custom ComfyUI node for upscaling HDR/Wide-range images using
    a Spandrel-loaded model, incorporating HDR reversal and a Hue-Copy
    mechanism to fix large color deviations.
    """

    # --- Class Properties ---
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                            "image": ("IMAGE",),
                            "model_name": (folder_paths.get_filename_list("upscale_models"),),
                            "small_blur": ("BOOLEAN", {"default": False, "tooltip": "Apply small blur to avoid hot-pixels."}),
                            "local_fix": ("BOOLEAN", {"default": False, "tooltip": "Apply local masking to suppress extreme hotspots in dark areas."}),
                            "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], {"default": "bislerp", "tooltip": "method used by the local_fix"}),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "HDR/Upscale"

    # --- Internal Model Loading (Standard ComfyUI/Spandrel) ---
    def _load_model_internal(self, model_name):
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        loader = ModelLoader()
        upscale_model_descriptor = loader.load_from_file(model_path)
        return upscale_model_descriptor

    # --- HDR Reversal Hook ---
    def _hdr_reversal_hook(self, module, input, output):
        """
        PyTorch Forward Hook: Takes the model's clamped output and applies the
        inverse activation function (logit or atanh) to restore the full HDR range.

        It uses a conservative epsilon (1e-4) to prevent inf/NaN values near boundaries.
        """
        reversal_func = getattr(module, '_reversal_func', None)
        if reversal_func is None:
            # Fallback: Should be set during the model loading/initialization
            logging.error("HDR reversal function is missing from the module.")
            return output

        # NOTE: Using a more aggressive clamp (larger eps) helps prevent
        # extreme values (hot pixels / large hue blocks) from the logit function.
        if reversal_func.__name__ == 'logit':
            """Apply inverse sigmoid to recover wider range values."""
            # Avoid edge cases
            epsilon = 1e-7
            clamped = torch.clamp(output, epsilon, 1 - epsilon)
            return torch.logit(clamped)
        elif reversal_func.__name__ == 'atanh':
            """Apply inverse tanh to recover wider range values."""
            # Avoid edge cases
            epsilon = 1e-6
            clamped = torch.clamp(output, -1 + epsilon, 1 - epsilon)
            return torch.atanh(clamped)

        return reversal_func(output) # Fallback to the original func call

    # --- Tiled Scaling Helper Method ---
    def _run_tiled_scale(self, in_img, upscale_model, scale_amount, upscale_model_descriptor):
        """
        Encapsulates the OOM-safe, tiled upscaling logic.
        """
        device = in_img.device
        tile = 512
        overlap = 64 # Increased overlap to 64 for better boundary blending

        s = None
        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = utils.ProgressBar(steps)

                s = utils.tiled_scale(
                    in_img,
                    lambda a: upscale_model(a),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=scale_amount,
                    pbar=pbar
                )
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                logging.warning(f"OOM detected. Reducing tile size to {tile}")
                if tile < 128:
                    # Cleanup hook if OOM causes function exit
                    if hasattr(upscale_model_descriptor, 'hook_handle'):
                        upscale_model_descriptor.hook_handle.remove()
                    upscale_model.to("cpu")
                    raise e
        return s

    def upscale(self, image, model_name, small_blur, local_fix, upscale_method):
        upscale_model_descriptor = self._load_model_internal(model_name)
        upscale_model = upscale_model_descriptor.model
        device = model_management.get_torch_device()
        scale = upscale_model_descriptor.scale

        # 1. Setup and Memory Management
        memory_required = model_management.module_size(upscale_model)
        # Conservative memory estimate based on scale factor
        memory_required += (512 * 512 * 3) * image.element_size() * max(scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        # Move model to device
        upscale_model.to(device)
        in_img_original = image.movedim(-1,-3).to(device) # NCHW on GPU

        # Determine the reversal function (e.g., torch.logit for sigmoid)
        reversal_func = self._get_reversal_func(upscale_model_descriptor)
        setattr(upscale_model, '_reversal_func', reversal_func)

        # Register the hook BEFORE the first tiled call
        hook_handle = upscale_model.register_forward_hook(self._hdr_reversal_hook)
        setattr(upscale_model_descriptor, 'hook_handle', hook_handle)

        # Apply a light input filter to pre-mitigate single-pixel noise
        in_img_filtered = in_img_original
        if small_blur:
            logging.info("Applying Gaussian blur to input.")
            in_img_filtered = gaussian_blur(in_img_original, kernel_size=3, sigma=0.1)

        # --- 2. First Pass: Full HDR Upscale ---
        s_unclamped = self._run_tiled_scale(in_img_filtered, upscale_model, scale, upscale_model_descriptor)

        # --- 3. Second Pass: Stable Color Upscale ---
        COLOR_STABLE_MIN = -1.0
        COLOR_STABLE_MAX = 1.0
        in_img_clamped = torch.clamp(in_img_filtered, COLOR_STABLE_MIN, COLOR_STABLE_MAX)
        s_clamped = self._run_tiled_scale(in_img_clamped, upscale_model, scale, upscale_model_descriptor)

        # --- 4. HSV Recombination (Hue-Copy) ---
        if KORNIA_AVAILABLE and s_unclamped is not None and s_clamped is not None:
            logging.info("Performing Kornia YCBCR Luma-Chrominance Copy.")

            # Convert to YCBCR (NCHW, C=3)
            ycbcr_clamped = rgb_to_ycbcr(s_clamped)  # Stable CB/CR
            ycbcr_unclamped = rgb_to_ycbcr(s_unclamped)  # Full Y
            ycbcr_clamped = ycbcr_clamped.to(device)
            ycbcr_unclamped = ycbcr_unclamped.to(device)

            # Extract the full-dynamic Luma (Y) from the unclamped result
            y_detail = ycbcr_unclamped[:, 0:1, :, :]

            # Clamp the upscaled Y-channel by the scaled original Y-channel
            y_stabilized = torch.clamp(y_detail, min=0.0, max=8.0)
            # 2. Apply a light median blur to kill single-pixel hot-spots
            try:
                # Use a 3x3 kernel size for light filtering
                y_stabilized = median_blur(y_stabilized, kernel_size=(3, 3))
            except Exception as e:
                logging.error(f"Median Blur on Y failed: {e}")

            # Extract the stable Chrominance (CB and CR) from the clamped result
            cb_stable = ycbcr_clamped[:, 1:2, :, :]
            cr_stable = ycbcr_clamped[:, 2:3, :, :]

            # Concatenate the new channels: Stabilized Y + Stable CB + Stable CR
            composite_ycbcr = torch.cat([y_stabilized, cb_stable, cr_stable], dim=1)

            # Convert back to RGB for the final output
            s_final = ycbcr_to_rgb(composite_ycbcr)

            # 's' is NCHW (Batch, Channel, Height, Width) on the GPU
            if small_blur:
                try:
                    s_final = median_blur(s_final, kernel_size=(3, 3))
                except Exception as e:
                    logging.error(f"Median Blur failed on output: {e}")
        else:
            logging.warning("Skipping Hue-Copy due to missing Kornia or processing error. Returning unclamped result.")
            s_final = s_unclamped

        # --- 5. Optional Local Hotspot Fix (Option 2) ---
        if local_fix:
            logging.info("Applying local hotspot suppression.")

            # A. Prepare the Mask
            # Get the luminance (Y) from the original input (filtered)
            y_original = rgb_to_ycbcr(in_img_filtered)[:, 0:1, :, :]

            # Scale the original Y channel up to match the upscaled size
            target_size = (s_final.shape[3], s_final.shape[2])
            y_original_scaled = utils.common_upscale(y_original, target_size[0], target_size[1], upscale_method, False)
            y_original_scaled = y_original_scaled.to(device)

            # Create a mask where the original Y value is below a low threshold (e.g., 0.1)
            # These are the dark areas prone to artificial spikes.
            THRESHOLD = 0.1
            mask = (y_original_scaled < THRESHOLD).float()

            # B. Prepare the Correction Image
            # Create a highly conservative version of the final image for blending.
            # This is a heavily clipped, desaturated version of s_final.
            s_conservative = torch.clamp(s_final, -1.0, 1.0)  # Clip to the stable range

            # C. Blend
            # Blend the aggressive (conservative) fix only in the masked dark areas.
            # Blending formula: s_final * (1 - mask) + s_conservative * mask
            s_final = s_final * (1.0 - mask) + s_conservative * mask

        # --- 6. Cleanup and Return (omitted for brevity) ---
        if hasattr(upscale_model_descriptor, 'hook_handle'):
            upscale_model_descriptor.hook_handle.remove()
        upscale_model.to("cpu")
        s = s_final.movedim(-3, -1)
        return (s,)

    # --- Helper to determine the reversal function ---
    def _get_reversal_func(self, upscale_model_descriptor):
        architecture_name = upscale_model_descriptor.architecture.name
        # if the model output is sigmoid or tanh
        # This function should return 'torch.logit' or 'torch.atanh' (or similar).
        if architecture_name in ["ESRGAN", "RealESRGAN", "SwinIR", "HAT"]:
            # Most traditional image upscalers are expected to be in a [0, 1] range
            # which you map to Sigmoid/Logit reversal for HDR.
            return torch.atanh
        elif "VAE" in architecture_name:
            # If a model is known to output a [-1, 1] range, you'd use atanh.
            return torch.atanh
        else:
            # Default to logit for safety, or raise an error if unknown
            return torch.logit
