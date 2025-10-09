"""
ComfyUI HDR VAE Decode Node

A custom VAE decode node that preserves full dynamic range and bit depth
capabilities for professional VFX workflows.

Author: Sumit Chatterjee
Contributor: Antonio Neto
Version: 1.1.6
License: MIT
"""

# HDR VAE Decode package with EXR export functionality
try:
    # Try relative imports first (ComfyUI package context)
    from .hdr_vae_decode import HDRVAEDecode
    from .linear_exr_export import LinearEXRExport
    from .hdr_upscale_with_model import HDRUpscaleWithModel
    print("✅ Using relative imports")
except ImportError:
    # Fall back to absolute imports (direct loading)
    try:
        from hdr_vae_decode import HDRVAEDecode
        from linear_exr_export import LinearEXRExport
        #from hdr_upscale_with_model import HDRUpscaleWithModel
        print("✅ Using absolute imports")
    except ImportError as e:
        print(f"❌ Failed to import HDR nodes: {e}")
        # Create minimal fallback
        class DummyNode:
            @classmethod
            def INPUT_TYPES(cls): return {"required": {}}
            RETURN_TYPES = ("IMAGE",)
            FUNCTION = "dummy"
            CATEGORY = "latent"
            def dummy(self): return (None,)

        HDRVAEDecode = DummyNode
        LinearEXRExport = DummyNode
        HDRUpscaleWithModel = DummyNode

# Create node mappings - HDR VAE Decode + Linear EXR Export
NODE_CLASS_MAPPINGS = {
    "HDRVAEDecode": HDRVAEDecode,
    "LinearEXRExport": LinearEXRExport,
    "HDRUpscaleWithModel": HDRUpscaleWithModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HDRVAEDecode": "HDR VAE Decode",
    "LinearEXRExport": "Linear EXR Export",
    "HDRUpscaleWithModel": "HDR Upscale with Model",
}

# Log what's available
nodes_loaded = []
if HDRVAEDecode.__name__ != "DummyNode":
    nodes_loaded.append(NODE_DISPLAY_NAME_MAPPINGS['HDRVAEDecode'])
if LinearEXRExport.__name__ != "DummyNode":
    nodes_loaded.append(NODE_DISPLAY_NAME_MAPPINGS['LinearEXRExport'])
if HDRUpscaleWithModel.__name__ != "DummyNode":
    nodes_loaded.append(NODE_DISPLAY_NAME_MAPPINGS['HDRUpscaleWithModel'])

if nodes_loaded:
    print(f"🎉 HDR Nodes loaded: {', '.join(nodes_loaded)}")
else:
    print("❌ All HDR nodes failed to load")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
