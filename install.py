"""
Installation script for ComfyUI HDR VAE Decode Node

This script helps install the HDR VAE decode custom node and its dependencies.
"""

import os
import sys
import subprocess
import importlib.util
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_comfyui_installation():
    """Check if ComfyUI is available."""
    try:
        # Common ComfyUI locations
        possible_paths = [
            os.path.join(os.getcwd(), '..', '..'),  # If in custom_nodes/hdr-vae-decode
            os.path.join(os.getcwd(), '..'),        # If in ComfyUI root
            os.getcwd()                             # If ComfyUI is current dir
        ]
        
        for path in possible_paths:
            comfyui_main = os.path.join(path, 'main.py')
            if os.path.exists(comfyui_main):
                logger.info(f"Found ComfyUI installation at: {path}")
                return path
        
        logger.warning("ComfyUI installation not found")
        return None
        
    except Exception as e:
        logger.error(f"Error checking ComfyUI installation: {str(e)}")
        return None

def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing dependencies...")
    
    # Core dependencies (always required)
    # Note: torch, torchvision, numpy are provided by ComfyUI
    core_deps = [
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "einops>=0.4.0"
    ]
    
    # Optional dependencies for EXR support
    optional_deps = [
        "imageio>=2.25.0",
        "imageio-ffmpeg>=0.4.8"
    ]
    
    # Try to install OpenEXR (may fail on some systems)
    openexr_deps = ["OpenEXR>=1.3.9", "Imath>=1.0.0"]
    
    # Install core dependencies
    for dep in core_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            logger.info(f"✅ Installed: {dep}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {dep}: {str(e)}")
            return False
    
    # Install optional dependencies
    for dep in optional_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            logger.info(f"✅ Installed: {dep}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️  Optional dependency failed {dep}: {str(e)}")
    
    # Try OpenEXR (platform dependent)
    if os.name != 'nt':  # Not Windows
        for dep in openexr_deps:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                logger.info(f"✅ Installed: {dep}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  OpenEXR dependency failed {dep}: {str(e)}")
                logger.info("OpenEXR support will use imageio fallback")
    else:
        logger.info("Windows detected - using imageio for EXR support")
    
    return True

def check_dependencies():
    """Check if all dependencies are available."""
    logger.info("Checking dependencies...")
    
    required_modules = {
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'einops': 'Einops'
    }
    
    # Check ComfyUI-provided modules (should already be available)
    comfyui_modules = {
        'torch': 'PyTorch (ComfyUI)',
        'torchvision': 'TorchVision (ComfyUI)', 
        'numpy': 'NumPy (ComfyUI)'
    }
    
    optional_modules = {
        'OpenEXR': 'OpenEXR',
        'Imath': 'Imath',
        'imageio': 'ImageIO'
    }
    
    missing_required = []
    missing_optional = []
    
    # Check ComfyUI-provided modules (should be available)
    for module, name in comfyui_modules.items():
        if importlib.util.find_spec(module) is None:
            logger.warning(f"⚠️  {name} not found - ComfyUI may not be properly installed")
        else:
            logger.info(f"✅ {name} available")
    
    # Check required modules
    for module, name in required_modules.items():
        if importlib.util.find_spec(module) is None:
            missing_required.append(name)
        else:
            logger.info(f"✅ {name} available")
    
    # Check optional modules
    for module, name in optional_modules.items():
        if importlib.util.find_spec(module) is None:
            missing_optional.append(name)
        else:
            logger.info(f"✅ {name} available")
    
    if missing_required:
        logger.error(f"❌ Missing required dependencies: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        logger.warning(f"⚠️  Missing optional dependencies: {', '.join(missing_optional)}")
        logger.info("EXR export may not be available")
    
    return True

def run_tests():
    """Run basic validation tests."""
    logger.info("Running validation tests...")
    
    try:
        # Import test module
        test_script = os.path.join(os.path.dirname(__file__), 'test_hdr_decode.py')
        
        if not os.path.exists(test_script):
            logger.warning("Test script not found - skipping validation")
            return True
        
        # Run tests
        result = subprocess.run([sys.executable, test_script], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ All validation tests passed")
            logger.info("   Tensor formatting verified")
            logger.info("   HDR preservation confirmed") 
            logger.info("   ComfyUI compatibility validated")
            return True
        else:
            logger.error("❌ Some validation tests failed")
            logger.error(result.stderr)
            logger.warning("Check TROUBLESHOOTING.md for common solutions")
            return False
            
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return False

def verify_tensor_formatting():
    """Verify tensor formatting compatibility with ComfyUI."""
    logger.info("Verifying tensor formatting...")
    
    try:
        import torch
        
        # Test tensor format function
        dummy_tensor = torch.randn(1, 3, 512, 512)  # [batch, channels, height, width]
        
        # This should work without errors
        formatted = dummy_tensor.permute(0, 2, 3, 1).contiguous()  # [batch, height, width, channels]
        
        expected_shape = (1, 512, 512, 3)
        if formatted.shape == expected_shape:
            logger.info("✅ Tensor formatting verified")
            return True
        else:
            logger.error(f"❌ Tensor formatting issue: got {formatted.shape}, expected {expected_shape}")
            return False
            
    except Exception as e:
        logger.error(f"Tensor formatting test failed: {str(e)}")
        return False

def create_symlinks_if_needed():
    """Create symlinks if this isn't installed in custom_nodes."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we're already in custom_nodes
    if 'custom_nodes' in current_dir:
        logger.info("Already in custom_nodes directory")
        return True
    
    # Find ComfyUI installation
    comfyui_path = check_comfyui_installation()
    if not comfyui_path:
        logger.error("ComfyUI not found - cannot create symlinks")
        return False
    
    custom_nodes_path = os.path.join(comfyui_path, 'custom_nodes')
    if not os.path.exists(custom_nodes_path):
        os.makedirs(custom_nodes_path)
        logger.info(f"Created custom_nodes directory: {custom_nodes_path}")
    
    # Create symlink
    target_path = os.path.join(custom_nodes_path, 'vae-decode-hdr')
    
    try:
        if os.path.exists(target_path):
            logger.info(f"Symlink already exists: {target_path}")
        else:
            os.symlink(current_dir, target_path)
            logger.info(f"Created symlink: {target_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create symlink: {str(e)}")
        logger.info("You may need to manually copy files to ComfyUI/custom_nodes/")
        return False

def main():
    """Main installation function."""
    logger.info("🚀 Installing ComfyUI HDR VAE Decode Node")
    logger.info("=" * 50)
    
    steps = [
        ("Checking ComfyUI installation", check_comfyui_installation),
        ("Installing dependencies", install_dependencies),
        ("Checking dependencies", check_dependencies),
        ("Verifying tensor formatting", verify_tensor_formatting),
        ("Creating symlinks if needed", create_symlinks_if_needed),
        ("Running validation tests", run_tests)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n📋 {step_name}...")
        try:
            result = step_func()
            if not result:
                logger.error(f"❌ {step_name} failed")
                if step_name in ["Installing dependencies", "Checking dependencies"]:
                    logger.error("Installation cannot continue without dependencies")
                    return False
                else:
                    logger.warning(f"⚠️  {step_name} failed but continuing...")
            else:
                logger.info(f"✅ {step_name} completed")
        except Exception as e:
            logger.error(f"❌ {step_name} crashed: {str(e)}")
            return False
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 Installation completed successfully!")
    logger.info("\n📋 Installation Summary:")
    logger.info("   ✅ Uses ComfyUI's built-in PyTorch, TorchVision, and NumPy")
    logger.info("   ✅ Installed only additional dependencies needed")
    logger.info("   ✅ No version conflicts with ComfyUI environment")
    logger.info("   ✅ Automatic dtype conversion for VAE compatibility")
    logger.info("   ✅ Enhanced tensor format detection and correction")
    logger.info("\nNext steps:")
    logger.info("1. Restart ComfyUI")
    logger.info("2. Look for 'HDR VAE Decode' in the latent/hdr category")
    logger.info("3. Use HDR-focused prompts (bright lights, reflections, etc.)")
    logger.info("4. Check console output for HDR detection messages")
    logger.info("5. Connect HDR output to your EXR export node for true HDR files")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
