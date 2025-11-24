"""Validation script to test model definitions and API structure"""

import torch
import sys
import os

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from helper_lib.model1 import get_model, Generator, Discriminator, EnergyModel, UNet, DiffusionModel
from helper_lib.generator import generate_samples, tensor_to_pil, tensor_to_base64


def test_gan_model():
    """Test GAN model instantiation"""
    print("Testing GAN model...")
    generator, discriminator = get_model('GAN', z_dim=100)
    
    # Test forward pass
    z = torch.randn(4, 100)
    images = generator(z)
    output = discriminator(images)
    
    print(f"  Generator output shape: {images.shape}")
    print(f"  Discriminator output shape: {output.shape}")
    assert images.shape == (4, 3, 32, 32), "Generator output shape incorrect"
    assert output.shape == (4, 1), "Discriminator output shape incorrect"
    print("  ✓ GAN model test passed")
    return True


def test_ebm_model():
    """Test EBM model instantiation"""
    print("\nTesting EBM model...")
    ebm = get_model('EBM', num_channels=3)
    
    # Test forward pass
    images = torch.randn(4, 3, 32, 32)
    energy = ebm(images)
    
    print(f"  EBM output shape: {energy.shape}")
    assert energy.shape == (4, 1), "EBM output shape incorrect"
    print("  ✓ EBM model test passed")
    return True


def test_diffusion_model():
    """Test Diffusion model instantiation"""
    print("\nTesting Diffusion model...")
    diffusion = get_model('Diffusion', image_size=32, num_channels=3, embedding_dim=32)
    
    # Test forward pass
    images = torch.randn(4, 3, 32, 32)
    pred_noises, actual_noises = diffusion(images)
    
    print(f"  Diffusion pred_noises shape: {pred_noises.shape}")
    print(f"  Diffusion actual_noises shape: {actual_noises.shape}")
    assert pred_noises.shape == actual_noises.shape, "Diffusion output shapes don't match"
    assert pred_noises.shape == (4, 3, 32, 32), "Diffusion output shape incorrect"
    print("  ✓ Diffusion model test passed")
    return True


def test_generation():
    """Test image generation functions"""
    print("\nTesting image generation...")
    
    # Test GAN generation
    generator, _ = get_model('GAN', z_dim=100)
    gan_images = generate_samples(generator, model_type='GAN', num_samples=2)
    print(f"  GAN generated images shape: {gan_images.shape}")
    assert gan_images.shape == (2, 3, 32, 32), "GAN generation shape incorrect"
    
    # Test conversion to PIL
    pil_images = tensor_to_pil(gan_images)
    print(f"  Converted to {len(pil_images)} PIL images")
    assert len(pil_images) == 2, "PIL conversion incorrect"
    
    # Test conversion to base64
    base64_images = tensor_to_base64(gan_images)
    print(f"  Converted to {len(base64_images)} base64 strings")
    assert len(base64_images) == 2, "Base64 conversion incorrect"
    
    print("  ✓ Generation test passed")
    return True


def test_api_imports():
    """Test API imports"""
    print("\nTesting API imports...")
    try:
        from app.main import app
        print("  ✓ FastAPI app imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error importing API: {e}")
        return False


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("Validating Generative Models Implementation")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("GAN Model", test_gan_model()))
    except Exception as e:
        print(f"  ✗ GAN test failed: {e}")
        results.append(("GAN Model", False))
    
    try:
        results.append(("EBM Model", test_ebm_model()))
    except Exception as e:
        print(f"  ✗ EBM test failed: {e}")
        results.append(("EBM Model", False))
    
    try:
        results.append(("Diffusion Model", test_diffusion_model()))
    except Exception as e:
        print(f"  ✗ Diffusion test failed: {e}")
        results.append(("Diffusion Model", False))
    
    try:
        results.append(("Image Generation", test_generation()))
    except Exception as e:
        print(f"  ✗ Generation test failed: {e}")
        results.append(("Image Generation", False))
    
    try:
        results.append(("API Imports", test_api_imports()))
    except Exception as e:
        print(f"  ✗ API import test failed: {e}")
        results.append(("API Imports", False))
    
    # Print results
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All validation tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

