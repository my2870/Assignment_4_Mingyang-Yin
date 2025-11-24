"""Create dummy trained models for testing the API"""

import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from helper_lib.model1 import get_model

def create_dummy_models():
    """Create dummy model checkpoints for testing"""
    os.makedirs('models', exist_ok=True)
    
    print("Creating dummy model checkpoints...")
    
    # Create GAN models
    print("  Creating GAN models...")
    generator, discriminator = get_model('GAN', z_dim=100)
    torch.save(generator.state_dict(), 'models/gan_generator.pth')
    torch.save(discriminator.state_dict(), 'models/gan_discriminator.pth')
    print("  ✓ GAN models saved")
    
    # Create EBM model
    print("  Creating EBM model...")
    ebm = get_model('EBM', num_channels=3)
    torch.save(ebm.state_dict(), 'models/ebm_model.pth')
    print("  ✓ EBM model saved")
    
    # Create Diffusion model
    print("  Creating Diffusion model...")
    diffusion = get_model('Diffusion', image_size=32, num_channels=3, embedding_dim=32)
    torch.save(diffusion.state_dict(), 'models/diffusion_model.pth')
    print("  ✓ Diffusion model saved")
    
    print("\n✓ All dummy models created successfully!")
    print("These models are untrained and will generate random noise.")
    print("To get proper results, train the models using: python app/train.py")

if __name__ == "__main__":
    create_dummy_models()

