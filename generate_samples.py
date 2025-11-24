"""Generate sample images from trained models"""

import torch
import sys
import os
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from helper_lib.model1 import get_model
from helper_lib.generator import generate_samples, tensor_to_pil

def generate_and_save_samples(model_type, device='cpu', num_samples=16):
    """Generate and save sample images"""
    print(f"\nGenerating {num_samples} samples from {model_type} model...")
    
    # Load model
    if model_type == 'GAN':
        model, _ = get_model('GAN', z_dim=100)
        model_path = 'models/gan_generator.pth'
    elif model_type == 'EBM':
        model = get_model('EBM', num_channels=3)
        model_path = 'models/ebm_model.pth'
    elif model_type == 'Diffusion':
        model = get_model('Diffusion', image_size=32, num_channels=3, embedding_dim=32)
        model_path = 'models/diffusion_model.pth'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Generate samples
    with torch.no_grad():
        images = generate_samples(model, model_type=model_type, num_samples=num_samples, device=device)
    
    # Convert to PIL images
    pil_images = tensor_to_pil(images)
    
    # Create grid
    grid_size = int(np.sqrt(num_samples))
    grid_img = Image.new('RGB', (grid_size * 32, grid_size * 32))
    
    for idx, img in enumerate(pil_images):
        if idx >= grid_size * grid_size:
            break
        row = idx // grid_size
        col = idx % grid_size
        grid_img.paste(img, (col * 32, row * 32))
    
    # Save
    output_path = f'samples_{model_type.lower()}.png'
    grid_img.save(output_path)
    print(f"✓ Saved samples to: {output_path}")
    
    return output_path

def main():
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Generate samples from all models
    models = []
    if os.path.exists('models/gan_generator.pth'):
        models.append('GAN')
    if os.path.exists('models/ebm_model.pth'):
        models.append('EBM')
    if os.path.exists('models/diffusion_model.pth'):
        models.append('Diffusion')
    
    if not models:
        print("No trained models found. Please train models first.")
        return
    
    print(f"\nFound models: {', '.join(models)}")
    
    for model_type in models:
        try:
            generate_and_save_samples(model_type, device=device, num_samples=16)
        except Exception as e:
            print(f"✗ Error generating samples for {model_type}: {e}")
    
    print("\n✓ All samples generated!")

if __name__ == "__main__":
    main()

