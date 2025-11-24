"""Image generation functions for trained models"""

import torch
import numpy as np
from PIL import Image
import io
import base64


def generate_samples(model, model_type='GAN', num_samples=10, device='cpu'):
    """
    Generate image samples from trained model
    
    Args:
        model: Trained model (Generator for GAN, EnergyModel for EBM, DiffusionModel for Diffusion)
        model_type: Type of model ('GAN', 'EBM', 'Diffusion')
        num_samples: Number of samples to generate
        device: Device to generate on
        
    Returns:
        Generated images as torch tensor (N, C, H, W) in range [-1, 1]
    """
    model.eval()
    
    with torch.no_grad():
        if model_type == 'GAN':
            return generate_gan_samples(model, num_samples, device)
        elif model_type == 'EBM':
            return generate_ebm_samples(model, num_samples, device)
        elif model_type == 'Diffusion':
            return generate_diffusion_samples(model, num_samples, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def generate_gan_samples(generator, num_samples, device):
    """Generate samples from GAN generator"""
    z = torch.randn(num_samples, generator.z_dim, device=device)
    images = generator(z)
    return images


def generate_ebm_samples(energy_model, num_samples, device, steps=100, step_size=10, noise=0.005):
    """Generate samples from Energy-Based Model using Langevin dynamics"""
    # Initialize with random noise
    samples = torch.rand(num_samples, 3, 32, 32, device=device) * 2 - 1
    
    # Set model to training mode temporarily for gradient computation
    was_training = energy_model.training
    energy_model.train()
    
    # Run Langevin dynamics
    for _ in range(steps):
        samples = samples.detach().requires_grad_(True)
        energy = energy_model(samples).sum()
        grad = torch.autograd.grad(energy, samples, create_graph=False)[0]
        with torch.no_grad():
            samples = samples - step_size * grad + noise * torch.randn_like(samples)
            samples = torch.clamp(samples, -1, 1)
    
    # Restore original mode
    if not was_training:
        energy_model.eval()
    
    return samples.detach()


def generate_diffusion_samples(diffusion_model, num_samples, device, 
                               num_steps=20, min_signal_rate=0.02, max_signal_rate=0.95):
    """Generate samples from Diffusion Model using reverse diffusion"""
    # Start from pure noise
    images = torch.randn(num_samples, 3, 32, 32, device=device)
    
    # Reverse diffusion process
    step_size = 1.0 / num_steps
    
    for step in range(num_steps):
        diffusion_times = torch.ones(num_samples, device=device) * (1.0 - step * step_size)
        noise_rates, signal_rates = diffusion_model.schedule_fn(diffusion_times)
        
        # Denoise
        pred_noises, pred_images = diffusion_model.denoise(images, noise_rates, signal_rates)
        
        # Take a step towards denoised image
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_model.schedule_fn(next_diffusion_times)
        
        images = next_signal_rates * pred_images + next_noise_rates * pred_noises
    
    # Final denoising
    _, images = diffusion_model.denoise(images, noise_rates, signal_rates)
    images = torch.clamp(images, -1, 1)
    
    return images


def tensor_to_pil(images):
    """
    Convert torch tensor to PIL images
    
    Args:
        images: Tensor of shape (N, C, H, W) in range [-1, 1]
        
    Returns:
        List of PIL images
    """
    # Convert from [-1, 1] to [0, 255]
    images = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    
    # Convert to numpy and then PIL
    images = images.cpu().numpy()
    pil_images = []
    
    for img in images:
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        pil_images.append(Image.fromarray(img))
    
    return pil_images


def tensor_to_base64(images):
    """
    Convert torch tensor to base64 encoded strings
    
    Args:
        images: Tensor of shape (N, C, H, W) in range [-1, 1]
        
    Returns:
        List of base64 encoded image strings
    """
    pil_images = tensor_to_pil(images)
    base64_images = []
    
    for img in pil_images:
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        base64_images.append(img_str)
    
    return base64_images

