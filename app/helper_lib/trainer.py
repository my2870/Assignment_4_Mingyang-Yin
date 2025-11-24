"""Training functions for GAN, EBM, and Diffusion models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ==================== GAN Training ====================

def train_gan(generator, discriminator, data_loader, criterion, optimizer_g, optimizer_d, 
              device='cpu', epochs=10):
    """
    Train GAN model
    
    Args:
        generator: Generator model
        discriminator: Discriminator model  
        data_loader: DataLoader for training data
        criterion: Loss function
        optimizer_g: Optimizer for generator
        optimizer_d: Optimizer for discriminator
        device: Device to train on
        epochs: Number of epochs
        
    Returns:
        Trained generator and discriminator models
    """
    generator.train()
    discriminator.train()
    
    for epoch in range(epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        
        for batch_idx, (real_images, _) in enumerate(tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train Discriminator
            optimizer_d.zero_grad()
            
            # Real images
            real_labels = torch.ones(batch_size, 1, device=device)
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, generator.z_dim, device=device)
            fake_images = generator(z)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_g.step()
            
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
        
        avg_d_loss = epoch_d_loss / len(data_loader)
        avg_g_loss = epoch_g_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{epochs}] D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}')
    
    return generator, discriminator


# ==================== Energy-Based Model Training ====================

class Buffer:
    """Replay buffer for energy-based model training"""
    def __init__(self, model, buffer_size=10000, device='cpu'):
        self.buffer_size = buffer_size
        self.model = model
        self.device = device
        self.examples = []
        
    def __len__(self):
        return len(self.examples)
    
    def push(self, samples):
        samples = samples.detach().cpu()
        for sample in samples:
            self.examples.append(sample)
            if len(self.examples) > self.buffer_size:
                self.examples.pop(0)
    
    def get(self, batch_size, steps=60, step_size=10, noise=0.005):
        """Sample from buffer and run MCMC"""
        n_new = batch_size if len(self.examples) == 0 else batch_size // 2
        n_old = batch_size - n_new
        
        # Initialize samples
        new_samples = torch.rand(n_new, 3, 32, 32) * 2 - 1
        old_samples = []
        
        if n_old > 0:
            indices = torch.randint(0, len(self.examples), (n_old,))
            old_samples = torch.stack([self.examples[i] for i in indices])
        
        if len(old_samples) > 0:
            init_samples = torch.cat([new_samples, old_samples], dim=0)
        else:
            init_samples = new_samples
            
        # Run MCMC
        init_samples = init_samples.to(self.device)
        samples = self.sample_langevin(init_samples, steps, step_size, noise)
        
        self.push(samples)
        return samples
    
    def sample_langevin(self, x, steps, step_size, noise):
        """Langevin MCMC sampling"""
        x = x.clone().detach()
        for _ in range(steps):
            x.requires_grad = True
            energy = self.model(x).sum()
            grad = torch.autograd.grad(energy, x)[0]
            x = x.detach()
            x = x - step_size * grad + noise * torch.randn_like(x)
            x = torch.clamp(x, -1, 1)
        return x.detach()


def train_ebm(model, data_loader, optimizer, device='cpu', epochs=10, 
              alpha=0.1, steps=60, step_size=10, noise=0.005):
    """
    Train Energy-Based Model
    
    Args:
        model: Energy model
        data_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs
        alpha: Weight for regularization
        steps: MCMC steps
        step_size: MCMC step size
        noise: MCMC noise
        
    Returns:
        Trained energy model
    """
    model.train()
    buffer = Buffer(model, device=device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_reg_loss = 0.0
        
        for batch_idx, (real_images, _) in enumerate(tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Get negative samples from buffer
            fake_images = buffer.get(batch_size, steps, step_size, noise)
            
            # Compute energies
            real_energy = model(real_images)
            fake_energy = model(fake_images)
            
            # Contrastive divergence loss
            cd_loss = (real_energy - fake_energy).mean()
            
            # Regularization (encourage low energy for real data)
            reg_loss = alpha * (real_energy ** 2 + fake_energy ** 2).mean()
            
            loss = cd_loss + reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += cd_loss.item()
            epoch_reg_loss += reg_loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        avg_reg_loss = epoch_reg_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{epochs}] CD_loss: {avg_loss:.4f}, Reg_loss: {avg_reg_loss:.4f}')
    
    return model


# ==================== Diffusion Model Training ====================

def train_diffusion(model, data_loader, optimizer, loss_fn, device='cpu', epochs=10):
    """
    Train Diffusion Model
    
    Args:
        model: Diffusion model
        data_loader: DataLoader for training data
        optimizer: Optimizer
        loss_fn: Loss function (typically L1Loss or MSELoss)
        device: Device to train on
        epochs: Number of epochs
        
    Returns:
        Trained diffusion model
    """
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            images = images.to(device)
            
            # Forward pass
            pred_noises, actual_noises = model(images)
            
            # Compute loss
            loss = loss_fn(pred_noises, actual_noises)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}')
    
    return model

