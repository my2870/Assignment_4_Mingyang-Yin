"""Training script for all models"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os

from helper_lib.model1 import get_model
from helper_lib.trainer import train_gan, train_ebm, train_diffusion


def get_cifar10_dataloader(batch_size=128, num_workers=2):
    """Get CIFAR-10 dataloader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader


def train_gan_model(device, epochs=50, batch_size=128, lr=0.0002):
    """Train GAN model"""
    print("=" * 60)
    print("Training GAN Model")
    print("=" * 60)
    
    # Get data
    train_loader = get_cifar10_dataloader(batch_size)
    
    # Create models
    generator, discriminator = get_model('GAN', z_dim=100)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Train
    generator, discriminator = train_gan(
        generator, discriminator, train_loader,
        criterion, optimizer_g, optimizer_d,
        device=device, epochs=epochs
    )
    
    # Save models
    os.makedirs('models', exist_ok=True)
    torch.save(generator.state_dict(), 'models/gan_generator.pth')
    torch.save(discriminator.state_dict(), 'models/gan_discriminator.pth')
    print("✓ GAN models saved to models/")
    
    return generator, discriminator


def train_ebm_model(device, epochs=30, batch_size=128, lr=0.0001):
    """Train Energy-Based Model"""
    print("=" * 60)
    print("Training Energy-Based Model")
    print("=" * 60)
    
    # Get data
    train_loader = get_cifar10_dataloader(batch_size)
    
    # Create model
    ebm = get_model('EBM', num_channels=3)
    ebm = ebm.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(ebm.parameters(), lr=lr, betas=(0.0, 0.999))
    
    # Train
    ebm = train_ebm(
        ebm, train_loader, optimizer,
        device=device, epochs=epochs,
        alpha=0.1, steps=60, step_size=10, noise=0.005
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(ebm.state_dict(), 'models/ebm_model.pth')
    print("✓ EBM model saved to models/")
    
    return ebm


def train_diffusion_model(device, epochs=30, batch_size=128, lr=0.001):
    """Train Diffusion Model"""
    print("=" * 60)
    print("Training Diffusion Model")
    print("=" * 60)
    
    # Get data
    train_loader = get_cifar10_dataloader(batch_size)
    
    # Create model
    diffusion = get_model('Diffusion', image_size=32, num_channels=3, embedding_dim=32)
    diffusion = diffusion.to(device)
    
    # Loss and optimizer
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(diffusion.network.parameters(), lr=lr, weight_decay=1e-4)
    
    # Train
    diffusion = train_diffusion(
        diffusion, train_loader, optimizer, loss_fn,
        device=device, epochs=epochs
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(diffusion.state_dict(), 'models/diffusion_model.pth')
    print("✓ Diffusion model saved to models/")
    
    return diffusion


def main():
    parser = argparse.ArgumentParser(description='Train generative models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['gan', 'ebm', 'diffusion', 'all'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\nUsing device: {device}\n")
    
    # Train models
    if args.model in ['gan', 'all']:
        train_gan_model(device, epochs=args.epochs, batch_size=args.batch_size)
    
    if args.model in ['ebm', 'all']:
        train_ebm_model(device, epochs=args.epochs, batch_size=args.batch_size)
    
    if args.model in ['diffusion', 'all']:
        train_diffusion_model(device, epochs=args.epochs, batch_size=args.batch_size)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

