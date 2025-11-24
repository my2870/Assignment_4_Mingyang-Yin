"""Model definitions for GAN, Energy-Based Models, and Diffusion Models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== GAN Models ====================

class Discriminator(nn.Module):
    """DCGAN-style Discriminator for CIFAR-10"""
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64, momentum=0.9)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128, momentum=0.9)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256, momentum=0.9)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x


class Generator(nn.Module):
    """DCGAN-style Generator for CIFAR-10"""
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        self.reshape = lambda x: x.view(x.size(0), z_dim, 1, 1)
        
        self.deconv1 = nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512, momentum=0.9)
        self.act1 = nn.ReLU(True)
        
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.act2 = nn.ReLU(True)
        
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
        self.act3 = nn.ReLU(True)
        
        self.deconv4 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.reshape(x)
        
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        
        x = self.deconv4(x)
        x = self.tanh(x)
        
        return x


# ==================== Energy-Based Model ====================

def swish(x):
    """Swish activation function"""
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """Energy-Based Model for CIFAR-10"""
    def __init__(self, num_channels=3):
        super(EnergyModel, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


# ==================== Diffusion Model Components ====================

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for noise variance"""
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        
    def forward(self, noise_variances):
        # noise_variances: (B,)
        frequencies = torch.exp(
            torch.linspace(
                0.0, 
                np.log(1000.0), 
                self.num_frequencies,
                device=noise_variances.device
            )
        )
        angular_speeds = 2.0 * np.pi * frequencies
        embeddings = torch.cat([
            torch.sin(angular_speeds[None, :] * noise_variances[:, None]),
            torch.cos(angular_speeds[None, :] * noise_variances[:, None])
        ], dim=1)
        return embeddings[:, None, None, :]  # (B, 1, 1, 2*num_frequencies)


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(F.relu(self.norm1(x)))
        x = self.conv2(F.relu(self.norm2(x)))
        return x + residual


class DownBlock(nn.Module):
    """Downsampling block with residual connections"""
    def __init__(self, out_channels, in_channels=None, block_depth=2):
        super().__init__()
        if in_channels is None:
            in_channels = out_channels
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(out_channels) for _ in range(block_depth)
        ])
        self.downsample = nn.AvgPool2d(2)
        
    def forward(self, x, skips):
        x = self.conv(x)
        for block in self.residual_blocks:
            x = block(x)
            skips.append(x)
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""
    def __init__(self, out_channels, in_channels=None, block_depth=2):
        super().__init__()
        if in_channels is None:
            in_channels = out_channels
            
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(out_channels) for _ in range(block_depth)
        ])
        
    def forward(self, x, skips):
        x = self.upsample(x)
        x = self.conv(x)
        for block in self.residual_blocks:
            skip = skips.pop()
            x = x + skip
            x = block(x)
        return x


class UNet(nn.Module):
    """UNet architecture for diffusion model"""
    def __init__(self, image_size, num_channels, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        
        self.down1 = DownBlock(32, in_channels=32+embedding_dim, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2)
        
        self.mid1 = ResidualBlock(96)
        self.mid2 = ResidualBlock(96)
        
        self.up1 = UpBlock(96, in_channels=96, block_depth=2)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)
        
        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)
        
    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)  # (B, 1, 1, 2*num_frequencies)
        # Reshape from (B, 1, 1, C) to (B, C, 1, 1) for interpolation
        B = noise_emb.shape[0]
        noise_emb = noise_emb.view(B, -1, 1, 1)  # (B, 32, 1, 1)
        # Interpolate to match image size
        noise_emb = F.interpolate(
            noise_emb, 
            size=(self.image_size, self.image_size), 
            mode='nearest'
        )
        x = torch.cat([x, noise_emb], dim=1)
        
        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)
        
        x = self.mid1(x)
        x = self.mid2(x)
        
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        
        return self.final(x)


class DiffusionModel(nn.Module):
    """Diffusion Model wrapper with noise scheduling"""
    def __init__(self, network, schedule_fn):
        super().__init__()
        self.network = network
        self.schedule_fn = schedule_fn
        
    def denoise(self, noisy_images, noise_rates, signal_rates):
        pred_noises = self.network(noisy_images, noise_rates**2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images
    
    def forward(self, images):
        batch_size = images.shape[0]
        noise = torch.randn_like(images)
        
        diffusion_times = torch.rand(batch_size, device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        
        noisy_images = signal_rates * images + noise_rates * noise
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates)
        
        return pred_noises, noise


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """Offset cosine diffusion schedule"""
    start_angle = torch.acos(torch.tensor(max_signal_rate))
    end_angle = torch.acos(torch.tensor(min_signal_rate))
    
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    
    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)
    
    return noise_rates[:, None, None, None], signal_rates[:, None, None, None]


# ==================== Model Factory ====================

def get_model(model_name, **kwargs):
    """Factory function to get model by name"""
    if model_name == 'GAN':
        return Generator(kwargs.get('z_dim', 100)), Discriminator()
    elif model_name == 'EBM':
        return EnergyModel(kwargs.get('num_channels', 3))
    elif model_name == 'Diffusion':
        unet = UNet(
            kwargs.get('image_size', 32), 
            kwargs.get('num_channels', 3),
            kwargs.get('embedding_dim', 32)
        )
        return DiffusionModel(unet, offset_cosine_diffusion_schedule)
    else:
        raise ValueError(f"Unknown model: {model_name}")

