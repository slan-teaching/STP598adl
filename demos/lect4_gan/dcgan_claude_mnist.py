"""
DC-GAN Implementation for MNIST Dataset
Generates and saves images periodically during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
BATCH_SIZE = 128
LATENT_DIM = 100
NUM_EPOCHS = 500
LEARNING_RATE = 0.0002
BETA1 = 0.5
IMAGE_SIZE = 28
NUM_CHANNELS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Create output directories
os.makedirs("generated_images", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 7 x 7
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 14 x 14
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: 64 x 28 x 28
            
            nn.Conv2d(64, 1, 3, 1, 1, bias=False),
            nn.Tanh()
            # Output: 1 x 28 x 28
        )
    
    def forward(self, z):
        return self.main(z)


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 64 x 14 x 14
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128 x 7 x 7
            
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 256 x 4 x 4
            
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1)


# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Function to save generated images
def save_generated_images(generator, epoch, num_images=64, fixed_noise=None):
    generator.eval()
    with torch.no_grad():
        if fixed_noise is None:
            noise = torch.randn(num_images, LATENT_DIM, 1, 1, device=DEVICE)
        else:
            noise = fixed_noise
        
        fake_images = generator(noise).cpu()
        
        # Denormalize from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
        
        # Create grid
        fig, axes = plt.subplots(8, 8, figsize=(12, 12))
        fig.suptitle(f'Generated Images at Epoch {epoch}', fontsize=16)
        
        for idx, ax in enumerate(axes.flat):
            if idx < num_images:
                img = fake_images[idx].squeeze().numpy()
                ax.imshow(img, cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'generated_images/epoch_{epoch:04d}.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    generator.train()


# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Initialize models
generator = Generator(LATENT_DIM).to(DEVICE)
discriminator = Discriminator().to(DEVICE)

# Apply weight initialization
generator.apply(weights_init)
discriminator.apply(weights_init)

print("\nGenerator Architecture:")
print(generator)
print("\nDiscriminator Architecture:")
print(discriminator)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

# Fixed noise for consistent visualization
fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

# Training history
G_losses = []
D_losses = []

print(f"\n{'='*60}")
print(f"Starting Training on {DEVICE}")
print(f"{'='*60}\n")

# Training loop
for epoch in range(1, NUM_EPOCHS + 1):
    epoch_G_loss = 0.0
    epoch_D_loss = 0.0
    num_batches = 0
    
    for batch_idx, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(DEVICE)
        
        # Labels
        real_labels = torch.ones(batch_size, 1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)
        
        # ==================
        # Train Discriminator
        # ==================
        optimizer_D.zero_grad()
        
        # Real images
        output_real = discriminator(real_images)
        loss_D_real = criterion(output_real, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())
        loss_D_fake = criterion(output_fake, fake_labels)
        
        # Total discriminator loss
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()
        
        # ==============
        # Train Generator
        # ==============
        optimizer_G.zero_grad()
        
        # Generate fake images and get discriminator output
        output = discriminator(fake_images)
        loss_G = criterion(output, real_labels)  # Generator wants discriminator to think images are real
        
        loss_G.backward()
        optimizer_G.step()
        
        # Accumulate losses
        epoch_G_loss += loss_G.item()
        epoch_D_loss += loss_D.item()
        num_batches += 1
    
    # Average losses for the epoch
    avg_G_loss = epoch_G_loss / num_batches
    avg_D_loss = epoch_D_loss / num_batches
    G_losses.append(avg_G_loss)
    D_losses.append(avg_D_loss)
    
    # Print progress
    print(f"Epoch [{epoch}/{NUM_EPOCHS}] | "
          f"D Loss: {avg_D_loss:.4f} | "
          f"G Loss: {avg_G_loss:.4f}")
    
    # Save generated images based on epoch rules
    should_save = False
    if epoch <= 100 and epoch % 10 == 0:
        should_save = True
    elif epoch > 100 and epoch % 100 == 0:
        should_save = True
    
    if should_save:
        print(f"  → Saving generated images at epoch {epoch}")
        save_generated_images(generator, epoch, fixed_noise=fixed_noise)
    
    # Save checkpoint every 100 epochs
    if epoch % 100 == 0:
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses,
        }
        torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch}.pth')
        print(f"  → Checkpoint saved at epoch {epoch}")

print(f"\n{'='*60}")
print("Training Complete!")
print(f"{'='*60}\n")

# Plot training losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
plt.close()

print("Loss plot saved as 'training_loss.png'")

# Generate final set of images
print("\nGenerating final images...")
save_generated_images(generator, NUM_EPOCHS, fixed_noise=fixed_noise)

# Save final model
final_checkpoint = {
    'epoch': NUM_EPOCHS,
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'G_losses': G_losses,
    'D_losses': D_losses,
}
torch.save(final_checkpoint, 'dcgan_mnist_final.pth')
print("\nFinal model saved as 'dcgan_mnist_final.pth'")

print("\n" + "="*60)
print("All outputs saved:")
print("  - Generated images: ./generated_images/")
print("  - Checkpoints: ./checkpoints/")
print("  - Loss plot: ./training_loss.png")
print("  - Final model: ./dcgan_mnist_final.pth")
print("="*60)
