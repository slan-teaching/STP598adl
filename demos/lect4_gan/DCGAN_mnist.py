import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Hyperparameters & Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 32      # MNIST is 28x28, but 32x32 is better for Conv layering
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 500     # Set high to see the 100-epoch interval logic
FEATURES_DISC = 64
FEATURES_GEN = 64

# Create a directory to save images
os.makedirs("saved_images", exist_ok=True)

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = datasets.MNIST(root="dataset/", train=True, transform=data_transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 2. Model Definitions ---
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            nn.Conv2d(features_d * 4, 1, 4, 2, 0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 4, channels_img, 4, 2, 1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# --- 3. Initialization ---
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Fixed noise for consistent visualization
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

def save_and_plot(generator, noise, epoch):
    generator.eval()
    with torch.no_grad():
        fake = generator(noise).detach().cpu()
    
    img_grid = torchvision.utils.make_grid(fake, normalize=True)
    
    # Save to disk
    torchvision.utils.save_image(img_grid, f"saved_images/epoch_{epoch}.png")
    
    # Plot to screen
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    plt.title(f"Epoch {epoch}")
    plt.axis("off")
    plt.show()
    generator.train()

# --- 4. Training Loop ---
print(f"Starting training on {device}...")

for epoch in range(1, NUM_EPOCHS + 1):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(real.shape[0], Z_DIM, 1, 1).to(device)
        fake = gen(noise)

        # Train Discriminator
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    # Visualization Logic
    is_early_stage = (epoch <= 100 and epoch % 10 == 0)
    is_late_stage = (epoch > 100 and epoch % 100 == 0)

    if is_early_stage or is_late_stage:
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")
        save_and_plot(gen, fixed_noise, epoch)

print("Training Complete.")