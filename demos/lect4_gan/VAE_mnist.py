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
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
IMAGE_SIZE = 32
CHANNELS_IMG = 1
Z_DIM = 20           # Latent space dimension
NUM_EPOCHS = 500
FEATURES_DIM = 32

os.makedirs("vae_saved_images", exist_ok=True)

transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 2. Model Definition ---
class VAE(nn.Module):
    def __init__(self, channels_img, features_d, z_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1), # 16x16
            nn.ReLU(),
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1), # 8x8
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Latent space parameters
        # After two 4x4 convs with stride 2 on 32x32 image, feature map is 8x8
        flattened_size = (features_d * 2) * 8 * 8
        self.fc_mu = nn.Linear(flattened_size, z_dim)
        self.fc_logvar = nn.Linear(flattened_size, z_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(z_dim, flattened_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (features_d * 2, 8, 8)),
            nn.ConvTranspose2d(features_d * 2, features_d, 4, 2, 1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(features_d, channels_img, 4, 2, 1), # 32x32
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        
        decoded_input = self.decoder_input(z)
        reconstruction = self.decoder(decoded_input)
        return reconstruction, mu, logvar

# --- 3. Loss and Initialization ---
model = VAE(CHANNELS_IMG, FEATURES_DIM, Z_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (BCE)
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# Fixed noise for consistent generation (sampling from latent space)
fixed_z = torch.randn(32, Z_DIM).to(device)

def save_and_plot_vae(model, z, epoch):
    model.eval()
    with torch.no_grad():
        # Pass the fixed latent vectors through the decoder only
        sample = model.decoder(model.decoder_input(z)).cpu()
    
    img_grid = torchvision.utils.make_grid(sample, normalize=True)
    torchvision.utils.save_image(img_grid, f"vae_saved_images/epoch_{epoch}.png")
    
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    plt.title(f"VAE Sample - Epoch {epoch}")
    plt.axis("off")
    plt.show()
    model.train()

# --- 4. Training Loop ---
print(f"Starting VAE training on {device}...")

for epoch in range(1, NUM_EPOCHS + 1):
    total_loss = 0
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        
        recon_batch, mu, logvar = model(real)
        loss = vae_loss(recon_batch, real, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Visualization Logic
    is_early_stage = (epoch <= 100 and epoch % 10 == 0)
    is_late_stage = (epoch > 100 and epoch % 100 == 0)

    if is_early_stage or is_late_stage:
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Avg Loss: {avg_loss:.4f}")
        save_and_plot_vae(model, fixed_z, epoch)

print("Training Complete.")