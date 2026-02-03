import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------
# 1. U-Net Model Architecture
# ---------------------------------------------------------

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# ---------------------------------------------------------
# 2. Synthetic Data Generation (Mimicking Figure 3: HeLa Cells)
# ---------------------------------------------------------

def create_weight_map(mask, w0=10, sigma=5):
    """
    Creates a weight map to force the network to learn borders between touching cells.
    As described in the U-Net paper Eq (2).
    """
    # Simple binary mask to multi-instance mask logic (dummy for demo)
    from scipy.ndimage import label
    labeled_array, num_features = label(mask)
    
    if num_features < 2:
        return np.ones_like(mask, dtype=np.float32)

    h, w = mask.shape
    dist_map = np.zeros((num_features, h, w))
    for i in range(1, num_features + 1):
        dist_map[i-1] = distance_transform_edt(labeled_array != i)

    dist_map = np.sort(dist_map, axis=0)
    d1 = dist_map[0]
    d2 = dist_map[1]
    
    weight = w0 * np.exp(-(d1 + d2)**2 / (2 * sigma**2))
    return (weight + 1).astype(np.float32)

def generate_fake_data(n_samples=5, size=256):
    images, masks, weights = [], [], []
    for _ in range(n_samples):
        img = np.zeros((size, size))
        mask = np.zeros((size, size), dtype=np.int64)
        # Generate random overlapping "cells"
        for _ in range(15):
            r = np.random.randint(10, 25)
            y, x = np.random.randint(r, size-r, size=2)
            Y, X = np.ogrid[:size, :size]
            dist = np.sqrt((X - x)**2 + (Y - y)**2)
            cell_area = dist <= r
            img[cell_area] = np.random.uniform(0.5, 1.0)
            mask[cell_area] = 1
        
        # Add noise
        img += np.random.normal(0, 0.1, img.shape)
        w_map = create_weight_map(mask)
        
        images.append(img[np.newaxis, ...])
        masks.append(mask)
        weights.append(w_map)
        
    return (torch.tensor(np.array(images), dtype=torch.float32), 
            torch.tensor(np.array(masks), dtype=torch.long),
            torch.tensor(np.array(weights), dtype=torch.float32))

# ---------------------------------------------------------
# 3. Training & Demo Script
# ---------------------------------------------------------

def main():
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Model
    model = UNet(n_channels=1, n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Generate Synthetic Data
    images, masks, weights = generate_fake_data(n_samples=10)
    images, masks, weights = images.to(device), masks.to(device), weights.to(device)

    # Simple Training Loop (Mini-demo)
    print("Training mini-epoch to initialize weights...")
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(images)
        # Weighted Cross Entropy
        loss = F.cross_entropy(output, masks, reduction='none')
        loss = (loss * weights).mean()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Inference for Figure 3 Demo
    model.eval()
    with torch.no_grad():
        test_img = images[0:1]
        pred = model(test_img)
        pred_mask = torch.argmax(pred, dim=1).cpu().numpy()[0]

    # ---------------------------------------------------------
    # 4. Visualization (Replicating Figure 3 Layout)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    ax[0].imshow(test_img.cpu().numpy()[0, 0], cmap='gray')
    ax[0].set_title("(a) Raw Image\n(Simulated HeLa)")
    
    ax[1].imshow(masks[0].cpu().numpy(), cmap='jet')
    ax[1].set_title("(b) Ground Truth\nSegmentation")
    
    w_plot = ax[2].imshow(weights[0].cpu().numpy(), cmap='magma')
    plt.colorbar(w_plot, ax=ax[2])
    ax[2].set_title("(c) Pixel-wise Loss\nWeight Map")
    
    ax[3].imshow(pred_mask, cmap='jet')
    ax[3].set_title("(d) U-Net Prediction\n(Result)")

    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.savefig('./unet_figure3_demo.png')
    plt.show()

if __name__ == "__main__":
    main()