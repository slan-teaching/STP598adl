import torch
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)

# Download an example image
import urllib
url, filename = ("https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png", "TCGA_CS_4944.png")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

input_image = Image.open(filename)
input_np = np.array(input_image)
# compute per-channel mean/std from numpy array
m, s = np.mean(input_np, axis=(0, 1)), np.std(input_np, axis=(0, 1))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=m, std=s),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model = model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# convert model output to probability map and mask
prob = torch.sigmoid(output)[0, 0].cpu().numpy()
mask = prob > 0.5

# plot input, probability map, and overlay
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(input_np)
axes[0].set_title('Input')
axes[0].axis('off')

axes[1].imshow(prob, cmap='gray')
axes[1].set_title('Predicted Probability')
axes[1].axis('off')

axes[2].imshow(input_np)
axes[2].imshow(mask, cmap='jet', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
out_path = 'unet_prediction.png'
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved plot to {out_path}")