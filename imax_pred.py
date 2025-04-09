import torch
import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

device = torch.device('cuda')

# encoder = models.UNetEncoder(n_channels=1, channels=[64, 128, 256, 512]).to(device)
# model = models.PreTrainedUNet(encoder=encoder, channels=[64, 128, 256, 512]).to(device)

# model = models.UNet(1, output_ch=5, channels=[64, 128, 256, 512]).to(device)

# model = models.UNetPP(
#     n_channels=1,
#     output_ch=5,
#     starting_ch=64,
#     level=4
# ).to(device)

best_state = torch.load('', map_location=device)
model.load_state_dict(best_state)

x = np.array(Image.open('data/imax/original/images/8.png'))
# x = image[:, :, 0]
x /= x.max()

model.eval()

xx = torch.unsqueeze(torch.Tensor(x), 0)
xx = torch.unsqueeze(xx, 1).to(device)

summed = np.zeros((5, 768, 768), dtype=np.float32)
window_w, window_h = 128, 128
stride = 32
for i in range(0, summed.shape[1] - window_h + 1, stride):
    for j in range(0, summed.shape[2] - window_w + 1, stride):
        print(i, j)
        with torch.no_grad():
            out = model(xx[:, :, i: i + window_w, j: j + window_h])
            summed[:, i: i + window_w, j: j + window_h] += (
                out.cpu().detach().numpy()[0]
            )

yhat = np.argmax(summed, axis=0)

fig, ax = plt.subplots()
cax = ax.imshow(yhat, cmap='jet_r', interpolation='nearest')
ax.axis('off')
mask_path = f'imax_val_pred.png'
plt.savefig(mask_path, bbox_inches='tight', pad_inches=0, dpi=300)
plt.close(fig)
