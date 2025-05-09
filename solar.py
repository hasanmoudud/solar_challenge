#!/usr/bin/env -S submit -M 4000 -m 16000 -f python -u
import numpy as np
from natsort import natsorted
import xarray as xr

import matplotlib.pyplot as plt
import sunpy.visualization.colormaps as cm
from codecarbon import track_emissions

ds = xr.open_dataset("/net/data/erum_data/all128.nc")
print(f"{ds.channel=}")

channel_names = [ch for ch in ds.channel.values if ch != '94A']

selected_channel = ds['DN'].sel(channel='171A')
print(f"{selected_channel=}")

img = selected_channel.isel(time=0)
cmap = plt.get_cmap('sdoaia171')
img.plot(cmap=cmap)
plt.savefig("single.png")
plt.close()

keys = natsorted(ds['channel'].data)
fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for ax, key in zip(axes.ravel(), keys):
    data = ds['DN'].sel(channel=key).isel(time=0)
    cmap = plt.get_cmap(f'sdoaia{key[:-1]}')
    im = data.plot(cmap=cmap, ax=ax, add_colorbar=False)
    ax.set_title(key)
    ax.axis('off')

plt.savefig("multiple.png")
plt.close()

# now solve it ;)
type(ds)
# you have many different DeepLearning library available

import torch
import tensorflow
import keras
import jax
import numpy as np
from natsort import natsorted
import xarray as xr
import matplotlib.pyplot as plt
import sunpy.visualization.colormaps as cm
from codecarbon import track_emissions, EmissionsTracker
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim
from tqdm import tqdm
import requests
import io

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

from torch.utils.data import DataLoader, TensorDataset, random_split

# --- Normalize and Prepare Tensors ---
def normalize_data(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data.astype(np.float32)

input_data = normalize_data(ds['DN'].sel(channel='94A').data)  # (T, H, W)
target_data = normalize_data(ds['DN'].sel(channel=[c for c in ds.channel.values if c != '94A']).data)  # (C, T, H, W)
target_data = np.transpose(target_data, (1, 0, 2, 3))  # (T, C, H, W)

input_tensor = torch.tensor(input_data).unsqueeze(1).to(device)  # (T, 1, H, W)
target_tensor = torch.tensor(target_data).to(device)             # (T, C, H, W)

print(f"Input tensor shape: {input_tensor.shape}")
print(f"Target tensor shape: {target_tensor.shape}")

dataset = TensorDataset(input_tensor, target_tensor)

train_size = int(0.8 * len(dataset))
val_size = int(0.10 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=1)

print("batch_size", batch_size)

# --- Define U-Net ---
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=8):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.dec1 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec3 = self.conv_block(64 + 32, 32)
        self.out = nn.Conv2d(32, out_channels, 1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))

        dec1 = self.dec1(torch.cat([nn.Upsample(scale_factor=2)(enc4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([nn.Upsample(scale_factor=2)(dec1), enc2], dim=1))
        dec3 = self.dec3(torch.cat([nn.Upsample(scale_factor=2)(dec2), enc1], dim=1))
        return self.out(dec3)

model = UNet().to(device)

# --- Define Loss and Optimizer ---
def combined_loss(output, target, mse_weight=0.999, ssim_weight=0.001):
    mse = nn.MSELoss()(output, target)
    ssim_loss = 1 - ssim(output, target, data_range=1.0)
    return mse_weight * mse + ssim_weight * ssim_loss

optimizer = optim.Adam(model.parameters(), lr=0.0009782530226759337)

# --- Modified Training Function with DataLoader ---
@track_emissions(project_name="solar_image_reconstruction")
def train_model(model, train_loader, val_loader, epochs=10, min_gain=0.0001, min_energy=0.0001):
    tracker = EmissionsTracker()
    tracker.start()
    prev_loss = float('inf')
    prev_emissions = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = combined_loss(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                output = model(x_val)
                val_loss += combined_loss(output, y_val).item()

        curr_emissions = tracker.stop()
        tracker.start()
        gain = prev_loss - val_loss
        delta_em = curr_emissions - prev_emissions
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Emissions: {curr_emissions:.4f} kgCO2eq")

        if epoch > 10 and gain < min_gain and delta_em < min_energy:
            print("Early stopping: minimal gain and emissions")
            break

        prev_loss, prev_emissions = val_loss, curr_emissions

    tracker.stop()

train_model(model, train_loader, val_loader, epochs=200)

# save train_model
torch.save(model.state_dict(), 'model.pth')
# --- Evaluation Metrics ---
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

# --- SSIM Computation Helper ---
def ssim_per_channel(output, target):
    """
    Compute SSIM for each channel (assuming input shape: [batch, channel, H, W]).
    Returns a list of SSIM values for one batch.
    """
    if output.dim() == 3:  # [channel, H, W]
        output = output.unsqueeze(0)  # [1, channel, H, W]
        target = target.unsqueeze(0)
    return [ssim(output[:, i:i+1], target[:, i:i+1], data_range=1.0).item()
            for i in range(output.shape[1])]

# --- Test Evaluation ---
model.eval()
psnrs, ssims = [], []
with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        pred = model(x_test)

        # Compute metrics for first sample in batch
        psnrs.append([psnr(pred[0, i].cpu().numpy(), y_test[0, i].cpu().numpy()) for i in range(pred.shape[1])])
        ssims.append(ssim_per_channel(pred, y_test))
        break  # only need one batch for visual evaluation

avg_psnr = np.mean(psnrs, axis=0)
avg_ssim = np.mean(ssims, axis=0)
print("Avg PSNR per channel:", avg_psnr)
print("Avg SSIM per channel:", avg_ssim)
print(f"Overall Avg PSNR: {np.mean(avg_psnr):.4f}, Overall Avg SSIM: {np.mean(avg_ssim):.4f}")

# --- Visualize Predictions vs Ground Truth ---

fig, axs = plt.subplots(2, len(channel_names), figsize=(2 * len(channel_names), 5))
for i in range(len(channel_names)):
    # Ground truth
    axs[0, i].imshow(y_test[0, i].cpu().numpy(), cmap='gray')
    axs[0, i].set_title(f"GT: {channel_names[i]}")
    axs[0, i].axis('off')

    # Prediction
    axs[1, i].imshow(pred[0, i].cpu().numpy(), cmap='gray')
    axs[1, i].set_title(f"Pred: {channel_names[i]}")
    axs[1, i].axis('off')

plt.tight_layout()
plt.savefig("test_vs_pred.png")
plt.close()


# import optuna
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset, random_split
# from pytorch_msssim import ssim
# import xarray as xr

# # Load and preprocess data
# ds = xr.open_dataset("/net/data/erum_data/all128.nc")
# input_data = ds['DN'].sel(channel='94A').data
# target_data = ds['DN'].sel(channel=[c for c in ds.channel.values if c != '94A']).data
# input_data = (input_data - input_data.min()) / (input_data.max() - input_data.min())
# target_data = (target_data - target_data.min()) / (target_data.max() - target_data.min())
# target_data = np.transpose(target_data, (1, 0, 2, 3))

# device = "cuda" if torch.cuda.is_available() else "cpu"
# input_tensor = torch.tensor(input_data).unsqueeze(1).float()
# target_tensor = torch.tensor(target_data).float()

# dataset = TensorDataset(input_tensor, target_tensor)
# train_size = int(0.8 * len(dataset))
# val_size = int(0.1 * len(dataset))
# test_size = len(dataset) - train_size - val_size
# train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])

# # Define UNet
# class UNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=8):
#         super().__init__()
#         self.enc1 = self.conv_block(in_channels, 32)
#         self.enc2 = self.conv_block(32, 64)
#         self.enc3 = self.conv_block(64, 128)
#         self.enc4 = self.conv_block(128, 256)
#         self.dec1 = self.conv_block(256 + 128, 128)
#         self.dec2 = self.conv_block(128 + 64, 64)
#         self.dec3 = self.conv_block(64 + 32, 32)
#         self.out = nn.Conv2d(32, out_channels, 1)

#     def conv_block(self, in_c, out_c):
#         return nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
#         enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
#         enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
#         dec1 = self.dec1(torch.cat([nn.Upsample(scale_factor=2)(enc4), enc3], dim=1))
#         dec2 = self.dec2(torch.cat([nn.Upsample(scale_factor=2)(dec1), enc2], dim=1))
#         dec3 = self.dec3(torch.cat([nn.Upsample(scale_factor=2)(dec2), enc1], dim=1))
#         return self.out(dec3)

# # Define loss function
# def combined_loss(output, target, mse_weight, ssim_weight):
#     mse = nn.MSELoss()(output, target)
#     ssim_loss = 1 - ssim(output, target, data_range=1.0)
#     return mse_weight * mse + ssim_weight * ssim_loss

# # Optuna objective function
# def objective(trial):
#     # Sample hyperparameters
#     lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
#     mse_weight = trial.suggest_float('mse_weight', 0.5, 1.0)
#     ssim_weight = 1.0 - mse_weight
#     batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])

#     # Create data loaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)

#     # Initialize model, optimizer
#     model = UNet().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     # Train for a few epochs
#     epochs = 5
#     for epoch in range(epochs):
#         model.train()
#         for x_batch, y_batch in train_loader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             output = model(x_batch)
#             loss = combined_loss(output, y_batch, mse_weight, ssim_weight)
#             loss.backward()
#             optimizer.step()

#     # Evaluate on validation set
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for x_val, y_val in val_loader:
#             x_val, y_val = x_val.to(device), y_val.to(device)
#             output = model(x_val)
#             val_loss += combined_loss(output, y_val, mse_weight, ssim_weight).item()

#     avg_val_loss = val_loss / len(val_loader)
#     print(f"Trial done - LR: {lr:.1e}, MSE weight: {mse_weight:.2f}, Batch: {batch_size}, Val Loss: {avg_val_loss:.4f}")
#     return avg_val_loss

# # Run optimization
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=30)

# # Print best result
# print("Best trial:")
# trial = study.best_trial
# print(f"  Value: {trial.value:.4f}")
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")


