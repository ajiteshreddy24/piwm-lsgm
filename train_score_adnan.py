import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab')
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/LSGM')
from score_sde.ncsnpp_linear import NCSNppLinear
from setup import DiffusionProcess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AdnanLatentDataset(Dataset):
    def __init__(self, z_scene_path, z_pose_path):
        z_scene = np.load(z_scene_path)
        z_pose = np.load(z_pose_path)
        # Filter out failed frames (zeros)
        valid = np.any(z_scene != 0, axis=1)
        self.z_scene = z_scene[valid]
        self.z_pose = z_pose[valid]
        print(f"Loaded {len(self.z_scene)} valid frames (filtered {(~valid).sum()} failed)")
        print(f"z_scene shape: {self.z_scene.shape}")
        print(f"z_pose shape: {self.z_pose.shape}")

    def __len__(self):
        return len(self.z_scene)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.z_scene[idx]),
                torch.FloatTensor(self.z_pose[idx]))

# Load data
dataset = AdnanLatentDataset(
    '/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/adnan_z_scene_train.npy',
    '/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/adnan_z_pose_train.npy'
)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

# Compute normalization stats for z_scene
print("\nComputing normalization stats...")
z_scene_mean = dataset.z_scene.mean(axis=0)
z_scene_std = dataset.z_scene.std(axis=0)
z_scene_std = np.where(z_scene_std < 1e-8, 1.0, z_scene_std)

save_dir = '/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/score_net_adnan'
os.makedirs(save_dir, exist_ok=True)
np.save(f'{save_dir}/z_scene_mean.npy', z_scene_mean)
np.save(f'{save_dir}/z_scene_std.npy', z_scene_std)
print(f"z_scene mean shape: {z_scene_mean.shape}")
print(f"z_scene std shape: {z_scene_std.shape}")

z_scene_mean_t = torch.FloatTensor(z_scene_mean).to(device)
z_scene_std_t = torch.FloatTensor(z_scene_std).to(device)

# Score network — latent_dim=28 (z_scene), physical_dim=4 (z_pose)
score_net = NCSNppLinear(
    latent_dim=28,
    physical_dim=4,
    nf=128,
    ch_mult=(1, 2, 2),
    num_res_blocks=2,
    temb_dim=128
).to(device)

print(f"\nScore network parameters: {sum(p.numel() for p in score_net.parameters()):,}")
print(f"latent_dim=28 (z_scene), physical_dim=4 (z_pose)")

diffusion = DiffusionProcess(T=1000, device=device)
optimizer = torch.optim.Adam(score_net.parameters(), lr=1e-4)

num_epochs = 1000
best_loss = float('inf')

print(f"\nTraining score network...")
print(f"Epochs: {num_epochs}, Batch size: 256\n")

for epoch in range(1, num_epochs + 1):
    score_net.train()
    epoch_loss = 0
    num_batches = 0

    for z_scene, z_pose in dataloader:
        z_scene = z_scene.to(device)
        z_pose = z_pose.to(device)

        # Normalize z_scene
        z_scene_norm = (z_scene - z_scene_mean_t) / z_scene_std_t

        # Random timestep
        t = torch.randint(0, 1000, (z_scene.shape[0],)).to(device)

        # Add noise
        l_t, noise = diffusion.add_noise(z_scene_norm, t)

        # Score network predicts noise conditioned on z_pose
        predicted_noise = score_net(l_t, t, z_pose)

        loss = nn.functional.mse_loss(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'score_net': score_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': best_loss,
        }, f'{save_dir}/best.pt')

    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d}/{num_epochs} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")

print(f"\nTraining complete! Best loss: {best_loss:.6f}")
