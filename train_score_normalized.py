
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab')
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/LSGM')

from lunar_dataset import LunarLatentDataset
from score_sde.ncsnpp_linear import NCSNppLinear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DiffusionProcess:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, l_0, t):
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
        noise = torch.randn_like(l_0)
        l_t = torch.sqrt(alpha_bar_t) * l_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return l_t, noise

def train(num_epochs=1000, batch_size=256, lr=1e-4,
          save_dir='/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/score_net_normalized'):

    os.makedirs(save_dir, exist_ok=True)

    dataset = LunarLatentDataset(
        '/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/train_latents.npy',
        '/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/train_states.npy'
    )

    # Compute normalization stats from TRAINING latents
    latent_mean = dataset.latents.mean(axis=0)
    latent_std = dataset.latents.std(axis=0)
    print(f"Latent mean: {latent_mean}")
    print(f"Latent std:  {latent_std}")
    np.save(f"{save_dir}/latent_mean.npy", latent_mean)
    np.save(f"{save_dir}/latent_std.npy", latent_std)

    latent_mean_t = torch.FloatTensor(latent_mean).to(device)
    latent_std_t = torch.FloatTensor(latent_std).to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    score_net = NCSNppLinear(
        latent_dim=3, physical_dim=3,
        nf=128, ch_mult=(1, 2, 2),
        num_res_blocks=2, temb_dim=128,
    ).to(device)

    start_epoch = 1
    best_loss = float("inf")
    ckpt_path = f"{save_dir}/best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        score_net.load_state_dict(ckpt["score_net"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["loss"]
        print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.6f}")

    diffusion = DiffusionProcess(T=1000, device=device)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=lr)

    print(f"Training (NORMALIZED latents) from epoch {start_epoch} to {num_epochs}")

    for epoch in range(start_epoch, num_epochs + 1):
        score_net.train()
        epoch_loss = 0
        num_batches = 0

        for z, f_i in loader:
            z = z.to(device)
            f_i = f_i.to(device)

            # NORMALIZE the latent before diffusion
            z_norm = (z - latent_mean_t) / latent_std_t

            t = torch.randint(0, 1000, (z.shape[0],)).to(device)
            l_t, noise = diffusion.add_noise(z_norm, t)
            predicted_noise = score_net(l_t, t, f_i)
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
                "epoch": epoch,
                "score_net": score_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": best_loss,
            }, f"{save_dir}/best.pt")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "score_net": score_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }, f"{save_dir}/epoch_{epoch}.pt")
            print(f"Epoch {epoch:4d}/{num_epochs} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")
        else:
            print(f"Epoch {epoch:4d}/{num_epochs} | Loss: {avg_loss:.6f}")

    print(f"Training complete! Best loss: {best_loss:.6f}")

if __name__ == "__main__":
    train(num_epochs=1000)
