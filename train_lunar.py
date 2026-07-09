import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class IntrinsicVAE(nn.Module):
    def __init__(self, latent_dim=128, state_dim=3, image_channels=3):
        super(IntrinsicVAE, self).__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        # 96x128 → 48x64 → 24x32 → 12x16 → 6x8 → 3x4
        self.fc_mu = nn.Linear(512 * 3 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 3 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 3 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 512, 3, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class ImageStateDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.images = data['frame']
        self.states = data['state'][:, :3]  # x, y, theta
        print(f"Loaded {len(self.images)} samples from {npz_path}")
        print(f"Image shape: {self.images.shape}")
        print(f"State shape: {self.states.shape}")
        print(f"x — mean: {self.states[:,0].mean():.4f}, std: {self.states[:,0].std():.4f}")
        print(f"y — mean: {self.states[:,1].mean():.4f}, std: {self.states[:,1].std():.4f}")
        print(f"θ — mean: {self.states[:,2].mean():.4f}, std: {self.states[:,2].std():.4f}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC → CHW
        state = torch.from_numpy(self.states[idx].astype(np.float32))
        return image, state


def intrinsic_vae_loss(recon_x, x, mu, logvar, state, state_weight=1000.0, kl_weight=1.0):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    state_pred = mu[:, :3]  # first 3 dims predict x, y, theta
    state_loss = nn.functional.mse_loss(state_pred, state, reduction='sum')
    total_loss = BCE + kl_weight * KLD + state_weight * state_loss
    return total_loss, BCE, KLD, state_loss


def main():
    latent_dim = 128
    state_dim = 3
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    kl_weight = 1.0
    state_weight = 1000.0

    data_path = '/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_train_intrinsic.npz'
    save_dir = '/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/intrinsic_vae'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = ImageStateDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = IntrinsicVAE(latent_dim=latent_dim, state_dim=state_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Latent dim: {latent_dim}, State dim: {state_dim}")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
    print(f"State weight: {state_weight}, KL weight: {kl_weight}\n")

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_bce = 0
        total_kld = 0
        total_state_loss = 0

        for batch_img, batch_state in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_img = batch_img.to(device)
            batch_state = batch_state.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_img)
            loss, bce, kld, state_loss = intrinsic_vae_loss(
                recon_batch, batch_img, mu, logvar, batch_state, state_weight, kl_weight
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()
            total_state_loss += state_loss.item()

        n = len(dataset)
        avg_loss = total_loss / n
        avg_bce = total_bce / n
        avg_kld = total_kld / n
        avg_state = total_state_loss / n

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} "
              f"BCE: {avg_bce:.4f} KLD: {avg_kld:.4f} State: {avg_state:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'latent_dim': latent_dim,
                'state_dim': state_dim,
            }, f"{save_dir}/best.pt")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'latent_dim': latent_dim,
                'state_dim': state_dim,
            }, f"{save_dir}/epoch_{epoch+1}.pt")
            print(f"✅ Saved checkpoint epoch {epoch+1}")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")

if __name__ == '__main__':
    main()
