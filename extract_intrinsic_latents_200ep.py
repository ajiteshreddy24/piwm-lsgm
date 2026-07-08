import sys
import torch
import numpy as np
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/PIWM/in-conti')
from train_lunar import IntrinsicVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = IntrinsicVAE(latent_dim=128, state_dim=3).to(device)
ckpt = torch.load('/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/intrinsic_vae_200ep/best.pt',
                  map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"✅ 200ep model loaded!")

batch_size = 256

# Training latents
print("Extracting training latents...")
train_data = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_train_intrinsic.npz')
train_imgs = train_data['frame']
train_states = train_data['state']

all_mu = []
for i in range(0, len(train_imgs), batch_size):
    batch = torch.FloatTensor(train_imgs[i:i+batch_size]).permute(0,3,1,2).to(device) / 255.0
    with torch.no_grad():
        h = model.encoder(batch)
        h = h.reshape(h.size(0), -1)
        mu = model.fc_mu(h)
    all_mu.append(mu.cpu().numpy())
    if i % 5000 == 0:
        print(f"  {i}/{len(train_imgs)} done...")

all_mu = np.concatenate(all_mu, axis=0)
np.save('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/intrinsic_z_fixed_train_200ep.npy', all_mu[:, :3])
np.save('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/intrinsic_z_unint_train_200ep.npy', all_mu[:, 3:])
print(f"✅ Saved train latents: z_fixed {all_mu[:,:3].shape}, z_unint {all_mu[:,3:].shape}")

# Test latents
print("Extracting test latents...")
test_data = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_test_intrinsic.npz')
test_imgs = test_data['frame']

all_mu_test = []
for i in range(0, len(test_imgs), batch_size):
    batch = torch.FloatTensor(test_imgs[i:i+batch_size]).permute(0,3,1,2).to(device) / 255.0
    with torch.no_grad():
        h = model.encoder(batch)
        h = h.reshape(h.size(0), -1)
        mu = model.fc_mu(h)
    all_mu_test.append(mu.cpu().numpy())

all_mu_test = np.concatenate(all_mu_test, axis=0)
np.save('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/intrinsic_z_fixed_test_200ep.npy', all_mu_test[:, :3])
np.save('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/intrinsic_z_unint_test_200ep.npy', all_mu_test[:, 3:])
print(f"✅ Saved test latents!")

# Check correlations
print(f"\nz_fixed correlation with true states (200ep):")
for dim, name in enumerate(['x', 'y', 'θ']):
    corr = np.corrcoef(all_mu[:, dim], train_states[:, dim])[0,1]
    print(f"  z_fixed[{dim}] vs true {name}: {corr:.4f}")
