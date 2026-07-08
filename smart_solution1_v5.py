import sys
import torch
import numpy as np
from tqdm import tqdm
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab')
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/LSGM')
from lunar_vae import load_vae
from score_sde.ncsnpp_linear import NCSNppLinear
from setup import DiffusionProcess
from constraint_checker import constraint_checker

encoder, decoder, device = load_vae()

latent_mean = np.load('/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/score_net_normalized/latent_mean.npy')
latent_std = np.load('/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/score_net_normalized/latent_std.npy')
latent_mean_t = torch.FloatTensor(latent_mean).to(device)
latent_std_t = torch.FloatTensor(latent_std).to(device)

score_net_norm = NCSNppLinear(latent_dim=3, physical_dim=3, nf=128, ch_mult=(1,2,2), num_res_blocks=2, temb_dim=128).to(device)
ckpt = torch.load('/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/score_net_normalized/best.pt', map_location=device, weights_only=False)
score_net_norm.load_state_dict(ckpt['score_net'])
score_net_norm.eval()

diffusion = DiffusionProcess(T=1000, device=device)

imgs = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_imgs_100x150.npy')
states = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_states_xy_angle.npy')

def reverse_diffusion_normalized(score_net, l_noisy_norm, f_i, diffusion, start_t=20, num_steps=100):
    score_net.eval()
    l_t = l_noisy_norm.clone()
    timesteps = torch.linspace(start_t, 0, num_steps).long()
    with torch.no_grad():
        for t_val in timesteps:
            t = t_val.expand(l_t.shape[0]).to(device)
            alpha_bar_t = diffusion.alpha_bars[t_val].to(device)
            alpha_bar_prev = diffusion.alpha_bars[t_val-1].to(device) if t_val > 0 else torch.tensor(1.0).to(device)
            predicted_noise = score_net(l_t, t, f_i)
            l_0_pred = (l_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            if t_val > 0:
                noise = torch.randn_like(l_t)
                sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                l_t = torch.sqrt(alpha_bar_prev) * l_0_pred + torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * predicted_noise + sigma_t * noise
            else:
                l_t = l_0_pred
    return l_t

eps_x = 0.05
eps_y = 0.10
eps_theta = 0.50
t_level = 20
max_iterations = 20
BAD_X_THRESH = 0.20
BAD_Y_THRESH = 0.30

results_v5 = []
skipped_bad = 0
visible_indices = [i for i in range(len(imgs)) if states[i,1] < 1.3]
print(f"Total visible frames: {len(visible_indices)}")
print(f"Running Smart Solution #1 v5 (20 iterations)...")

for idx in tqdm(visible_indices):
    img = imgs[idx]
    true_x, true_y, true_theta = states[idx,0], states[idx,1], states[idx,2]

    img_tensor = torch.FloatTensor(img).permute(2,0,1).unsqueeze(0).to(device)
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0
    with torch.no_grad():
        z = encoder(img_tensor)

    with torch.no_grad():
        img_decoded_init = decoder(z).permute(0,2,3,1).cpu().numpy()[0]
    result_init = constraint_checker(img_decoded_init, true_x, true_y, true_theta)

    if not result_init['visible']:
        skipped_bad += 1
        continue
    if result_init['x_err'] > BAD_X_THRESH or result_init['y_err'] > BAD_Y_THRESH:
        skipped_bad += 1
        continue

    f_i_full = torch.FloatTensor([[true_x, true_y, true_theta]]).to(device)
    frame_history = []

    for iteration in range(max_iterations + 1):
        with torch.no_grad():
            img_decoded = decoder(z).permute(0,2,3,1).cpu().numpy()[0]

        result = constraint_checker(img_decoded, true_x, true_y, true_theta)

        x_err = result['x_err'] if result['visible'] else None
        y_err = result['y_err'] if result['visible'] else None
        theta_err = result['theta_err'] if result['visible'] and result.get('pixel_count',0) >= 50 else None

        x_ok = x_err is not None and x_err <= eps_x
        y_ok = y_err is not None and y_err <= eps_y
        theta_ok = theta_err is not None and theta_err <= eps_theta

        frame_history.append({
            'iteration': iteration,
            'x_err': x_err, 'y_err': y_err, 'theta_err': theta_err,
            'x_ok': x_ok, 'y_ok': y_ok, 'theta_ok': theta_ok,
            'xy_ok': x_ok and y_ok,
            'all_ok': x_ok and y_ok and theta_ok
        })

        if iteration == max_iterations:
            break

        z_before = z.clone()

        z_norm = (z - latent_mean_t) / latent_std_t
        t_batch = torch.tensor([t_level]).to(device)
        l_noisy_norm, _ = diffusion.add_noise(z_norm, t_batch)
        l_corr_norm = reverse_diffusion_normalized(
            score_net_norm, l_noisy_norm, f_i_full, diffusion,
            start_t=t_level, num_steps=100
        )
        z_corrected = l_corr_norm * latent_std_t + latent_mean_t

        if x_ok:
            z_corrected[:, 0] = z_before[:, 0]
        if y_ok:
            z_corrected[:, 1] = z_before[:, 1]
        if theta_ok:
            z_corrected[:, 2] = z_before[:, 2]

        z = z_corrected

    results_v5.append({'idx': idx, 'history': frame_history})

print(f"\nSkipped badly encoded frames: {skipped_bad}")
print(f"Frames evaluated: {len(results_v5)}")

np.save('/blue/iruchkin/aj.byreddy/tea_lab/results_smart_solution1_v5.npy', results_v5)
print("Results saved!")

print("\n" + "="*65)
print("SMART SOLUTION #1 v5 — 20 iterations")
print("="*65)
for it in range(max_iterations + 1):
    x_errs = [r['history'][it]['x_err'] for r in results_v5 if r['history'][it]['x_err'] is not None]
    y_errs = [r['history'][it]['y_err'] for r in results_v5 if r['history'][it]['y_err'] is not None]
    theta_errs = [r['history'][it]['theta_err'] for r in results_v5 if r['history'][it]['theta_err'] is not None]
    xy_ok = sum(r['history'][it]['xy_ok'] for r in results_v5)
    all_ok = sum(r['history'][it]['all_ok'] for r in results_v5)
    print(f"\nIteration {it} ({len(results_v5)} frames):")
    print(f"  x error: mean={np.mean(x_errs):.4f} median={np.median(x_errs):.4f}")
    print(f"  y error: mean={np.mean(y_errs):.4f} median={np.median(y_errs):.4f}")
    if theta_errs:
        print(f"  theta error: mean={np.mean(theta_errs):.4f} median={np.median(theta_errs):.4f}")
    print(f"  x,y satisfied: {xy_ok}/{len(results_v5)} ({100*xy_ok/len(results_v5):.1f}%)")
    print(f"  all satisfied: {all_ok}/{len(results_v5)} ({100*all_ok/len(results_v5):.1f}%)")
