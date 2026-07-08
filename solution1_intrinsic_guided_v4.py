import sys
import torch
import numpy as np
from tqdm import tqdm
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab')
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/LSGM')
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/PIWM/in-conti')
from score_sde.ncsnpp_linear import NCSNppLinear
from setup import DiffusionProcess
from constraint_checker import constraint_checker
from differentiable_checker import differentiable_constraint_loss
from train_lunar import IntrinsicVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Zhenjiang's 200ep VAE
vae = IntrinsicVAE(latent_dim=128, state_dim=3).to(device)
ckpt_vae = torch.load('/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/intrinsic_vae_200ep/best.pt',
                      map_location=device, weights_only=False)
vae.load_state_dict(ckpt_vae['model_state_dict'])
vae.eval()
print(f"✅ Zhenjiang VAE loaded!")

# Load guided score network
score_net = NCSNppLinear(
    latent_dim=125, physical_dim=3,
    nf=128, ch_mult=(1,2,2),
    num_res_blocks=2, temb_dim=128
).to(device)
ckpt = torch.load('/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/score_net_intrinsic_guided/best.pt',
                  map_location=device, weights_only=False)
score_net.load_state_dict(ckpt['score_net'])
score_net.eval()
print(f"✅ Guided score network loaded! Epoch {ckpt['epoch']}, Loss {ckpt['loss']:.6f}")

# Load normalization stats
z_unint_mean = np.load('/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/score_net_intrinsic_guided/z_unint_mean.npy')
z_unint_std = np.load('/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/score_net_intrinsic_guided/z_unint_std.npy')
z_unint_mean_t = torch.FloatTensor(z_unint_mean).to(device)
z_unint_std_t = torch.FloatTensor(z_unint_std).to(device)

diffusion = DiffusionProcess(T=1000, device=device)

# Load test data
test_imgs = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_imgs_100x150.npy')
test_states = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_states_xy_angle.npy')
test_data = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_test_intrinsic.npz')
test_imgs_96 = test_data['frame']  # 96x128 for VAE

def reverse_diffusion_guided(score_net, z_unint_noisy_norm, z_fixed,
                              diffusion, vae, z_unint_mean_t, z_unint_std_t,
                              true_x, true_y, start_t=20, num_steps=100,
                              guidance_scale=0.1):
    """
    Reverse diffusion with constraint guidance.
    At each step:
    1. Standard score network denoising step
    2. Compute constraint violation via differentiable checker
    3. Gradient of violation w.r.t. current latent
    4. Steer latent toward lower violation
    """
    score_net.eval()
    l_t = z_unint_noisy_norm.clone()
    timesteps = torch.linspace(start_t, 0, num_steps).long()

    for t_val in timesteps:
        t = t_val.expand(l_t.shape[0]).to(device)
        alpha_bar_t = diffusion.alpha_bars[t_val].to(device)
        alpha_bar_prev = diffusion.alpha_bars[t_val-1].to(device) if t_val > 0 else torch.tensor(1.0).to(device)

        # Step 1 — Standard score network denoising
        with torch.no_grad():
            predicted_noise = score_net(l_t, t, z_fixed)
            l_0_pred = (l_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            if t_val > 0:
                noise = torch.randn_like(l_t)
                sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                l_t_prev = torch.sqrt(alpha_bar_prev) * l_0_pred + \
                           torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * predicted_noise + \
                           sigma_t * noise
            else:
                l_t_prev = l_0_pred

        # Step 2 — Constraint guidance
        l_t_prev = l_t_prev.detach().requires_grad_(True)

        # Unnormalize → decode → differentiable constraint check
        z_denoised = l_t_prev * z_unint_std_t + z_unint_mean_t
        z_combined = torch.cat([z_fixed, z_denoised], dim=1)
        img_decoded = vae.decode(z_combined)

        constraint_loss, x_err, y_err = differentiable_constraint_loss(
            img_decoded, true_x, true_y, eps_x=0.05, eps_y=0.10
        )

        # Step 3 — Gradient of violation w.r.t. current latent
        if constraint_loss > 0:
            grad = torch.autograd.grad(constraint_loss, l_t_prev)[0]
            # Step 4 — Steer toward lower violation
            l_t = (l_t_prev - guidance_scale * grad).detach()
        else:
            l_t = l_t_prev.detach()

    return l_t

# Parameters
eps_x = 0.05
eps_y = 0.10
eps_theta = 0.50
t_level = 20
max_iterations = 10
guidance_scale = 1.0  # constraint guidance strength
BAD_X_THRESH = 0.20
BAD_Y_THRESH = 0.30

results = []
skipped = 0
visible_indices = [i for i in range(len(test_states)) if test_states[i,1] < 1.3]
print(f"\nRunning Solution #1 (guided inference) on {len(visible_indices)} visible frames...")
print(f"guidance_scale={guidance_scale}, t_level={t_level}, max_iterations={max_iterations}")

for idx in tqdm(visible_indices):
    true_x, true_y, true_theta = test_states[idx,0], test_states[idx,1], test_states[idx,2]
    true_x_t = torch.tensor([true_x], dtype=torch.float32, device=device)
    true_y_t = torch.tensor([true_y], dtype=torch.float32, device=device)

    # Encode using 96x128 image
    img_96 = test_imgs_96[idx].astype(np.float32) / 255.0
    img_tensor = torch.FloatTensor(img_96).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        h = vae.encoder(img_tensor)
        h = h.reshape(h.size(0), -1)
        mu = vae.fc_mu(h)

    z_fixed = mu[:, :3]   # FROZEN
    z_unint = mu[:, 3:]   # diffusion acts here

    # Check initial quality
    with torch.no_grad():
        z_combined = torch.cat([z_fixed, z_unint], dim=1)
        recon = vae.decode(z_combined)
    recon_img = recon[0].permute(1,2,0).cpu().numpy()

    result_init = constraint_checker(recon_img, true_x, true_y, true_theta)
    if not result_init['visible']:
        skipped += 1
        continue
    if result_init['x_err'] > BAD_X_THRESH or result_init['y_err'] > BAD_Y_THRESH:
        skipped += 1
        continue

    frame_history = []

    for iteration in range(max_iterations + 1):
        with torch.no_grad():
            z_combined = torch.cat([z_fixed, z_unint], dim=1)
            recon = vae.decode(z_combined)
        recon_img = recon[0].permute(1,2,0).cpu().numpy()

        result = constraint_checker(recon_img, true_x, true_y, true_theta)

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

        # Guided reverse diffusion
        z_unint_norm = (z_unint - z_unint_mean_t) / z_unint_std_t
        t_batch = torch.tensor([t_level]).to(device)
        l_noisy_norm, _ = diffusion.add_noise(z_unint_norm, t_batch)

        l_corr_norm = reverse_diffusion_guided(
            score_net, l_noisy_norm, z_fixed,
            diffusion, vae, z_unint_mean_t, z_unint_std_t,
            true_x_t, true_y_t,
            start_t=t_level, num_steps=100,
            guidance_scale=guidance_scale
        )
        z_unint = l_corr_norm * z_unint_std_t + z_unint_mean_t

    results.append({'idx': idx, 'history': frame_history})

print(f"\nSkipped: {skipped}, Evaluated: {len(results)}")
np.save('/blue/iruchkin/aj.byreddy/tea_lab/results_solution1_intrinsic_guided_v4.npy', results)
print("✅ Results saved!")

print("\n" + "="*65)
print("SOLUTION #1 GUIDED v4 — Results")
print("="*65)
for it in range(max_iterations + 1):
    x_errs = [r['history'][it]['x_err'] for r in results if r['history'][it]['x_err'] is not None]
    y_errs = [r['history'][it]['y_err'] for r in results if r['history'][it]['y_err'] is not None]
    theta_errs = [r['history'][it]['theta_err'] for r in results if r['history'][it]['theta_err'] is not None]
    xy_ok = sum(r['history'][it]['xy_ok'] for r in results)
    all_ok = sum(r['history'][it]['all_ok'] for r in results)
    print(f"\nIteration {it} ({len(results)} frames):")
    if x_errs: print(f"  x error: mean={np.mean(x_errs):.4f} median={np.median(x_errs):.4f}")
    if y_errs: print(f"  y error: mean={np.mean(y_errs):.4f} median={np.median(y_errs):.4f}")
    if theta_errs: print(f"  theta error: mean={np.mean(theta_errs):.4f} median={np.median(theta_errs):.4f}")
    print(f"  x,y satisfied: {xy_ok}/{len(results)} ({100*xy_ok/len(results):.1f}%)")
    print(f"  all satisfied: {all_ok}/{len(results)} ({100*all_ok/len(results):.1f}%)")
