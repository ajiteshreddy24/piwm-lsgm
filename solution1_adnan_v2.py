import sys
import torch
import numpy as np
from tqdm import tqdm
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab')
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/LSGM')
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/adnan_vae')
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/adnan_checker')
import config
import checkpoints
from zlander_recon_fig import load, encode_frame
from checker import purple_mask, largest_component, TiltReader
from constraint_checker import constraint_checker  # our checker for physical units
from score_sde.ncsnpp_linear import NCSNppLinear
from setup import DiffusionProcess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Adnan's VAE
m = load("factored_clean_noaug_best", device)
print("✅ Adnan VAE loaded!")

# Load score network
score_net = NCSNppLinear(
    latent_dim=28, physical_dim=4,
    nf=128, ch_mult=(1,2,2),
    num_res_blocks=2, temb_dim=128
).to(device)
ckpt = torch.load('/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/score_net_adnan/best.pt',
                  map_location=device, weights_only=False)
score_net.load_state_dict(ckpt['score_net'])
score_net.eval()
print(f"✅ Score network loaded! Epoch {ckpt['epoch']}, Loss {ckpt['loss']:.6f}")

# Load normalization stats
z_scene_mean = np.load('/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/score_net_adnan/z_scene_mean.npy')
z_scene_std = np.load('/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/score_net_adnan/z_scene_std.npy')
z_scene_mean_t = torch.FloatTensor(z_scene_mean).to(device)
z_scene_std_t = torch.FloatTensor(z_scene_std).to(device)

diffusion = DiffusionProcess(T=1000, device=device)

# Load test data
test_imgs = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_imgs_100x150.npy')
test_states = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_states_xy_angle.npy')

def reverse_diffusion_scene(score_net, z_scene_noisy_norm, z_pose, diffusion, start_t=20, num_steps=100):
    score_net.eval()
    l_t = z_scene_noisy_norm.clone()
    timesteps = torch.linspace(start_t, 0, num_steps).long()
    with torch.no_grad():
        for t_val in timesteps:
            t = t_val.expand(l_t.shape[0]).to(device)
            alpha_bar_t = diffusion.alpha_bars[t_val].to(device)
            alpha_bar_prev = diffusion.alpha_bars[t_val-1].to(device) if t_val > 0 else torch.tensor(1.0).to(device)
            predicted_noise = score_net(l_t, t, z_pose)
            l_0_pred = (l_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            if t_val > 0:
                noise = torch.randn_like(l_t)
                sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                l_t = torch.sqrt(alpha_bar_prev) * l_0_pred + torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * predicted_noise + sigma_t * noise
            else:
                l_t = l_0_pred
    return l_t

# Parameters
eps_x = 0.05
eps_y = 0.10
eps_theta = 0.50
t_level = 20
max_iterations = 10
BAD_X_THRESH = 0.20
BAD_Y_THRESH = 0.30

results = []
skipped = 0
visible_indices = [i for i in range(len(test_imgs)) if test_states[i,1] < 1.3]
print(f"\nRunning Solution #1 (Adnan VAE + our checker) on {len(visible_indices)} visible frames...")

for idx in tqdm(visible_indices):
    img = test_imgs[idx]
    true_x, true_y, true_theta = test_states[idx,0], test_states[idx,1], test_states[idx,2]

    # Encode image → z_pose + z_scene
    img_tensor = torch.FloatTensor(img.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    try:
        with torch.no_grad():
            z = encode_frame(m, img_tensor)
        z_pose = z[:, :4]   # FROZEN — x, y, cosθ, sinθ
        z_scene = z[:, 4:]  # diffusion acts here
    except:
        skipped += 1
        continue

    # Check initial quality using our constraint checker
    with torch.no_grad():
        z_combined = torch.cat([z_pose, z_scene], dim=1)
        recon = m['vae'].decode(z_combined).clamp(0,1)
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
        # Decode current latent
        with torch.no_grad():
            z_combined = torch.cat([z_pose, z_scene], dim=1)
            recon = m['vae'].decode(z_combined).clamp(0,1)
        recon_img = recon[0].permute(1,2,0).cpu().numpy()

        # Constraint check using our checker (physical units)
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

        # Diffusion on z_scene only — z_pose stays FROZEN
        z_scene_norm = (z_scene - z_scene_mean_t) / z_scene_std_t
        t_batch = torch.tensor([t_level]).to(device)
        l_noisy_norm, _ = diffusion.add_noise(z_scene_norm, t_batch)
        l_corr_norm = reverse_diffusion_scene(
            score_net, l_noisy_norm, z_pose, diffusion,
            start_t=t_level, num_steps=100
        )
        z_scene = l_corr_norm * z_scene_std_t + z_scene_mean_t

    results.append({'idx': idx, 'history': frame_history})

print(f"\nSkipped: {skipped}, Evaluated: {len(results)}")
np.save('/blue/iruchkin/aj.byreddy/tea_lab/results_solution1_adnan_v2.npy', results)
print("✅ Results saved!")

print("\n" + "="*65)
print("SOLUTION #1 ADNAN v2 — Results")
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
