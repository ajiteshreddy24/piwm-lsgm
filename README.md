# PIWM LSGM Constraint Correction

Latent diffusion-based correction mechanism for Physically Interpretable World Models (PIWM) on Lunar Lander.

## Files

### Constraint Checkers
- `constraint_checker.py` — extracts x, y, θ from decoded images using color segmentation, centroid, and PCA
- `differentiable_checker.py` — differentiable version for x, y using soft sigmoid mask (for training)
- `differentiable_checker_v2.py` — differentiable version for x, y, and θ using soft mask + Adnan's ThetaBranch CNN

### Core
- `setup.py` — diffusion process (forward noise addition, reverse denoising)

### Score Network Training
- `train_score_normalized.py` — trains score network on Mrinall's VAE latents (no constraint guidance)
- `train_score_intrinsic_guided.py` — trains score network on Zhenjiang's VAE latents with constraint penalty in loss
- `train_score_adnan.py` — trains score network on Adnan's VAE latents (z_scene, conditioned on z_pose)

### Solution #1 Loop (Latent Correction)
- `smart_solution1_v5.py` — Mrinall VAE, 20 iterations, full f_i conditioning
- `solution1_intrinsic_200ep.py` — Zhenjiang 200ep VAE, no guidance
- `solution1_intrinsic_guided_v2.py` — Zhenjiang 200ep VAE, x,y guidance (scale=0.1)
- `solution1_intrinsic_guided_v4.py` — Zhenjiang 200ep VAE, x,y guidance (scale=1.0)
- `solution1_intrinsic_guided_v6.py` — Zhenjiang 200ep VAE, x,y,θ guidance (scale=1.0)
- `solution1_adnan_v2.py` — Adnan's disentangled VAE, z_pose frozen, diffusion on z_scene

### Data Preparation
- `prepare_intrinsic_data.py` — resizes images from 100x150 to 96x128 for Zhenjiang's VAE
- `extract_intrinsic_latents_200ep.py` — extracts z_fixed and z_unint from Zhenjiang's 200ep VAE
- `extract_adnan_latents.py` — extracts z_pose and z_scene from Adnan's VAE
