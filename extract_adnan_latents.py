import sys
import torch
import numpy as np
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/adnan_vae')
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/adnan_checker')
import config
import checkpoints
from zlander_recon_fig import load, encode_frame
from checker import purple_mask, largest_component, TiltReader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Adnan's VAE
m = load("factored_clean_noaug_best", device)
print("✅ Adnan VAE loaded!")

# Load data
train_imgs = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/train_imgs.npy')
train_states = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/train_states.npy')
test_imgs = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_imgs_100x150.npy')
test_states = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_states_xy_angle.npy')

print(f"Train: {train_imgs.shape}, Test: {test_imgs.shape}")

# Calibrate tilt reader on training data
print("\nCalibrating tilt reader...")
cal_masks, cal_thetas = [], []
for i in range(0, len(train_imgs), 10):
    mask = largest_component(purple_mask(train_imgs[i]))
    if mask is not None and mask.sum() >= 8:
        cal_masks.append(mask)
        cal_thetas.append(train_states[i, 2])
tilt_reader = TiltReader()
tilt_reader.calibrate(cal_masks, cal_thetas)
print(f"Calibrated! Floor error: {tilt_reader.floor_deg:.2f}°")

def extract_latents(imgs, states, desc=""):
    """Extract z_pose (4-dim) and z_scene (28-dim) from images."""
    z_poses = []
    z_scenes = []
    failed = 0
    
    for i in range(len(imgs)):
        img = imgs[i].astype(np.float32) / 255.0
        img_tensor = torch.FloatTensor(img).permute(2,0,1).unsqueeze(0)
        
        try:
            with torch.no_grad():
                z = encode_frame(m, img_tensor)
            z_poses.append(z[0, :4].cpu().numpy())   # x, y, cos θ, sin θ
            z_scenes.append(z[0, 4:].cpu().numpy())  # scene code (28-dim)
        except Exception as e:
            # Lander not visible — use zeros as placeholder
            z_poses.append(np.zeros(4))
            z_scenes.append(np.zeros(28))
            failed += 1
        
        if i % 5000 == 0:
            print(f"  {desc}: {i}/{len(imgs)} done... (failed: {failed})")
    
    return np.array(z_poses), np.array(z_scenes), failed

# Extract training latents
print("\nExtracting training latents...")
train_z_poses, train_z_scenes, train_failed = extract_latents(train_imgs, train_states, "train")
print(f"Train: z_pose={train_z_poses.shape}, z_scene={train_z_scenes.shape}, failed={train_failed}")

np.save('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/adnan_z_pose_train.npy', train_z_poses)
np.save('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/adnan_z_scene_train.npy', train_z_scenes)
print("✅ Saved training latents!")

# Extract test latents
print("\nExtracting test latents...")
test_z_poses, test_z_scenes, test_failed = extract_latents(test_imgs, test_states, "test")
print(f"Test: z_pose={test_z_poses.shape}, z_scene={test_z_scenes.shape}, failed={test_failed}")

np.save('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/adnan_z_pose_test.npy', test_z_poses)
np.save('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/adnan_z_scene_test.npy', test_z_scenes)
print("✅ Saved test latents!")

# Verify pose accuracy
print("\nVerifying z_pose accuracy:")
# Only check frames where encoding succeeded (z_pose not zeros)
valid = np.any(train_z_poses != 0, axis=1)
print(f"Valid train frames: {valid.sum()}/{len(train_z_poses)}")
for dim, name in enumerate(['x', 'y']):
    corr = np.corrcoef(train_z_poses[valid, dim], train_states[valid, dim])[0,1]
    print(f"  z_pose[{dim}] vs true {name}: correlation = {corr:.4f}")
