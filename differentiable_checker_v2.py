"""
Differentiable constraint checker v2 — x, y, AND theta.
Uses:
- Soft sigmoid mask for x,y centroid (differentiable)
- Adnan's ThetaBranch CNN for theta (differentiable)
- grid_sample for differentiable crop extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Calibration constants
CALIB_X_SLOPE = 0.01329
CALIB_X_INTERCEPT = -0.9906
CALIB_Y_SLOPE = -0.02001
CALIB_Y_INTERCEPT = 1.3958
TEMPERATURE = 0.01
CROP = 24

def soft_purple_mask(img_tensor):
    """
    Differentiable soft purple mask.
    img_tensor: (B, 3, H, W) float in [0,1]
    returns: (B, H, W) soft mask
    """
    r = img_tensor[:, 0]
    g = img_tensor[:, 1]
    b = img_tensor[:, 2]
    PURPLE_BIAS = 0.051
    PURPLE_MIN = 0.10
    cond1 = torch.sigmoid((b - r - PURPLE_BIAS) / TEMPERATURE)
    cond2 = torch.sigmoid((b - g - PURPLE_BIAS) / TEMPERATURE)
    cond3 = torch.sigmoid((b - PURPLE_MIN) / TEMPERATURE)
    return cond1 * cond2 * cond3


def soft_centroid(mask):
    """
    Differentiable centroid from soft mask.
    mask: (B, H, W)
    returns: cx (B,), cy (B,) in pixel coordinates
    """
    B, H, W = mask.shape
    device = mask.device
    x_coords = torch.arange(W, dtype=torch.float32, device=device).view(1,1,W).expand(B,H,W)
    y_coords = torch.arange(H, dtype=torch.float32, device=device).view(1,H,1).expand(B,H,W)
    total = mask.sum(dim=[1,2]) + 1e-8
    cx = (mask * x_coords).sum(dim=[1,2]) / total
    cy = (mask * y_coords).sum(dim=[1,2]) / total
    return cx, cy


def differentiable_crop(img, cx, cy, crop_size=24):
    """
    Differentiable crop centered on (cx, cy) using grid_sample.
    img: (B, 3, H, W)
    cx, cy: (B,) pixel coordinates
    returns: (B, 3, crop_size, crop_size)
    """
    B, C, H, W = img.shape
    device = img.device

    # Normalize cx, cy to [-1, 1] for grid_sample
    cx_norm = (cx / (W - 1)) * 2 - 1  # (B,)
    cy_norm = (cy / (H - 1)) * 2 - 1  # (B,)

    # Create sampling grid for crop_size × crop_size patch
    half = crop_size / 2
    xs = torch.linspace(-half, half, crop_size, device=device) / (W / 2)
    ys = torch.linspace(-half, half, crop_size, device=device) / (H / 2)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (crop, crop)

    # Offset by centroid position
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1) + cx_norm.view(B, 1, 1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1) + cy_norm.view(B, 1, 1)

    # Stack into grid (B, crop, crop, 2)
    grid = torch.stack([grid_x, grid_y], dim=-1)

    # Sample
    crop = F.grid_sample(img, grid, align_corners=True, padding_mode='border')
    return crop  # (B, 3, crop_size, crop_size)


def differentiable_constraint_loss_v2(img_tensor, x_orig, y_orig, theta_orig,
                                       branch, eps_x=0.02, eps_y=0.01, eps_theta=0.30):
    """
    Differentiable constraint loss for x, y, AND theta.

    img_tensor: (B, 3, H, W) decoded image
    x_orig, y_orig: (B,) physical coords from original image R(a_i)
    theta_orig: (B,) true theta
    branch: Adnan's ThetaBranch CNN (differentiable)
    """
    # Soft mask → centroid
    mask = soft_purple_mask(img_tensor)
    cx, cy = soft_centroid(mask)

    # Physical x, y
    x_hat = CALIB_X_SLOPE * cx + CALIB_X_INTERCEPT
    y_hat = CALIB_Y_SLOPE * cy + CALIB_Y_INTERCEPT

    # x, y violations (hinge loss)
    x_err = torch.abs(x_hat - x_orig)
    y_err = torch.abs(y_hat - y_orig)
    x_violation = F.relu(x_err - eps_x) ** 2
    y_violation = F.relu(y_err - eps_y) ** 2

    # Differentiable crop → ThetaBranch → (cosθ, sinθ)
    crop = differentiable_crop(img_tensor, cx, cy, crop_size=CROP)
    cos_sin = branch(crop)  # (B, 2)
    cos_hat = cos_sin[:, 0]
    sin_hat = cos_sin[:, 1]

    # θ from cos,sin
    theta_hat = torch.atan2(sin_hat, cos_hat)  # (B,)

    # Circular θ error
    theta_err = torch.abs(torch.atan2(
        torch.sin(theta_hat - theta_orig),
        torch.cos(theta_hat - theta_orig)
    ))
    theta_violation = F.relu(theta_err - eps_theta) ** 2

    # Combined loss
    constraint_loss = (x_violation + y_violation + theta_violation).mean()

    return constraint_loss, x_err.mean().item(), y_err.mean().item(), theta_err.mean().item()


def test_v2():
    """Test differentiable checker v2 with theta."""
    import sys
    sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/adnan_vae')
    import config
    from zlander_recon_fig import load

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")

    # Load Adnan's branch
    m = load("factored_clean_noaug_best", device)
    branch = m['branch']
    branch.eval()

    # Load test image
    imgs = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_imgs_100x150.npy')
    states = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_states_xy_angle.npy')
    good = [i for i in range(len(states)) if states[i,1] < 1.0]
    idx = good[10]

    img = imgs[idx].astype(np.float32) / 255.0
    img_tensor = torch.FloatTensor(img).permute(2,0,1).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    x_orig = torch.tensor([states[idx,0]], dtype=torch.float32, device=device)
    y_orig = torch.tensor([states[idx,1]], dtype=torch.float32, device=device)
    theta_orig = torch.tensor([states[idx,2]], dtype=torch.float32, device=device)

    loss, x_err, y_err, theta_err = differentiable_constraint_loss_v2(
        img_tensor, x_orig, y_orig, theta_orig, branch,
        eps_x=0.02, eps_y=0.01, eps_theta=0.30
    )

    print(f"Constraint loss: {loss.item():.6f}")
    print(f"x error: {x_err:.4f}, y error: {y_err:.4f}, theta error: {theta_err:.4f}")

    loss.backward()
    if img_tensor.grad is not None:
        print(f"✅ Gradients flow! Grad norm: {img_tensor.grad.norm().item():.6f}")
    else:
        print("❌ No gradients!")

if __name__ == "__main__":
    test_v2()
