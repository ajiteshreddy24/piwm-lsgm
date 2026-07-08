import numpy as np
from PIL import Image
import os

print("Loading data...")
train_imgs = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/train_imgs.npy')
train_states = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/train_states.npy')

print(f"Train images: {train_imgs.shape}")
print(f"Train states: {train_states.shape}")

# Resize images to 96x128 (divisible by 32, works with 5 stride-2 conv layers)
TARGET_H, TARGET_W = 96, 128

print(f"Resizing {len(train_imgs)} training images to {TARGET_H}x{TARGET_W}...")
train_imgs_resized = np.zeros((len(train_imgs), TARGET_H, TARGET_W, 3), dtype=np.uint8)
for i in range(len(train_imgs)):
    pil_img = Image.fromarray(train_imgs[i])
    pil_img = pil_img.resize((TARGET_W, TARGET_H))  # PIL uses (W, H)
    train_imgs_resized[i] = np.array(pil_img)
    if i % 5000 == 0:
        print(f"  {i}/{len(train_imgs)} done...")

print(f"Resized train images: {train_imgs_resized.shape}")

# Save training NPZ with x, y, theta (all 3 state dims)
np.savez('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_train_intrinsic.npz',
         frame=train_imgs_resized,
         state=train_states)  # already (30776, 3) — x, y, theta
print("✅ Saved lunar_train_intrinsic.npz")

# Do the same for test data
print("\nLoading test data...")
test_imgs = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_imgs_100x150.npy')
test_states = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_states_xy_angle.npy')

print(f"Test images: {test_imgs.shape}")
print(f"Test states: {test_states.shape}")

print(f"Resizing {len(test_imgs)} test images to {TARGET_H}x{TARGET_W}...")
test_imgs_resized = np.zeros((len(test_imgs), TARGET_H, TARGET_W, 3), dtype=np.uint8)
for i in range(len(test_imgs)):
    pil_img = Image.fromarray(test_imgs[i])
    pil_img = pil_img.resize((TARGET_W, TARGET_H))
    test_imgs_resized[i] = np.array(pil_img)

np.savez('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_test_intrinsic.npz',
         frame=test_imgs_resized,
         state=test_states)
print("✅ Saved lunar_test_intrinsic.npz")

print("\nDone! Data ready for intrinsic VAE training.")
print(f"Train: {train_imgs_resized.shape}, states: {train_states.shape}")
print(f"Test:  {test_imgs_resized.shape}, states: {test_states.shape}")
