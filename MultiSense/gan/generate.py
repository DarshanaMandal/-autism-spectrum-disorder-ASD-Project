"""
MultiSense GAN: Synthetic Pose Generator
Uses the trained Conditional Pose Generator to synthesize balanced,
high-fidelity synthetic pose sequences for training data augmentation.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch

from generator import ConditionalPoseGenerator


def generate_synthetic_samples(checkpoint_path, synthetic_dir, count_per_class=None, device="cpu"):
    if count_per_class is None:
        # Default augmentation counts to balance minority classes
        count_per_class = {
            0: 500,  # ArmFlapping
            1: 500,  # HeadBanging
            2: 200   # Spinning
        }

    os.makedirs(synthetic_dir, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator = ConditionalPoseGenerator(
        latent_dim=checkpoint["latent_dim"],
        num_classes=checkpoint["num_classes"],
        feature_dim=checkpoint["feature_dim"],
        seq_len=checkpoint["seq_len"]
    )
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.to(device)
    generator.eval()

    synth_X_list = []
    synth_y_list = []

    class_names = {0: "ArmFlapping", 1: "HeadBanging", 2: "Spinning"}
    print(f"[*] Generating synthetic pose sequences using trained GAN...")

    for class_id, n_samples in count_per_class.items():
        if n_samples <= 0:
            continue
        c_name = class_names.get(class_id, f"Class_{class_id}")
        print(f"    - Generating {n_samples} synthetic sequences for {c_name}...")
        
        samples = generator.sample(n_samples, class_id=class_id, device=device)
        synth_X_list.append(samples)
        synth_y_list.extend([class_id] * n_samples)

    X_synthetic = np.concatenate(synth_X_list, axis=0).astype(np.float32)
    y_synthetic = np.array(synth_y_list, dtype=np.int64)

    # Save purely synthetic dataset
    np.save(os.path.join(synthetic_dir, "X_train_synthetic.npy"), X_synthetic)
    np.save(os.path.join(synthetic_dir, "y_train_synthetic.npy"), y_synthetic)

    # Also build and save combined augmented training set (Real + Synthetic)
    base_dir = Path(synthetic_dir).parent
    seq_dir = os.path.join(base_dir, "sequences")
    X_train_real = np.load(os.path.join(seq_dir, "X_train_real.npy"))
    y_train_real = np.load(os.path.join(seq_dir, "y_train_real.npy"))

    X_train_augmented = np.concatenate([X_train_real, X_synthetic], axis=0)
    y_train_augmented = np.concatenate([y_train_real, y_synthetic], axis=0)

    # Shuffle augmented dataset
    perm = np.random.permutation(len(X_train_augmented))
    X_train_augmented = X_train_augmented[perm]
    y_train_augmented = y_train_augmented[perm]

    np.save(os.path.join(seq_dir, "X_train_augmented.npy"), X_train_augmented)
    np.save(os.path.join(seq_dir, "y_train_augmented.npy"), y_train_augmented)

    print("\n" + "="*60)
    print(f"[✓] Synthetic Generation Complete!")
    print(f"    - Pure Synthetic shape: {X_synthetic.shape}")
    print(f"    - Combined Augmented shape (Real + GAN): {X_train_augmented.shape}")
    print("="*60 + "\n")
    return X_synthetic, y_synthetic


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    checkpoint = base_dir / "models" / "gan_checkpoint.pth"
    synth_dir = base_dir / "data" / "synthetic"
    generate_synthetic_samples(str(checkpoint), str(synth_dir))
