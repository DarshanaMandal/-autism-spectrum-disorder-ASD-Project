"""
MultiSense GAN: Conditional Pose Sequence GAN Trainer
Trains a conditional recurrent generator and discriminator using Least Squares GAN (LSGAN)
loss for stable convergence without mode collapse.
Trained STRICTLY on the training split to uphold zero data leakage.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloader
from generator import ConditionalPoseGenerator
from discriminator import ConditionalPoseDiscriminator


def train_conditional_gan(data_dir, models_dir, epochs=20, batch_size=64, lr=0.0003, latent_dim=64, device="cpu"):
    device = torch.device(device)
    print(f"[*] Training Conditional Pose Sequence GAN on device: {device}")

    # Load real training data ONLY
    X_train = np.load(os.path.join(data_dir, "X_train_real.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train_real.npy"))

    num_samples, seq_len, feature_dim = X_train.shape
    num_classes = 3

    print(f"    - Training data shape: {X_train.shape}")
    print(f"    - Sequence length: {seq_len}, Features: {feature_dim}, Classes: {num_classes}")

    dataloader = get_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)

    # Instantiate models
    generator = ConditionalPoseGenerator(
        latent_dim=latent_dim,
        num_classes=num_classes,
        feature_dim=feature_dim,
        seq_len=seq_len,
        hidden_dim=128
    ).to(device)

    discriminator = ConditionalPoseDiscriminator(
        num_classes=num_classes,
        feature_dim=feature_dim,
        seq_len=seq_len,
        hidden_dim=128
    ).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.MSELoss()

    os.makedirs(models_dir, exist_ok=True)

    print(f"[*] Commencing GAN training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        total_d_loss = 0.0
        total_g_loss = 0.0
        batches = 0

        generator.train()
        discriminator.train()

        for real_x, real_labels, real_onehot in dataloader:
            B = real_x.size(0)
            real_x = real_x.to(device)
            real_onehot = real_onehot.to(device)

            real_target = torch.ones((B, 1), device=device)
            fake_target = torch.zeros((B, 1), device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            d_real_out = discriminator(real_x, real_onehot)
            d_real_loss = criterion(d_real_out, real_target)

            # Generate synthetic batch with matching labels
            z = torch.randn(B, latent_dim, device=device)
            fake_x = generator(z, real_onehot)

            d_fake_out = discriminator(fake_x.detach(), real_onehot)
            d_fake_loss = criterion(d_fake_out, fake_target)

            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            g_fake_out = discriminator(fake_x, real_onehot)
            g_loss = criterion(g_fake_out, real_target)

            g_loss.backward()
            optimizer_G.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            batches += 1

        avg_d_loss = total_d_loss / max(1, batches)
        avg_g_loss = total_g_loss / max(1, batches)

        print(f"    Epoch [{epoch:02d}/{epochs:02d}] | D_Loss: {avg_d_loss:.4f} | G_Loss: {avg_g_loss:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(models_dir, "gan_checkpoint.pth")
    torch.save({
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "latent_dim": latent_dim,
        "feature_dim": feature_dim,
        "seq_len": seq_len,
        "num_classes": num_classes
    }, checkpoint_path)

    print("\n" + "="*60)
    print(f"[✓] GAN Training Complete! Checkpoint saved to: {checkpoint_path}")
    print("="*60 + "\n")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    seq_dir = base_dir / "data" / "sequences"
    models_dir = base_dir / "models"
    train_conditional_gan(str(seq_dir), str(models_dir), epochs=args.epochs, batch_size=args.batch_size, device=args.device)
