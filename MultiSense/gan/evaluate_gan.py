"""
MultiSense GAN Evaluation: Distribution Comparison & Kinematic Validation
Validates synthetic pose sequences against real training sequences:
1. Compares mean, variance, and kinematic velocities (wrists, head, torso).
2. Computes Wasserstein / Earth Mover's Distance and cosine similarity.
3. Exports a publication-ready 4-panel figure: results/real_vs_synthetic.png.
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance


def compute_kinematic_distributions(sequences):
    """
    Args:
        sequences: (N, 60, 132)
    Returns:
        wrist_vels: (N,) average wrist velocities
        head_vels: (N,) average head velocities
        body_disps: (N,) total torso motion
        variances: (N,) overall sequence variances
    """
    N, T, D = sequences.shape
    # Reshape to (N, T, 33, 4)
    seq = sequences.reshape(N, T, 33, 4)
    
    # Left wrist: 15, Right wrist: 16
    # Head: 0
    # Torso: (hip 23 + hip 24) / 2
    wrist_diff = np.diff(seq[:, :, [15, 16], :3], axis=1)  # (N, T-1, 2, 3)
    wrist_vel = np.mean(np.linalg.norm(wrist_diff, axis=-1), axis=(1, 2))  # (N,)

    head_diff = np.diff(seq[:, :, 0, :3], axis=1)  # (N, T-1, 3)
    head_vel = np.mean(np.linalg.norm(head_diff, axis=-1), axis=1)  # (N,)

    torso = (seq[:, :, 23, :3] + seq[:, :, 24, :3]) / 2.0
    body_disp = np.sum(np.linalg.norm(np.diff(torso, axis=1), axis=-1), axis=1)  # (N,)

    variances = np.mean(np.var(sequences, axis=1), axis=1)  # (N,)

    return wrist_vel, head_vel, body_disp, variances


def evaluate_synthetic_data(real_seq_path, synth_seq_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    X_real = np.load(real_seq_path)
    X_synth = np.load(synth_seq_path)

    print(f"[*] Comparing {len(X_real)} Real sequences vs {len(X_synth)} Synthetic sequences...")

    r_wrist, r_head, r_body, r_var = compute_kinematic_distributions(X_real)
    s_wrist, s_head, s_body, s_var = compute_kinematic_distributions(X_synth)

    # Statistical distances
    wd_wrist = float(wasserstein_distance(r_wrist, s_wrist))
    wd_head = float(wasserstein_distance(r_head, s_head))
    wd_body = float(wasserstein_distance(r_body, s_body))
    wd_var = float(wasserstein_distance(r_var, s_var))

    metrics = {
        "wrist_velocity": {
            "real_mean": float(np.mean(r_wrist)),
            "synth_mean": float(np.mean(s_wrist)),
            "real_std": float(np.std(r_wrist)),
            "synth_std": float(np.std(s_wrist)),
            "wasserstein_distance": wd_wrist
        },
        "head_velocity": {
            "real_mean": float(np.mean(r_head)),
            "synth_mean": float(np.mean(s_head)),
            "real_std": float(np.std(r_head)),
            "synth_std": float(np.std(s_head)),
            "wasserstein_distance": wd_head
        },
        "body_displacement": {
            "real_mean": float(np.mean(r_body)),
            "synth_mean": float(np.mean(s_body)),
            "wasserstein_distance": wd_body
        },
        "sequence_variance": {
            "real_mean": float(np.mean(r_var)),
            "synth_mean": float(np.mean(s_var)),
            "wasserstein_distance": wd_var
        }
    }

    metrics_json_path = os.path.join(results_dir, "gan_evaluation_metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # 4-Panel Visualization Plot
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Wrist Velocity Distribution
    ax = axes[0, 0]
    sns.kdeplot(r_wrist, ax=ax, label="Real Pose Data", color="#1f77b4", fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(s_wrist, ax=ax, label="Synthetic GAN Data", color="#ff7f0e", fill=True, alpha=0.35, linewidth=2)
    ax.set_title("Wrist Velocity Distribution (Arm Flapping Dynamics)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mean Velocity (norm units / frame)")
    ax.legend(frameon=True)

    # 2. Head Velocity Distribution
    ax = axes[0, 1]
    sns.kdeplot(r_head, ax=ax, label="Real Pose Data", color="#2ca02c", fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(s_head, ax=ax, label="Synthetic GAN Data", color="#d62728", fill=True, alpha=0.35, linewidth=2)
    ax.set_title("Head Velocity Distribution (Head Banging Dynamics)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mean Velocity (norm units / frame)")
    ax.legend(frameon=True)

    # 3. Body Displacement
    ax = axes[1, 0]
    sns.kdeplot(r_body, ax=ax, label="Real Pose Data", color="#9467bd", fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(s_body, ax=ax, label="Synthetic GAN Data", color="#8c564b", fill=True, alpha=0.35, linewidth=2)
    ax.set_title("Body Center Displacement (Spinning Dynamics)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Cumulative Torso Path Length")
    ax.legend(frameon=True)

    # 4. Pose Sequence Variance
    ax = axes[1, 1]
    sns.kdeplot(r_var, ax=ax, label="Real Pose Data", color="#17becf", fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(s_var, ax=ax, label="Synthetic GAN Data", color="#bcbd22", fill=True, alpha=0.35, linewidth=2)
    ax.set_title("Overall Kinematic Variance Distribution", fontsize=13, fontweight='bold')
    ax.set_xlabel("Sequence Variance")
    ax.legend(frameon=True)

    plt.suptitle("MultiSense GAN Validation: Real vs. Synthetic Biomechanical Kinematics", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "real_vs_synthetic.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("\n" + "="*60)
    print(f"[✓] GAN Evaluation Complete!")
    print(f"    - Wrist Velocity Wasserstein Distance: {wd_wrist:.4f}")
    print(f"    - Head Velocity Wasserstein Distance:  {wd_head:.4f}")
    print(f"    - Figure saved: {fig_path}")
    print(f"    - Metrics saved: {metrics_json_path}")
    print("="*60 + "\n")
    return metrics


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    real_path = base_dir / "data" / "sequences" / "X_train_real.npy"
    synth_path = base_dir / "data" / "synthetic" / "X_train_synthetic.npy"
    res_dir = base_dir / "results"
    evaluate_synthetic_data(str(real_path), str(synth_path), str(res_dir))
