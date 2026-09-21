"""
MultiSense Preprocessing: Skeletal Pose Normalization
Applies anatomical body-centric normalization:
1. Centering: Subtraction of hip midpoint (landmarks 23 & 24) to eliminate absolute camera translation.
2. Scaling: Normalization by torso scale (shoulder width / torso height) to achieve scale and distance invariance.
3. Preserves confidence/visibility channels in [0, 1].
"""

import os
import glob
import numpy as np
from pathlib import Path


def normalize_single_sequence(pose_seq):
    """
    Args:
        pose_seq: numpy array of shape (T, 132) or (T, 33, 4)
    Returns:
        normalized_seq: numpy array of same shape
    """
    is_flat = (pose_seq.ndim == 2 and pose_seq.shape[1] == 132)
    if is_flat:
        T = pose_seq.shape[0]
        seq = pose_seq.reshape(T, 33, 4).copy()
    else:
        seq = pose_seq.copy()
        T = seq.shape[0]

    # MediaPipe landmark indices:
    # 11: left_shoulder, 12: right_shoulder
    # 23: left_hip, 24: right_hip
    for t in range(T):
        frame = seq[t]  # (33, 4) -> x, y, z, visibility
        left_hip = frame[23, :3]
        right_hip = frame[24, :3]
        left_shoulder = frame[11, :3]
        right_shoulder = frame[12, :3]

        # Torso center as midpoint between hips
        hip_center = (left_hip + right_hip) / 2.0
        shoulder_center = (left_shoulder + right_shoulder) / 2.0

        # Scale factor: distance between shoulders, or torso height
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
        torso_height = np.linalg.norm(shoulder_center - hip_center)
        scale = shoulder_dist if shoulder_dist > 0.05 else (torso_height if torso_height > 0.05 else 1.0)

        # Center and scale (x, y, z)
        frame[:, :3] = (frame[:, :3] - hip_center) / scale

    if is_flat:
        return seq.reshape(T, 132)
    return seq


def normalize_all_poses(pose_dir, output_dir=None):
    if output_dir is None:
        output_dir = pose_dir

    categories = ["ArmFlapping", "HeadBanging", "Spinning"]
    total_norm = 0

    print(f"[*] Normalizing pose sequences in {pose_dir}...")
    for cat in categories:
        src_cat_dir = os.path.join(pose_dir, cat)
        dst_cat_dir = os.path.join(output_dir, cat)
        os.makedirs(dst_cat_dir, exist_ok=True)

        npy_files = sorted(glob.glob(os.path.join(src_cat_dir, "*.npy")))
        for fpath in npy_files:
            data = np.load(fpath)
            norm_data = normalize_single_sequence(data)
            np.save(os.path.join(dst_cat_dir, Path(fpath).name), norm_data.astype(np.float32))
            total_norm += 1

    print(f"[✓] Successfully normalized {total_norm} pose sequences.")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    pose_dir = base_dir / "data" / "pose"
    normalize_all_poses(str(pose_dir))
