"""
MultiSense Preprocessing: Pose Sequence Creator & Strict Data Splitter
1. Loads normalized pose sequences (T, 132).
2. Segregates at VIDEO/SUBJECT level into 80% Train and 20% Test (zero-leakage guarantee).
3. Windows sequences into fixed-length temporal windows (T = 60 frames).
4. Saves X_train_real.npy, y_train_real.npy, X_test_real.npy, y_test_real.npy.
"""

import os
import glob
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

LABEL_MAP = {
    "ArmFlapping": 0,
    "HeadBanging": 1,
    "Spinning": 2
}
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def window_sequence(seq, seq_len=60, stride=30):
    """
    Slices a continuous pose sequence into fixed-length windows of length seq_len.
    If seq is shorter than seq_len, pads with edge replication.
    """
    T = seq.shape[0]
    if T < seq_len:
        pad_width = seq_len - T
        pad = np.repeat(seq[-1:], pad_width, axis=0)
        return np.expand_dims(np.concatenate([seq, pad], axis=0), axis=0)

    windows = []
    for start in range(0, T - seq_len + 1, stride):
        windows.append(seq[start:start + seq_len])
    
    # If no window created (e.g. slight discrepancy), take first seq_len
    if len(windows) == 0:
        windows.append(seq[:seq_len])

    return np.array(windows)


def create_dataset_splits(pose_dir, sequences_dir, seq_len=60, stride=30, test_size=0.2, random_state=42):
    os.makedirs(sequences_dir, exist_ok=True)
    categories = ["ArmFlapping", "HeadBanging", "Spinning"]

    # Collect all video files and their class
    video_records = []
    for cat in categories:
        cat_dir = os.path.join(pose_dir, cat)
        npy_files = sorted(glob.glob(os.path.join(cat_dir, "*.npy")))
        for fpath in npy_files:
            # Extract video prefix to group multiple segments of same video
            stem = Path(fpath).stem
            # e.g. v_ArmFlapping_01_b_01 -> base video is v_ArmFlapping_01
            base_vid = "_".join(stem.split("_")[:3])
            video_records.append({
                "filepath": fpath,
                "base_video": base_vid,
                "category": cat,
                "label": LABEL_MAP[cat]
            })

    # Group by base video for zero-leakage subject-level split
    unique_base = list(set(r["base_video"] for r in video_records))
    base_labels = []
    for b in unique_base:
        matching = [r["label"] for r in video_records if r["base_video"] == b]
        base_labels.append(matching[0])

    train_bases, test_bases = train_test_split(
        unique_base,
        test_size=test_size,
        stratify=base_labels,
        random_state=random_state
    )

    train_bases_set = set(train_bases)
    test_bases_set = set(test_bases)

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for r in video_records:
        seq_data = np.load(r["filepath"])
        windows = window_sequence(seq_data, seq_len=seq_len, stride=stride)
        labels = [r["label"]] * len(windows)

        if r["base_video"] in train_bases_set:
            X_train_list.append(windows)
            y_train_list.extend(labels)
        else:
            X_test_list.append(windows)
            y_test_list.extend(labels)

    X_train = np.concatenate(X_train_list, axis=0).astype(np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_test = np.concatenate(X_test_list, axis=0).astype(np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)

    # Save arrays
    np.save(os.path.join(sequences_dir, "X_train_real.npy"), X_train)
    np.save(os.path.join(sequences_dir, "y_train_real.npy"), y_train)
    np.save(os.path.join(sequences_dir, "X_test_real.npy"), X_test)
    np.save(os.path.join(sequences_dir, "y_test_real.npy"), y_test)

    meta = {
        "sequence_length": seq_len,
        "feature_dim": X_train.shape[2],
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_class_distribution": {
            REV_LABEL_MAP[i]: int(np.sum(y_train == i)) for i in range(3)
        },
        "test_class_distribution": {
            REV_LABEL_MAP[i]: int(np.sum(y_test == i)) for i in range(3)
        },
        "train_videos_count": len(train_bases),
        "test_videos_count": len(test_bases)
    }

    with open(os.path.join(sequences_dir, "split_metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n" + "="*60)
    print("[✓] Sequence Dataset Created with Zero-Leakage Split!")
    print(f"    - X_train_real shape: {X_train.shape}")
    print(f"    - y_train_real distribution: {meta['train_class_distribution']}")
    print(f"    - X_test_real shape:  {X_test.shape}")
    print(f"    - y_test_real distribution:  {meta['test_class_distribution']}")
    print("="*60 + "\n")
    return meta


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    pose_dir = base_dir / "data" / "pose"
    seq_dir = base_dir / "data" / "sequences"
    create_dataset_splits(str(pose_dir), str(seq_dir))
