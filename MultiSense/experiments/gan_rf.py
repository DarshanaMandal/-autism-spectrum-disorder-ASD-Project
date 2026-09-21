"""
MultiSense Experiments: Experiment B - GAN-Augmented Random Forest
Trains on Combined Real + Synthetic Pose Features, evaluates on UNTOUCHED real test set.
Exports side-by-side comparative analysis: results/experiment_comparison.csv.
"""

import os
import sys
from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / "models"))
sys.path.append(str(base_dir / "preprocessing"))

from train_rf import train_and_evaluate_rf


def run_gan_augmented_experiment():
    real_train_feat = base_dir / "data" / "features" / "features_train_real.csv"
    aug_train_feat = base_dir / "data" / "features" / "features_train_augmented.csv"
    test_feat = base_dir / "data" / "features" / "features_test_real.csv"
    model_baseline_path = base_dir / "models" / "random_forest_baseline.pkl"
    model_gan_path = base_dir / "models" / "random_forest_gan.pkl"
    results_dir = base_dir / "results"

    print("\n" + "="*70)
    print("  RUNNING COMPLETE EXPERIMENTAL COMPARISON: NO-GAN vs. WITH-GAN")
    print("="*70)

    # 1. Run Baseline
    print("\n>>> Running Baseline (Real Only)...")
    metrics_baseline, _ = train_and_evaluate_rf(
        str(real_train_feat),
        str(test_feat),
        str(model_baseline_path),
        str(results_dir),
        exp_name="Baseline (No GAN)"
    )

    # 2. Run GAN-Augmented
    print("\n>>> Running GAN-Augmented (Real + Synthetic)...")
    metrics_gan, _ = train_and_evaluate_rf(
        str(aug_train_feat),
        str(test_feat),
        str(model_gan_path),
        str(results_dir),
        exp_name="GAN-Augmented RF"
    )

    # 3. Create Comparison Table
    comparison_rows = [metrics_baseline, metrics_gan]
    df_comp = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(results_dir, "experiment_comparison.csv")
    df_comp.to_csv(comp_path, index=False)

    print("\n" + "="*70)
    print("  SUMMARY: EXPERIMENT COMPARISON TABLE")
    print("="*70)
    print(df_comp[["experiment", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]].to_string(index=False))
    print(f"\n[✓] Comparison exported to: {comp_path}")
    print("="*70 + "\n")
    return df_comp


if __name__ == "__main__":
    run_gan_augmented_experiment()
