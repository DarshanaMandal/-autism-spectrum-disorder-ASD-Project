"""
MultiSense Experiments: Experiment A - Baseline Random Forest (No GAN)
Trains solely on real training pose features and evaluates on untouched real test set.
"""

import os
import sys
from pathlib import Path

# Add project root and models to path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / "models"))
sys.path.append(str(base_dir / "preprocessing"))

from train_rf import train_and_evaluate_rf


def run_baseline_experiment():
    train_feat = base_dir / "data" / "features" / "features_train_real.csv"
    test_feat = base_dir / "data" / "features" / "features_test_real.csv"
    model_path = base_dir / "models" / "random_forest_baseline.pkl"
    results_dir = base_dir / "results"

    print("\n" + "="*70)
    print("  EXPERIMENT A: BASELINE RANDOM FOREST (REAL DATA ONLY - NO GAN)")
    print("="*70)
    metrics, _ = train_and_evaluate_rf(
        str(train_feat),
        str(test_feat),
        str(model_path),
        str(results_dir),
        exp_name="Baseline (No GAN)"
    )
    return metrics


if __name__ == "__main__":
    run_baseline_experiment()
