import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class DriftDetector:
    """Simple drift detection for model inputs"""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.baseline_data = None
        self.baseline_stats = {}

    def set_baseline(self, data: np.ndarray):
        """Set baseline data distribution (usually training data)"""
        self.baseline_data = data

        # Compute statistics for each feature
        for i, feature_name in enumerate(self.feature_names):
            feature_data = data[:, i]
            self.baseline_stats[feature_name] = {
                "mean": np.mean(feature_data),
                "std": np.std(feature_data),
                "min": np.min(feature_data),
                "max": np.max(feature_data),
                "distribution": feature_data,  # Store for statistical tests
            }

    def detect_drift(self, new_data: np.ndarray, alpha: float = 0.05) -> Dict:
        """Detect if new data has drifted from baseline"""
        if self.baseline_data is None:
            raise ValueError("Must set baseline data first")

        drift_results = {
            "overall_drift": False,
            "feature_results": {},
            "drift_score": 0.0,
        }

        drift_count = 0

        for i, feature_name in enumerate(self.feature_names):
            baseline_feature = self.baseline_data[:, i]
            new_feature = new_data[:, i]

            # Statistical test for distribution difference (Kolmogorov-Smirnov)
            ks_statistic, p_value = stats.ks_2samp(baseline_feature, new_feature)

            # Simple threshold-based checks
            baseline_mean = self.baseline_stats[feature_name]["mean"]
            baseline_std = self.baseline_stats[feature_name]["std"]
            new_mean = np.mean(new_feature)

            # Check if new mean is more than 2 standard deviations away
            mean_shift = abs(new_mean - baseline_mean) / baseline_std

            has_drift = (p_value < alpha) or (mean_shift > 2.0)

            drift_results["feature_results"][feature_name] = {
                "drift_detected": has_drift,
                "ks_statistic": ks_statistic,
                "p_value": p_value,
                "mean_shift_std": mean_shift,
                "baseline_mean": baseline_mean,
                "new_mean": new_mean,
            }

            if has_drift:
                drift_count += 1

        drift_results["overall_drift"] = drift_count > 0
        drift_results["drift_score"] = drift_count / len(self.feature_names)

        return drift_results

    def visualize_drift(self, new_data: np.ndarray, max_features: int = 4):
        """Create visualizations showing drift for top features"""
        if self.baseline_data is None:
            print("No baseline data available")
            return

        # Detect drift first
        drift_results = self.detect_drift(new_data)

        # Sort features by drift severity
        features_by_drift = sorted(
            drift_results["feature_results"].items(),
            key=lambda x: x[1]["ks_statistic"],
            reverse=True,
        )

        # Plot top features
        n_features = min(max_features, len(self.feature_names))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        fig.suptitle("Data Drift Detection - Top Features", fontsize=14)

        for i in range(n_features):
            feature_name, drift_info = features_by_drift[i]
            feature_idx = self.feature_names.index(feature_name)

            ax = axes[i]

            # Plot histograms
            baseline_feature = self.baseline_data[:, feature_idx]
            new_feature = new_data[:, feature_idx]

            ax.hist(
                baseline_feature, bins=20, alpha=0.5, label="Baseline", color="blue"
            )
            ax.hist(new_feature, bins=20, alpha=0.5, label="Current", color="red")

            # Add drift indicators
            drift_status = (
                "DRIFT DETECTED" if drift_info["drift_detected"] else "No Drift"
            )
            color = "red" if drift_info["drift_detected"] else "green"

            ax.set_title(f"{feature_name}\n{drift_status}", color=color)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def demo_drift_detection():
    print("=== Part 3: Drift Detection Example ===")

    # Simulate VisionaryAI's defect detection system
    feature_names = ["brightness", "contrast", "edge_density", "color_variance"]
    detector = DriftDetector(feature_names)

    print("\n1. Setting up baseline data (training data)...")
    # Generate baseline data (what model was trained on)
    np.random.seed(42)
    baseline_data = np.random.normal(
        loc=[0.5, 0.6, 0.4, 0.3],  # Feature means during training
        scale=[0.1, 0.15, 0.08, 0.12],  # Feature standard deviations
        size=(1000, 4),  # 1000 samples, 4 features
    )

    detector.set_baseline(baseline_data)

    print("2. Simulating current production data...")
    # Scenario: Factory upgraded cameras, images are now brighter and higher contrast
    current_data = np.random.normal(
        loc=[0.7, 0.8, 0.4, 0.3],  # Brightness and contrast increased!
        scale=[0.1, 0.15, 0.08, 0.12],
        size=(500, 4),  # Smaller sample of recent data
    )

    print("3. Detecting drift...")
    drift_results = detector.detect_drift(current_data)

    print(f"\n📊 Overall drift detected: {drift_results['overall_drift']}")
    print(f"📊 Drift score: {drift_results['drift_score']:.2f}")

    print("\n📋 Feature-level results:")
    for feature, result in drift_results["feature_results"].items():
        status = "🚨 DRIFT" if result["drift_detected"] else "✅ OK"
        print(f"   {feature}: {status}")
        print(f"      Mean shift: {result['mean_shift_std']:.2f} std deviations")
        print(f"      P-value: {result['p_value']:.4f}")

    print("\n4. Visualizing drift...")
    detector.visualize_drift(current_data)

    print("\n✅ Key Point: Monitor input data distributions, not just model outputs")
    print("✅ Key Point: Use statistical tests to detect significant changes")
    print("✅ Key Point: Visualize drift to understand what changed")


if __name__ == "__main__":
    demo_drift_detection()
