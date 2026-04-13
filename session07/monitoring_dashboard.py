import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from typing import List, Dict, Any


class ModelMonitor:
    """Simple monitoring dashboard for ML models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics_history = []

    def add_metrics(self, timestamp: datetime, metrics: Dict[str, float]):
        """Add a set of metrics at a specific timestamp"""
        entry = {"timestamp": timestamp, **metrics}
        self.metrics_history.append(entry)

    def get_dataframe(self) -> pd.DataFrame:
        """Convert metrics history to pandas DataFrame for analysis"""
        return pd.DataFrame(self.metrics_history)

    def plot_performance_trends(self):
        """Create dashboard showing model performance trends"""
        if not self.metrics_history:
            print("No metrics data available")
            return

        df = self.get_dataframe()

        # Create subplots for different metric types
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{self.model_name} - Performance Dashboard", fontsize=16)

        # Plot 1: Model Accuracy over time
        axes[0, 0].plot(df["timestamp"], df["accuracy"], "b-", linewidth=2)
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(
            y=0.9, color="r", linestyle="--", alpha=0.7, label="Threshold"
        )
        axes[0, 0].legend()

        # Plot 2: Response Time
        axes[0, 1].plot(df["timestamp"], df["avg_response_time_ms"], "g-", linewidth=2)
        axes[0, 1].set_title("Average Response Time")
        axes[0, 1].set_ylabel("Response Time (ms)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(
            y=100, color="r", linestyle="--", alpha=0.7, label="SLA Limit"
        )
        axes[0, 1].legend()

        # Plot 3: Confidence Score Distribution
        axes[1, 0].hist(df["avg_confidence"], bins=20, alpha=0.7, color="orange")
        axes[1, 0].set_title("Confidence Score Distribution")
        axes[1, 0].set_xlabel("Average Confidence")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Business Impact (if available)
        if "business_metric" in df.columns:
            axes[1, 1].plot(
                df["timestamp"], df["business_metric"], "purple", linewidth=2
            )
            axes[1, 1].set_title("Business Impact")
            axes[1, 1].set_ylabel("Business Metric")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Business metrics\nnot available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Business Impact")

        plt.tight_layout()
        plt.show()

    def detect_performance_issues(self) -> List[str]:
        """Detect potential performance issues"""
        if not self.metrics_history:
            return ["No data available for analysis"]

        df = self.get_dataframe()
        issues = []

        # Check recent accuracy trend
        recent_data = df.tail(10)  # Last 10 data points
        if len(recent_data) >= 3:
            accuracy_trend = np.polyfit(
                range(len(recent_data)), recent_data["accuracy"], 1
            )[0]
            if accuracy_trend < -0.001:  # Declining trend
                issues.append("⚠️  Model accuracy is declining")

        # Check current accuracy level
        latest_accuracy = df["accuracy"].iloc[-1]
        if latest_accuracy < 0.9:
            issues.append(
                f"🚨 Current accuracy ({latest_accuracy:.3f}) below threshold (0.9)"
            )

        # Check response time
        latest_response_time = df["avg_response_time_ms"].iloc[-1]
        if latest_response_time > 100:
            issues.append(
                f"🚨 Response time ({latest_response_time:.1f}ms) exceeds SLA (100ms)"
            )

        # Check confidence scores
        avg_confidence = df["avg_confidence"].mean()
        if avg_confidence < 0.7:
            issues.append(f"⚠️  Average confidence ({avg_confidence:.3f}) is low")

        return issues if issues else ["✅ No performance issues detected"]


def demo_monitoring_dashboard():
    print("=== Part 2: Simple Monitoring Dashboard ===")

    # Create monitor for VisionaryAI's defect detection system
    monitor = ModelMonitor("Defect Detection System")

    # Simulate metrics data over time (normally this comes from logs)
    print("\n1. Generating sample metrics data...")
    base_time = datetime.now() - timedelta(days=7)

    for i in range(50):  # 50 data points over a week
        timestamp = base_time + timedelta(hours=i * 3)  # Every 3 hours

        # Simulate gradual performance degradation
        accuracy_base = 0.95 - (i * 0.001)  # Slight decline over time
        accuracy = accuracy_base + np.random.normal(0, 0.01)  # Add noise

        response_time = 45 + (i * 0.5) + np.random.normal(0, 5)  # Gradual increase
        confidence = 0.85 + np.random.normal(0, 0.05)

        # Simulate business metric (could be production throughput, quality scores, etc.)
        business_metric = 98.5 - (i * 0.02) + np.random.normal(0, 0.5)

        metrics = {
            "accuracy": max(0.8, min(1.0, accuracy)),
            "avg_response_time_ms": max(20, response_time),
            "avg_confidence": max(0.5, min(1.0, confidence)),
            "business_metric": max(90, business_metric),
            "predictions_per_hour": 1000 + np.random.normal(0, 50),
        }

        monitor.add_metrics(timestamp, metrics)

    print("2. Creating performance dashboard...")
    monitor.plot_performance_trends()

    print("\n3. Detecting performance issues...")
    issues = monitor.detect_performance_issues()
    for issue in issues:
        print(f"   {issue}")

    print("\n✅ Key Point: Track both technical and business metrics")
    print("✅ Key Point: Look for trends, not just point-in-time values")
    print("✅ Key Point: Set up automated issue detection")


if __name__ == "__main__":
    demo_monitoring_dashboard()
