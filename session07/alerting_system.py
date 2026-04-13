import time
from datetime import datetime
from typing import Dict, List, Any
from enum import Enum


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    CONSOLE = "console"
    EMAIL = "email"
    SLACK = "slack"
    PAGER = "pager"


class SimpleAlerter:
    """Basic alerting system for ML model monitoring"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.alert_rules = []
        self.alert_history = []

    def add_rule(
        self,
        metric_name: str,
        threshold: float,
        condition: str,  # 'greater_than', 'less_than'
        level: AlertLevel,
        channels: List[AlertChannel],
    ):
        """Add an alerting rule"""

        rule = {
            "metric_name": metric_name,
            "threshold": threshold,
            "condition": condition,
            "level": level,
            "channels": channels,
            "enabled": True,
        }
        self.alert_rules.append(rule)

    def check_metrics(self, metrics: Dict[str, float]):
        """Check metrics against all rules and fire alerts"""

        alerts_fired = []

        for rule in self.alert_rules:
            if not rule["enabled"]:
                continue

            metric_name = rule["metric_name"]
            if metric_name not in metrics:
                continue

            current_value = metrics[metric_name]
            threshold = rule["threshold"]
            condition = rule["condition"]

            should_alert = False

            if condition == "greater_than" and current_value > threshold:
                should_alert = True
            elif condition == "less_than" and current_value < threshold:
                should_alert = True

            if should_alert:
                alert = self._create_alert(rule, current_value)
                alerts_fired.append(alert)
                self._send_alert(alert)

        return alerts_fired

    def _create_alert(self, rule: Dict, current_value: float) -> Dict:
        """Create alert object"""

        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": self.model_name,
            "metric_name": rule["metric_name"],
            "current_value": current_value,
            "threshold": rule["threshold"],
            "condition": rule["condition"],
            "level": rule["level"],
            "channels": rule["channels"],
            "message": self._generate_message(rule, current_value),
        }

        self.alert_history.append(alert)
        return alert

    def _generate_message(self, rule: Dict, current_value: float) -> str:
        """Generate human-readable alert message"""

        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
        }

        emoji = level_emoji.get(rule["level"], "📊")

        message = (
            f"{emoji} {rule['level'].value.upper()}: {self.model_name}\n"
            f"Metric: {rule['metric_name']}\n"
            f"Current: {current_value:.3f}\n"
            f"Threshold: {rule['threshold']:.3f}\n"
            f"Condition: {rule['condition']}"
        )

        # Add suggested actions
        if rule["metric_name"] == "accuracy" and rule["level"] == AlertLevel.CRITICAL:
            message += "\n\n🔧 Suggested Actions:\n- Check for data drift\n- Review recent model changes\n- Consider rolling back"
        elif rule["metric_name"] == "response_time_ms":
            message += "\n\n🔧 Suggested Actions:\n- Check system resources\n- Review recent deployments\n- Scale up if needed"

        return message

    def _send_alert(self, alert: Dict):
        """Send alert through configured channels"""

        for channel in alert["channels"]:
            if channel == AlertChannel.CONSOLE:
                self._send_console_alert(alert)
            elif channel == AlertChannel.SLACK:
                self._send_slack_alert(alert)
            elif channel == AlertChannel.EMAIL:
                self._send_email_alert(alert)
            elif channel == AlertChannel.PAGER:
                self._send_pager_alert(alert)

    def _send_console_alert(self, alert: Dict):
        """Print alert to console (for demo)"""
        print(f"\n{'=' * 50}")
        print(f"ALERT: {alert['level'].value.upper()}")
        print(f"{'=' * 50}")
        print(alert["message"])
        print(f"{'=' * 50}")

    def _send_slack_alert(self, alert: Dict):
        """Simulate Slack alert"""
        print(f"📱 [SLACK] Sent to #ml-alerts channel: {alert['metric_name']} alert")

    def _send_email_alert(self, alert: Dict):
        """Simulate email alert"""
        print(
            f"📧 [EMAIL] Sent to ml-team@visionaryai.com: {alert['level'].value} alert"
        )

    def _send_pager_alert(self, alert: Dict):
        """Simulate pager alert"""
        print(
            f"📟 [PAGER] Paging on-call engineer: CRITICAL {alert['metric_name']} alert"
        )


def demo_alerting_system():
    print("=== Part 4: Basic Alerting System ===")

    # Set up alerter for VisionaryAI's defect detection system
    alerter = SimpleAlerter("Defect Detection System")

    print("\n1. Setting up alert rules...")

    # Rule 1: Critical accuracy threshold
    alerter.add_rule(
        metric_name="accuracy",
        threshold=0.90,
        condition="less_than",
        level=AlertLevel.CRITICAL,
        channels=[AlertChannel.CONSOLE, AlertChannel.PAGER, AlertChannel.SLACK],
    )

    # Rule 2: Warning for response time
    alerter.add_rule(
        metric_name="response_time_ms",
        threshold=100,
        condition="greater_than",
        level=AlertLevel.WARNING,
        channels=[AlertChannel.CONSOLE, AlertChannel.SLACK],
    )

    # Rule 3: Critical response time
    alerter.add_rule(
        metric_name="response_time_ms",
        threshold=200,
        condition="greater_than",
        level=AlertLevel.CRITICAL,
        channels=[AlertChannel.CONSOLE, AlertChannel.PAGER, AlertChannel.EMAIL],
    )

    print("   ✅ 3 alert rules configured")

    print("\n2. Testing with normal metrics (no alerts expected)...")
    normal_metrics = {"accuracy": 0.94, "response_time_ms": 45, "confidence": 0.88}

    alerts = alerter.check_metrics(normal_metrics)
    if not alerts:
        print("   ✅ No alerts fired - system healthy")

    print("\n3. Testing with degraded performance...")
    degraded_metrics = {
        "accuracy": 0.88,  # Below critical threshold!
        "response_time_ms": 150,  # Above warning threshold
        "confidence": 0.82,
    }

    alerts = alerter.check_metrics(degraded_metrics)
    print(f"   🚨 {len(alerts)} alerts fired")

    print("\n4. Testing with severe issues...")
    severe_metrics = {
        "accuracy": 0.85,  # Still critical
        "response_time_ms": 250,  # Now critical too!
        "confidence": 0.75,
    }

    alerts = alerter.check_metrics(severe_metrics)
    print(f"   🚨 {len(alerts)} alerts fired")

    print(f"\n📊 Total alerts in history: {len(alerter.alert_history)}")

    print("\n✅ Key Point: Set meaningful thresholds based on business impact")
    print("✅ Key Point: Use escalation - warnings to Slack, critical to pagers")
    print("✅ Key Point: Include suggested actions in alerts")


if __name__ == "__main__":
    demo_alerting_system()
