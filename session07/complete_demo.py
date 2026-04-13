if __name__ == "__main__":
    print("🚀 VisionaryAI Production Operations & Monitoring Demo")
    print("=" * 60)

    # Import and run all demo parts
    from basic_ml_logging import demo_basic_logging
    from monitoring_dashboard import demo_monitoring_dashboard
    from drift_detection import demo_drift_detection
    from alerting_system import demo_alerting_system

    try:
        demo_basic_logging()
        print("\n" + "=" * 60)

        demo_monitoring_dashboard()
        print("\n" + "=" * 60)

        demo_drift_detection()
        print("\n" + "=" * 60)

        demo_alerting_system()
        print("\n" + "=" * 60)

        print("\n🎉 Demo Complete!")
        print("\nKey Takeaways:")
        print("1. 📝 Structure your ML logs with JSON for easy analysis")
        print("2. 📊 Monitor trends, not just point-in-time metrics")
        print("3. 🔍 Detect drift before it impacts your business")
        print("4. 🚨 Set up smart alerts with clear action items")
        print("\nNext: Implement these patterns in your production ML systems!")

    except ImportError as e:
        print(f"Error importing demo modules: {e}")
        print("Make sure all demo files are in the same directory")
