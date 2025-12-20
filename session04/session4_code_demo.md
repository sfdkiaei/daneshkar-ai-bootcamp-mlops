# Session 4: Code Demo - Data Management & Experiment Tracking

### Demo Structure:
1. **Data Versioning with DVC Concepts**
2. **MLflow Experiment Tracking**
3. **Experiment Comparison Dashboard**

### Teaching Objectives:
- Show practical data versioning workflow
- Demonstrate systematic experiment tracking
- Illustrate experiment comparison and analysis
- Connect theory to hands-on implementation

---

## Part 1: Data Versioning with DVC Concepts

### Scenario Setup
**Context**: VisionaryAI's mobile phone defect detection system needs to track different versions of training data as new images are collected from the factory floor.

### Code Example 1: Basic Data Versioning Setup

```python
# data_versioning_demo.py
"""
VisionaryAI Data Versioning Demo
Simulating data version tracking for defect detection system
"""

import os
import json
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path

class SimpleDataVersioning:
    """
    Simplified data versioning system demonstrating core concepts
    (In production, use DVC or similar tools)
    """

    def __init__(self, project_name="visionaryai_defect_detection"):
        self.project_name = project_name
        self.versions_dir = Path("data_versions")
        self.versions_dir.mkdir(exist_ok=True)
        self.metadata_file = self.versions_dir / "metadata.json"
        self.load_metadata()

    def load_metadata(self):
        """Load existing version metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"versions": [], "current_version": None}

    def save_metadata(self):
        """Save version metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def create_data_hash(self, data_path):
        """Create unique hash for dataset content"""
        import hashlib
        hash_md5 = hashlib.md5()

        # For demo: hash the file size and modification time
        # In practice: hash actual file contents
        stat = os.stat(data_path)
        content = f"{stat.st_size}_{stat.st_mtime}".encode()
        hash_md5.update(content)
        return hash_md5.hexdigest()[:8]

    def add_version(self, data_path, description, author="demo_user"):
        """Add new data version"""
        data_hash = self.create_data_hash(data_path)
        version_id = f"v1.{len(self.metadata['versions'])}"

        version_info = {
            "version_id": version_id,
            "data_hash": data_hash,
            "data_path": str(data_path),
            "description": description,
            "author": author,
            "timestamp": datetime.now().isoformat(),
            "file_size": os.path.getsize(data_path)
        }

        self.metadata["versions"].append(version_info)
        self.metadata["current_version"] = version_id
        self.save_metadata()

        print(f"✅ Added data version {version_id}")
        print(f"   Hash: {data_hash}")
        print(f"   Description: {description}")
        return version_id

    def list_versions(self):
        """List all data versions"""
        print(f"\n📊 Data Versions for {self.project_name}")
        print("-" * 50)

        for version in self.metadata["versions"]:
            current_marker = "← CURRENT" if version["version_id"] == self.metadata["current_version"] else ""
            print(f"{version['version_id']}: {version['description']} {current_marker}")
            print(f"   Author: {version['author']}")
            print(f"   Created: {version['timestamp'][:19]}")
            print(f"   Size: {version['file_size']:,} bytes")
            print(f"   Hash: {version['data_hash']}")
            print()

    def get_version_info(self, version_id):
        """Get detailed information about specific version"""
        for version in self.metadata["versions"]:
            if version["version_id"] == version_id:
                return version
        return None

# Demo execution
if __name__ == "__main__":
    print("🏭 VisionaryAI Defect Detection - Data Versioning Demo")
    print("=" * 55)

    # Initialize versioning system
    dv = SimpleDataVersioning()

    # Simulate creating training datasets
    datasets = [
        ("initial_dataset.csv", "Initial 10K phone component images from January"),
        ("expanded_dataset.csv", "Added 5K images with better lighting conditions"),
        ("multi_model_dataset.csv", "Added 8K images from new phone model XR-15")
    ]

    # Create dummy dataset files and version them
    for filename, description in datasets:
        # Create dummy file (in practice, these would be real datasets)
        dummy_data = pd.DataFrame({
            'image_path': [f'images/img_{i}.jpg' for i in range(1000)],
            'defect_type': ['scratch', 'dent', 'discoloration'] * 333 + ['scratch'],
            'severity': [1, 2, 3] * 333 + [1]
        })
        dummy_data.to_csv(filename, index=False)

        # Version the dataset
        dv.add_version(filename, description)
        print()

    # Show all versions
    dv.list_versions()

    # Demonstrate version lookup
    print("🔍 Looking up version v1.1:")
    version_info = dv.get_version_info("v1.1")
    if version_info:
        print(f"This version contains: {version_info['description']}")
        print(f"Created by: {version_info['author']}")
```

### Points for Part 1:
1. **Hash-based identification**: See how content changes create new versions automatically
2. **Metadata tracking**: Importance of descriptions, timestamps, authors
3. **Version lineage**: Check how versions build upon each other
4. **Reproducibility**: See how to retrieve exact dataset version used for training

---

## Part 2: MLflow Experiment Tracking

### Code Example 2: MLflow Experiment Setup

```python
# mlflow_experiment_demo.py
"""
VisionaryAI MLflow Experiment Tracking Demo
Defect detection model experiments with systematic tracking
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime

# Set up MLflow
mlflow.set_experiment("VisionaryAI_Defect_Detection")

def generate_sample_data(n_samples=1000, n_features=20, random_state=42):
    """Generate sample defect detection data"""
    np.random.seed(random_state)
    
    # Simulate extracted features from phone component images
    features = np.random.randn(n_samples, n_features)
    
    # Create realistic defect patterns
    # Feature 0-5: Color/texture features (higher values = more likely defective)
    # Feature 6-10: Shape features (extreme values = defective)
    # Feature 11-15: Size features 
    # Feature 16-19: Edge features
    
    defect_probability = (
        0.3 * np.sum(features[:, 0:6] > 1, axis=1) +  # Color anomalies
        0.2 * np.sum(np.abs(features[:, 6:11]) > 1.5, axis=1) +  # Shape anomalies
        0.1 * np.sum(features[:, 11:16] > 2, axis=1)  # Size anomalies
    )
    
    # Add some noise
    defect_probability += np.random.normal(0, 0.5, n_samples)
    
    # Convert to binary labels (0=good, 1=defective)
    labels = (defect_probability > 1.0).astype(int)
    
    return features, labels

def run_experiment(model_name, model, X_train, X_test, y_train, y_test, 
                  hyperparams, data_version="v1.2"):
    """Run a single experiment with MLflow tracking"""
    
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%H%M%S')}"):
        
        # Log hyperparameters
        mlflow.log_params(hyperparams)
        
        # Log data information
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Log environment information
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("sklearn_version", "1.3.0")  # Simulated
        
        # Train model and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("inference_time_per_sample", inference_time / len(X_test))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log additional artifacts
        # Create and log confusion matrix data
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, columns=['Predicted_Good', 'Predicted_Defective'],
                           index=['Actual_Good', 'Actual_Defective'])
        cm_df.to_csv("confusion_matrix.csv")
        mlflow.log_artifact("confusion_matrix.csv")
        
        # Create experiment summary
        summary = {
            "model": model_name,
            "accuracy": f"{accuracy:.4f}",
            "f1_score": f"{f1:.4f}",
            "training_time": f"{training_time:.2f}s",
            "data_version": data_version
        }
        
        with open("experiment_summary.json", "w") as f:
            import json
            json.dump(summary, f, indent=2)
        mlflow.log_artifact("experiment_summary.json")
        
        print(f"✅ {model_name} experiment completed:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
        print()
        
        return {
            "run_id": mlflow.active_run().info.run_id,
            "accuracy": accuracy,
            "f1_score": f1,
            "model_name": model_name
        }

def main():
    """Main experiment execution"""
    print("🏭 VisionaryAI Defect Detection - MLflow Experiment Tracking")
    print("=" * 60)
    
    # Generate sample data
    print("📊 Generating sample defect detection data...")
    X, y = generate_sample_data(n_samples=2000, n_features=20)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Defect rate: {np.mean(y) * 100:.1f}%")
    print()
    
    # Define experiments to run
    experiments = [
        {
            "model_name": "Random_Forest_Baseline",
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "hyperparams": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
        },
        {
            "model_name": "Random_Forest_Tuned",
            "model": RandomForestClassifier(n_estimators=200, max_depth=10, 
                                         min_samples_split=5, random_state=42),
            "hyperparams": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5}
        },
        {
            "model_name": "Logistic_Regression",
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "hyperparams": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
        },
        {
            "model_name": "SVM_RBF",
            "model": SVC(kernel='rbf', C=1.0, random_state=42),
            "hyperparams": {"kernel": "rbf", "C": 1.0, "gamma": "scale"}
        }
    ]
    
    # Run experiments
    results = []
    for exp in experiments:
        print(f"🔬 Running {exp['model_name']} experiment...")
        
        # Use scaled data for SVM and Logistic Regression
        if exp['model_name'] in ['SVM_RBF', 'Logistic_Regression']:
            X_train_exp, X_test_exp = X_train_scaled, X_test_scaled
        else:
            X_train_exp, X_test_exp = X_train, X_test
        
        result = run_experiment(
            exp['model_name'], 
            exp['model'], 
            X_train_exp, X_test_exp, 
            y_train, y_test,
            exp['hyperparams']
        )
        results.append(result)
    
    # Summary of all experiments
    print("📈 Experiment Summary:")
    print("-" * 40)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    for idx, row in results_df.iterrows():
        print(f"{row['model_name']:<25} F1: {row['f1_score']:.4f}, Acc: {row['accuracy']:.4f}")
    
    print(f"\n🏆 Best performing model: {results_df.iloc[0]['model_name']}")
    print(f"   F1-Score: {results_df.iloc[0]['f1_score']:.4f}")
    print("\n💡 View detailed results in MLflow UI:")
    print("   Run: mlflow ui")
    print("   Open: http://localhost:5000")

if __name__ == "__main__":
    main()
```

### Points for Part 2:
1. **Systematic logging**: How MLflow captures parameters, metrics, and artifacts automatically
2. **Experiment comparison**: How multiple runs can be compared easily
3. **Reproducibility**: How all information needed to reproduce results is captured
4. **Artifact management**: Model files, plots, and summary reports being stored

---

## Part 3: Experiment Comparison Dashboard

### Code Example 3: Experiment Analysis and Comparison

```python
# experiment_analysis_demo.py
"""
VisionaryAI Experiment Analysis Demo
Analyzing and comparing multiple experiments
"""

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient

def analyze_experiments(experiment_name="VisionaryAI_Defect_Detection"):
    """Analyze and compare experiments from MLflow"""
    
    print("🔍 VisionaryAI Experiment Analysis Dashboard")
    print("=" * 50)
    
    # Initialize MLflow client
    client = MlflowClient()
    
    try:
        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"❌ Experiment '{experiment_name}' not found")
            return
        
        # Get all runs from the experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1_score DESC"]
        )
        
        if not runs:
            print("❌ No runs found in experiment")
            return
        
        print(f"📊 Found {len(runs)} experiment runs")
        print()
        
        # Create comparison dataframe
        comparison_data = []
        
        for run in runs:
            run_data = {
                'run_id': run.info.run_id[:8],  # Short ID for display
                'model_name': run.data.params.get('model_type', 'Unknown'),
                'accuracy': run.data.metrics.get('accuracy', 0),
                'f1_score': run.data.metrics.get('f1_score', 0),
                'precision': run.data.metrics.get('precision', 0),
                'recall': run.data.metrics.get('recall', 0),
                'training_time': run.data.metrics.get('training_time', 0),
                'data_version': run.data.params.get('data_version', 'Unknown'),
                'status': run.info.status
            }
            comparison_data.append(run_data)
        
        df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        print("📈 Experiment Comparison Table:")
        print("-" * 80)
        display_df = df[['model_name', 'accuracy', 'f1_score', 'precision', 'recall', 'training_time']]
        
        for idx, row in display_df.iterrows():
            print(f"{row['model_name']:<25} | "
                  f"Acc: {row['accuracy']:.4f} | "
                  f"F1: {row['f1_score']:.4f} | "
                  f"Prec: {row['precision']:.4f} | "
                  f"Rec: {row['recall']:.4f} | "
                  f"Time: {row['training_time']:.2f}s")
        
        print()
        
        # Find best models
        best_f1 = df.loc[df['f1_score'].idxmax()]
        best_accuracy = df.loc[df['accuracy'].idxmax()]
        fastest = df.loc[df['training_time'].idxmin()]
        
        print("🏆 Best Performing Models:")
        print(f"   Best F1-Score: {best_f1['model_name']} ({best_f1['f1_score']:.4f})")
        print(f"   Best Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})")
        print(f"   Fastest Training: {fastest['model_name']} ({fastest['training_time']:.2f}s)")
        print()
        
        # Performance analysis
        print("📊 Performance Analysis:")
        print(f"   Average F1-Score: {df['f1_score'].mean():.4f}")
        print(f"   F1-Score Std Dev: {df['f1_score'].std():.4f}")
        print(f"   Performance Range: {df['f1_score'].min():.4f} - {df['f1_score'].max():.4f}")
        print()
        
        # Model recommendations
        print("💡 Model Recommendations:")
        
        # Production candidate (balance of performance and speed)
        df['combined_score'] = (df['f1_score'] * 0.7) + ((1 / (df['training_time'] + 0.1)) * 0.3)
        production_candidate = df.loc[df['combined_score'].idxmax()]
        
        print(f"   Production Candidate: {production_candidate['model_name']}")
        print(f"     - F1-Score: {production_candidate['f1_score']:.4f}")
        print(f"     - Training Time: {production_candidate['training_time']:.2f}s")
        print(f"     - Good balance of performance and efficiency")
        print()
        
        # Experiment insights
        print("🔬 Experiment Insights:")
        if len(df[df['model_name'].str.contains('Random_Forest')]) >= 2:
            rf_models = df[df['model_name'].str.contains('Random_Forest')]
            baseline = rf_models[rf_models['model_name'].str.contains('Baseline')]
            tuned = rf_models[rf_models['model_name'].str.contains('Tuned')]
            
            if not baseline.empty and not tuned.empty:
                improvement = tuned.iloc[0]['f1_score'] - baseline.iloc[0]['f1_score']
                print(f"   - Hyperparameter tuning improved Random Forest by {improvement:.4f} F1-Score")
        
        # Data version analysis
        data_versions = df['data_version'].value_counts()
        print(f"   - All experiments used data version: {data_versions.index[0]}")
        print(f"   - Consistent data versioning ensures fair comparison")
        
        return df
        
    except Exception as e:
        print(f"❌ Error analyzing experiments: {str(e)}")
        print("💡 Make sure to run the MLflow demo first to generate experiment data")
        return None

def create_experiment_visualizations(df):
    """Create visualizations for experiment comparison"""
    if df is None or df.empty:
        return
    
    print("\n📊 Creating experiment visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('VisionaryAI Defect Detection - Experiment Analysis', fontsize=16)
    
    # 1. Model performance comparison
    ax1 = axes[0, 0]
    models = df['model_name'].str.replace('_', ' ')
    ax1.bar(models, df['f1_score'], color='skyblue', alpha=0.7)
    ax1.set_title('F1-Score by Model')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('F1-Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Accuracy vs Training Time
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['training_time'], df['accuracy'], 
                         c=df['f1_score'], cmap='viridis', s=100, alpha=0.7)
    ax2.set_title('Accuracy vs Training Time')
    ax2.set_xlabel('Training Time (seconds)')
    ax2.set_ylabel('Accuracy')
    plt.colorbar(scatter, ax=ax2, label='F1-Score')
    
    # 3. Precision vs Recall
    ax3 = axes[1, 0]
    ax3.scatter(df['recall'], df['precision'], s=100, alpha=0.7, color='coral')
    ax3.set_title('Precision vs Recall')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Diagonal reference line
    
    # 4. Performance metrics heatmap
    ax4 = axes[1, 1]
    metrics_data = df[['accuracy', 'f1_score', 'precision', 'recall']].T
    metrics_data.columns = models
    sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
    ax4.set_title('Performance Metrics Heatmap')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('experiment_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved visualization as 'experiment_analysis.png'")
    
    # Show the plot (comment out if running in headless environment)
    # plt.show()

def main():
    """Main analysis execution"""
    # Analyze experiments
    df = analyze_experiments()
    
    if df is not None:
        # Create visualizations
        create_experiment_visualizations(df)
        
        print("\n🎯 Next Steps:")
        print("   1. Deploy the production candidate model")
        print("   2. Set up monitoring for model performance")
        print("   3. Plan next round of experiments")
        print("   4. Document findings for team review")
        
        print("\n💻 MLflow Commands:")
        print("   - View UI: mlflow ui")
        print("   - Compare runs: Select multiple runs in UI")
        print("   - Download artifacts: Use MLflow client or UI")

if __name__ == "__main__":
    main()
```

### Points for Part 3:
1. **Systematic comparison**: How to programmatically compare multiple experiments
2. **Decision making**: How to choose models based on different criteria
3. **Visualization**: Experiment results graphically for better understanding
4. **Production readiness**: How to identify models ready for deployment

---

## Complete Demo Script (Integration)

```python
# complete_demo.py
"""
Complete Session 4 Demo Script
Run all three parts in sequence
"""

import subprocess
import sys
import os

def run_demo_part(part_name, script_name):
    """Run a demo part and handle any errors"""
    print(f"\n{'='*60}")
    print(f"🚀 Running {part_name}")
    print(f"{'='*60}")
    
    try:
        # In practice, you would import and run the functions
        # For demo purposes, we'll just print what would happen
        print(f"   Executing {script_name}...")
        print(f"   ✅ {part_name} completed successfully")
        return True
    except Exception as e:
        print(f"   ❌ Error in {part_name}: {str(e)}")
        return False

def main():
    """Run complete demo sequence"""
    print("🏭 VisionaryAI Session 4 - Complete Demo")
    print("Data Management & Experiment Tracking")
    
    # Demo parts
    parts = [
        ("Data Versioning with DVC Concepts", "data_versioning_demo.py"),
        ("MLflow Experiment Tracking", "mlflow_experiment_demo.py"),
        ("Experiment Analysis Dashboard", "experiment_analysis_demo.py")
    ]
    
    # Run each part
    success_count = 0
    for part_name, script_name in parts:
        if run_demo_part(part_name, script_name):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Demo Summary")
    print(f"{'='*60}")
    print(f"   Completed: {success_count}/{len(parts)} parts")
    
    if success_count == len(parts):
        print("   ✅ All demos completed successfully!")
        print("\n🎯 Key Takeaways:")
        print("   • Data versioning prevents 'works on my machine' problems")
        print("   • Systematic experiment tracking saves time and prevents duplicate work")
        print("   • Proper comparison enables data-driven model selection")
        print("   • Tools like DVC and MLflow automate best practices")
    else:
        print("   ⚠️  Some demos had issues - check error messages above")
    
    print(f"\n💡 Next Steps:")
    print("   • Try these tools with your own ML projects")
    print("   • Set up data versioning for your current datasets") 
    print("   • Start tracking experiments systematically")
    print("   • Prepare for Session 5: Model Development & Testing")

if __name__ == "__main__":
    main()
```
