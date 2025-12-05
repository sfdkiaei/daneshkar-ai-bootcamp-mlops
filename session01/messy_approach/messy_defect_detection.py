# Session 1 Code Demo: From Messy to Organized ML Code
# VisionaryAI - Computer Vision Defect Detection Example

# =============================================================================
# PART 1: THE MESSY APPROACH (What NOT to do)
# =============================================================================

# messy_defect_detection.py (or more realistically, a Jupyter notebook cell)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import cv2
import os

# Load data (hardcoded paths, no error handling)
data = pd.read_csv("/Users/john/Desktop/factory_data.csv")
images_path = "/Users/john/Desktop/images/"

# Quick data exploration (results not saved)
print(data.shape)
print(data.head())

# Data preprocessing (magic numbers everywhere)
data = data.dropna()
data = data[data["defect_score"] > 0.1]  # Why 0.1? Nobody knows!

# Feature extraction (no documentation)
features = []
labels = []
for idx, row in data.iterrows():
    img_path = images_path + row["image_filename"]
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        # Some magical feature extraction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])  # Why 32 bins?
        features.append(hist.flatten())
        labels.append(row["is_defective"])

# Convert to numpy (inefficient)
X = np.array(features)
y = np.array(labels)

# Split data (random seed? What's that?)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model (hyperparameters chosen randomly)
model = RandomForestClassifier(n_estimators=50, max_depth=10)
model.fit(X_train, y_train)

# Evaluate (single metric, no context)
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Accuracy: {acc}")  # 0.87 - Is this good? Bad? We don't know!

# Save model (no versioning, overwrites previous models)
pickle.dump(model, open("defect_model.pkl", "wb"))

print("Done! Model saved.")