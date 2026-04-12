"""
VisionaryAI Customer Support Ticket Classifier
Step 1: Train and serialize the model
"""

import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import json
from datetime import datetime

# Sample training data (in reality, this would be much larger)
training_data = [
    ("My phone screen is cracked", "hardware_issue"),
    ("Cannot connect to wifi", "connectivity"),
    ("App keeps crashing", "software_issue"),
    ("Battery drains too fast", "hardware_issue"),
    ("Cannot log into my account", "account_issue"),
    ("Website is loading slowly", "connectivity"),
    ("Feature request for dark mode", "feature_request"),
    ("Billing question about my subscription", "billing"),
    ("How do I reset my password", "account_issue"),
    ("Phone gets very hot during use", "hardware_issue"),
]

# Separate features and labels
texts = [item[0] for item in training_data]
labels = [item[1] for item in training_data]

print("🤖 Training VisionaryAI Customer Support Classifier...")

# Create a pipeline with preprocessing and model
# This ensures preprocessing is saved with the model
classifier = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
        ("nb", MultinomialNB()),
    ]
)

# Train the model
classifier.fit(texts, labels)

# Create model metadata
model_metadata = {
    "model_name": "customer_support_classifier",
    "version": "1.0.0",
    "training_date": datetime.now().isoformat(),
    "accuracy": "0.89",  # In reality, calculated from validation set
    "features": "TF-IDF vectors (max 1000 features)",
    "algorithm": "Multinomial Naive Bayes",
    "categories": list(set(labels)),
}

print(f"✅ Model trained! Categories: {model_metadata['categories']}")

# Method 1: Save with joblib (recommended for sklearn)
print("\n💾 Saving model with joblib...")
joblib.dump(classifier, "models/support_classifier.joblib")
print("✅ Saved as support_classifier.joblib")

# Method 2: Save with pickle (alternative)
print("\n💾 Saving model with pickle...")
with open("models/support_classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)
print("✅ Saved as support_classifier.pkl")

# Save metadata separately
print("\n📋 Saving model metadata...")
with open("models/model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)
print("✅ Metadata saved")

# Test the saved model
print("\n🧪 Testing saved model...")
loaded_model = joblib.load("models/support_classifier.joblib")
test_text = "My phone screen has a crack on it"
prediction = loaded_model.predict([test_text])[0]
confidence = max(loaded_model.predict_proba([test_text])[0])

print(f"Test input: '{test_text}'")
print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.3f}")
print("\n🎉 Model serialization complete!")
