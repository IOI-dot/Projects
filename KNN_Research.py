import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data
data_file = "C:/Users/Omar/Downloads/combined_scattering_data.pkl"
with open(data_file, "rb") as f:
    df = pickle.load(f)

# Extract features and labels
X = np.array([np.array(h) for h in df['scattering_pattern']])
y = df['particle_type'].map({'plastic': 0, 'colloid': 1}).values  # 0 = plastic, 1 = colloid

# Flatten each heatmap into 1D vector
X_flat = X.reshape(X.shape[0], -1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train KNN
knn = KNeighborsClassifier(n_neighbors=4, metric="euclidean")
knn.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Plastic", "Colloid"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
