import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# Load Iris Dataset
iris = load_iris()

data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

data["class"] = iris.target_names[iris.target]


# ==========================================
# QUESTION 2 - PROBLEM IDENTIFICATION
# ==========================================

print("==========================================")
print("QUESTION 2 - SUPERVISED & UNSUPERVISED LEARNING")
print("==========================================")

# Input Features
features = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

X = data[features]

# Target Variable
y = data["class"]

print("\n===== Problem Identification =====")

print("\nInput Features:")
print(features)

print("\nTarget Variable:")
print("class")

print("\nType of Machine Learning Problem:")
print("Supervised Learning - Classification")

print("\nExpected Output:")
print("Predicted Iris flower class")

print("\nUnsupervised Learning:")
print("The same numerical features will be used without target labels.")
print("The objective is to identify possible natural groups or patterns.")
# ==========================================
# SUPERVISED LEARNING - CLASSIFICATION
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\n===== Supervised Learning =====")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Random Forest classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel: Random Forest Classifier")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# ==========================================
# SUPERVISED LEARNING - CONFUSION MATRIX
# ==========================================

print("\n===== Confusion Matrix =====")

cm = confusion_matrix(y_test, y_pred)

print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()
# ==========================================
# UNSUPERVISED LEARNING - K-MEANS CLUSTERING
# ==========================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("\n===== Unsupervised Learning =====")

# Use features only - target/class is NOT used
X_unsupervised = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unsupervised)

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Find clusters
clusters = kmeans.fit_predict(X_scaled)

print("\nAlgorithm: K-Means Clustering")
print("Number of Clusters:", 3)

print("\nNumber of Samples in Each Cluster:")
print(pd.Series(clusters).value_counts().sort_index())

# Add cluster labels for analysis
data["cluster"] = clusters

print("\nFirst 10 Cluster Assignments:")
print(data[features + ["cluster"]].head(10))
# ==========================================
# K-MEANS CLUSTER VISUALIZATION
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    data["petal length (cm)"],
    data["petal width (cm)"],
    c=data["cluster"],
    cmap="viridis",
    s=50
)

# Plot cluster centers after converting them back to original scale
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:, 2],
    centers[:, 3],
    marker="X",
    s=200,
    edgecolor="black",
    label="Cluster Centers"
)

plt.title("K-Means Clustering of Iris Dataset")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.tight_layout()
plt.show()
# ==========================================
# COMPARISON: CLUSTERS VS ACTUAL CLASSES
# ==========================================

print("\n===== Cluster and Class Comparison =====")

comparison = pd.crosstab(
    data["cluster"],
    data["class"]
)

print("\nK-Means Clusters vs Actual Iris Classes:")
print(comparison)

# Visual comparison
plt.figure(figsize=(8, 5))

sns.heatmap(
    comparison,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("K-Means Clusters vs Actual Iris Classes")
plt.xlabel("Actual Iris Class")
plt.ylabel("K-Means Cluster")
plt.tight_layout()
plt.show()
print("\n===== Final Analysis =====")

print("\nSupervised Learning:")
print("- Uses labelled data with a target variable.")
print("- The model learns to predict the Iris flower class.")
print("- Random Forest achieved 90% accuracy on the test data.")

print("\nUnsupervised Learning:")
print("- Uses the numerical features without class labels.")
print("- K-Means identifies natural groups in the feature data.")
print("- Three clusters were created from the Iris feature patterns.")

print("\nComparison:")
print("- Q1 showed that petal length and petal width provide clear separation between Iris classes.")
print("- These same features also show clear grouping during K-Means clustering.")
print("- Supervised Learning is suitable when labelled data and a known prediction target are available.")
print("- Unsupervised Learning is useful when labels are unavailable and the objective is to discover hidden patterns.")

print("\nReal-World Applications:")

print("\nSupervised Learning:")
print("1. Email spam detection - predicts whether an email is spam or not.")
print("2. Medical diagnosis - predicts disease classes from patient data.")

print("\nUnsupervised Learning:")
print("1. Customer segmentation - groups customers based on purchasing behavior.")
print("2. Anomaly detection - identifies unusual patterns in system or transaction data.")

print("\nConclusion:")
print("The choice between supervised and unsupervised learning depends on")
print("whether labelled data is available and what the AI system is expected to achieve.")