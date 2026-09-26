import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris Dataset
iris = load_iris()

data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

data["class"] = iris.target_names[iris.target]

print("===== Iris Dataset =====")

print("Number of Records:", data.shape[0])
print("Number of Features:", data.shape[1])

print("\nFeature Names:")
print(data.columns.tolist())

print("\nData Types:")
print(data.dtypes)

print("\nTarget/Class Variable:")
print("class")

print("\nBasic Statistics:")
print(data.describe())
# NumPy Statistical Analysis
print("\n===== NumPy Statistical Analysis =====")

features = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

for feature in features:
    values = data[feature].to_numpy()

    print("\n", feature)
    print("Mean:", round(np.mean(values), 2))
    print("Minimum:", round(np.min(values), 2))
    print("Maximum:", round(np.max(values), 2))
    print("Standard Deviation:", round(np.std(values), 2))
    # Pandas Data Manipulation
print("\n===== Pandas Data Manipulation =====")

# 1. Selecting relevant columns
selected_columns = data[
    ["sepal length (cm)", "petal length (cm)", "class"]
]

print("\nSelected Columns:")
print(selected_columns.head())

# 2. Filtering records
filtered_data = data[data["petal length (cm)"] > 5.0]

print("\nFiltered Records (Petal Length > 5.0):")
print(filtered_data.head())

# 3. Grouping by Iris class
grouped_data = data.groupby("class")[
    ["sepal length (cm)", "sepal width (cm)",
     "petal length (cm)", "petal width (cm)"]
].mean().round(2)

print("\nAverage Features by Iris Class:")
print(grouped_data)

# 4. Class counts
print("\nNumber of Records in Each Class:")
print(data["class"].value_counts())
# Matplotlib Visualizations

# 1. Sepal Length vs Sepal Width
plt.figure(figsize=(7, 5))

for flower_class in data["class"].unique():
    subset = data[data["class"] == flower_class]
    plt.scatter(
        subset["sepal length (cm)"],
        subset["sepal width (cm)"],
        label=flower_class
    )

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()
plt.tight_layout()
plt.show()


# 2. Petal Length vs Petal Width
plt.figure(figsize=(7, 5))

for flower_class in data["class"].unique():
    subset = data[data["class"] == flower_class]
    plt.scatter(
        subset["petal length (cm)"],
        subset["petal width (cm)"],
        label=flower_class
    )

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.tight_layout()
plt.show()


# 3. Average Petal Length by Iris Class
class_means = data.groupby("class")["petal length (cm)"].mean()

plt.figure(figsize=(7, 5))
class_means.plot(kind="bar")

plt.title("Average Petal Length by Iris Class")
plt.xlabel("Iris Class")
plt.ylabel("Average Petal Length (cm)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# Analysis of Results
print("\n===== Analysis of Results =====")

print("\nPattern 1:")
print("Petal length and petal width show clear differences among Iris classes.")
print("Setosa has much smaller petal measurements, while Virginica generally has the largest.")

print("\nPattern 2:")
print("Sepal measurements show more overlap among Iris classes than petal measurements.")
print("Therefore, petal length and petal width are more useful for distinguishing the classes.")

print("\nUseful Features for Machine Learning:")
print("1. Petal Length")
print("2. Petal Width")
print("3. Sepal Length")
print("4. Sepal Width")

print("\nConclusion:")
print("The analyzed Iris dataset can be used as a foundation for Machine Learning.")
print("Data analysis helps identify useful features and understand class relationships before model training.")
# 4. Pair Plot of Iris Features by Species
import seaborn as sns

plt.figure(figsize=(10, 8))

sns.pairplot(
    data,
    hue="class",
    diag_kind="hist"
)

plt.suptitle("Pair Plot of Iris Features by Species", y=1.02)
plt.show()
# ==========================================
# Additional Visualizations
# ==========================================

# 5. Distribution of Petal Width by Species
plt.figure(figsize=(8, 5))

sns.histplot(
    data=data,
    x="petal width (cm)",
    hue="class",
    kde=True,
    element="step"
)

plt.title("Distribution of Petal Width by Species")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 6. Distribution of Sepal Length by Iris Species
plt.figure(figsize=(8, 5))

sns.histplot(
    data=data,
    x="sepal length (cm)",
    hue="class",
    kde=True,
    element="step"
)

plt.title("Distribution of Sepal Length by Iris Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ==========================================
# Machine Learning - Confusion Matrices
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Prepare features and target
X = data[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
]

y = data["class"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 7. Random Forest Confusion Matrix
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)

disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=random_forest.classes_
)

disp_rf.plot()

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()


# 8. General Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=random_forest.classes_
)

disp.plot()

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()