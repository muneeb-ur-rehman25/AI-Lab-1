import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==============================
# 1. DATASET
# ==============================

data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5],
    "Marks": [40, 50, 60, 70, 80]
})

print("Given Dataset:")
print(data)


# ==============================
# 2. MANUAL CALCULATIONS
# ==============================

X = data["Hours"].values
Y = data["Marks"].values

n = len(X)

sum_X = np.sum(X)
sum_Y = np.sum(Y)
sum_XY = np.sum(X * Y)
sum_X2 = np.sum(X ** 2)

mean_X = sum_X / n
mean_Y = sum_Y / n

# Calculate slope
m = (
    (n * sum_XY) - (sum_X * sum_Y)
) / (
    (n * sum_X2) - (sum_X ** 2)
)

# Calculate intercept
c = mean_Y - (m * mean_X)

print("\n===== MANUAL CALCULATIONS =====")
print("Sum of X:", sum_X)
print("Sum of Y:", sum_Y)
print("Sum of XY:", sum_XY)
print("Sum of X^2:", sum_X2)

print("\nMean of X:", mean_X)
print("Mean of Y:", mean_Y)

print("\nSlope (m):", m)
print("Intercept (c):", c)

print("\nRegression Equation:")
print(f"Y = {m}X + {c}")


# ==============================
# 3. PREDICTION FOR 6 HOURS
# ==============================

new_hours = 6

predicted_marks = m * new_hours + c

print("\n===== PREDICTION =====")
print("Study Hours:", new_hours)
print("Predicted Marks:", predicted_marks)


# ==============================
# 4. SKLEARN LINEAR REGRESSION
# ==============================

X_train = X.reshape(-1, 1)
Y_train = Y

model = LinearRegression()

model.fit(X_train, Y_train)

sklearn_slope = model.coef_[0]
sklearn_intercept = model.intercept_

prediction = model.predict([[6]])[0]

print("\n===== SCIKIT-LEARN MODEL =====")
print("Slope:", sklearn_slope)
print("Intercept:", sklearn_intercept)
print("Predicted Marks for 6 Hours:", prediction)


# ==============================
# 5. GRAPH
# ==============================

predicted_values = model.predict(X_train)

plt.figure(figsize=(8, 5))

plt.scatter(
    X,
    Y,
    label="Actual Data"
)

plt.plot(
    X,
    predicted_values,
    label="Regression Line"
)

plt.scatter(
    [6],
    [prediction],
    marker="*",
    s=150,
    label="Prediction (6 Hours)"
)

plt.title("Linear Regression: Hours Studied vs Marks")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()