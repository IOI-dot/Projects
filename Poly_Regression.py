import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("StudentPerformanceFactors.csv")
# Scatter plot of Study Hours vs Score
plt.scatter(df['Hours_Studied'], df['Exam_Score'])
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()

# Encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)
print(df_encoded.columns)
# Define features and target
X = df_encoded.drop(columns=['Exam_Score','Distance_from_Home_Moderate','Distance_from_Home_Near'])
y = df_encoded['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Visualization with respect to Hours_Studied
plt.scatter(X_test['Hours_Studied'], y_test, color='blue', label="Actual")
plt.scatter(X_test['Hours_Studied'], y_pred, color='red', label="Predicted")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Actual vs Predicted Exam Scores (by Study Hours)")
plt.legend()
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Transform features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train-test split
X_train_poly, X_test_poly, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# Train model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predictions
y_poly_pred = poly_model.predict(X_test_poly)

# Compare performance
print("Linear R2:", r2_score(y_test, y_pred))
print("Polynomial R2:", r2_score(y_test, y_poly_pred))