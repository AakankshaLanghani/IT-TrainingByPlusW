import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "yearly_full_release_long_format.csv"
df = pd.read_csv(file_path)

# Filter for total energy consumption (Demand)
df_energy = df[df["Variable"] == "Demand"].copy()
df_energy = df_energy[['Area', 'Year', 'Value']].dropna()
df_energy.rename(columns={'Area': 'Country', 'Value': 'Energy Consumption (TWh)'}, inplace=True)

# Feature engineering: Create 'Years Since Start'
df_energy["Years Since Start"] = df_energy["Year"] - df_energy["Year"].min()

# Define features and target variable
X = df_energy[["Years Since Start"]]
y = df_energy["Energy Consumption (TWh)"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implementing Gradient Descent for Linear Regression
class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.0001, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def fit(self, X, y):
        m = len(y)
        X_b = np.c_[np.ones((m, 1)), X]  # Add bias term
        self.theta = np.random.randn(X_b.shape[1], 1)

        for _ in range(self.epochs):
            gradients = (2/m) * X_b.T @ (X_b @ self.theta - y.values.reshape(-1, 1))
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

# Train the model using Gradient Descent
gd_model = GradientDescentLinearRegression(learning_rate=0.0001, epochs=10000)
gd_model.fit(X_train, y_train)

# Make predictions
y_pred_gd = gd_model.predict(X_test)

# Evaluate the model
mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)

print(f"Gradient Descent Model - Mean Squared Error: {mse_gd:.2f}")
print(f"Gradient Descent Model - R-squared Score: {r2_gd:.2f}")
