import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Switch to TkAgg backend for compatibility
import matplotlib.pyplot as plt

# Features are the surface area of the house (in m²)
features = np.array([50, 80, 100, 120, 150, 180, 200])

# Result is the known price for each house (in thousands of dollars)
result = np.array([150, 200, 250, 270, 300, 350, 400])

# Formula y = ax + b
# where a: the slope / b: intercept / y: result / x: features

# The formula to calculate the linear regression is:
# a = sum((xi - x̄) * (yi - ȳ)) / sum((xi - x̄)^2)
# b = ȳ - a * x̄

# x̄ = mean of features
# ȳ = mean of result
n = len(features)  # Length of the features and result array must be the same
x_bar = np.mean(features)  # x̄
y_bar = np.mean(result)  # ȳ

X = features - x_bar
Y = result - y_bar

# Calculate slope (a) and intercept (b)
a = np.sum(X * Y) / np.sum(X**2)
print(f"The value of the slope (a) is: {a:.2f}")

b = y_bar - (a * x_bar)
print(f"The value of the intercept (b) is: {b:.2f}")

# Function to predict the price for a given feature (surface area)
def predict(feature):
    return a * feature + b

# Predicting the price for a house with a surface area of 130 m²
predicted_price = predict(130)
print(f"Predicted price for a house with surface area 130 m² is: ${predicted_price:.2f} thousand")

# Visualization of the data and regression line
X_line = np.linspace(features.min() - 10, features.max() + 10, 100)
y_line = b + a * X_line

# Plotting the data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(features, result, color='blue', s=100, label='Training Data')
plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Area (m²)')
plt.ylabel('Price ($1000)')
plt.title('Simple Linear Regression: House Prices')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# Model evaluation

# Mean Squared Error (MSE)
predictions = predict(features)

mse = np.mean((result - predictions) ** 2)

print(f"Mean Squared Error: {mse:.2f}")

# R² Score

ss_total = np.sum((result - y_bar) ** 2)
ss_residual = np.sum((result - predictions) ** 2)

r2 = 1 - (ss_residual / ss_total)

print(f"R² Score: {r2:.3f}")


