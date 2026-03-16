import numpy as np
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
matplotlib.use('TkAgg')  # Switch to TkAgg backend for compatibility
import matplotlib.pyplot as plt

# =============================================================================
# SIMPLE LINEAR REGRESSION FROM SCRATCH
# =============================================================================
# This implementation demonstrates the mathematical foundation of linear regression
# without using machine learning libraries, showing exactly how the algorithm works


# Features are the surface area of the house (in m²)
# These are our independent variables (X) - what we use to make predictions
features = np.array([50, 80, 100, 120, 150, 180, 200])

# Result is the known price for each house (in thousands of dollars)
# These are our dependent variables (y) - what we want to predict
result = np.array([150, 200, 250, 270, 300, 350, 400])

print(f"{'Feature (m²)':>12} | {'Price ($1000)':>12}")
print("-" * 27)

for f, r in zip(features, result):
    print(f"{f:12} | {r:12}")

# LINEAR REGRESSION FORMULA: y = ax + b
# Where:
#   a = slope (coefficient) - represents price increase per m²
#   b = intercept - base price when area is 0 (theoretical)
#   y = predicted price
#   x = house surface area

# =============================================================================
# STEP 1: CALCULATE MEANS (CENTROID OF DATA)
# =============================================================================
# x̄ (x_bar) = mean of all feature values (average house size)
# ȳ (y_bar) = mean of all result values (average house price)
# The regression line always passes through the point (x̄, ȳ)
n = len(features)  # Number of data points (must match between X and y)
x_bar = np.mean(features)  # Calculate mean of features: x̄ = Σx / n
y_bar = np.mean(result)  # Calculate mean of results: ȳ = Σy / n

print(f"Number of data points: {n}")
print(f"Mean house area (x̄): {x_bar:.2f} m²")
print(f"Mean house price (ȳ): {y_bar:.2f} thousand $")

# =============================================================================
# STEP 2: CALCULATE DEVIATIONS FROM MEAN
# =============================================================================

X = features - x_bar  # Xi = xi - x̄ (deviation of each area from mean)
Y = result - y_bar  # Yi = yi - ȳ (deviation of each price from mean)

print(f"\nDeviations from mean calculated")
print(f"X deviations (first 3): {X[:3]}")
print(f"Y deviations (first 3): {Y[:3]}")

# =============================================================================
# STEP 3: CALCULATE SLOPE (a) USING LEAST SQUARES METHOD
# =============================================================================
# Formula: a = Σ[(xi - x̄)(yi - ȳ)] / Σ[(xi - x̄)²]

a = np.sum(X * Y) / np.sum(X ** 2)
print(f"\n{'=' * 50}")
print(f"SLOPE (a): {a:.4f}")
print(f"Interpretation: For every 1 m² increase, price increases by ${a:.2f}k")
print(f"{'=' * 50}")

# =============================================================================
# STEP 4: CALCULATE INTERCEPT (b)
# =============================================================================
# Formula: b = ȳ - a * x̄

b = y_bar - (a * x_bar)
print(f"INTERCEPT (b): {b:.4f}")
print(f"Interpretation: Base price (theoretical at 0 m²): ${b:.2f}k")
print(f"{'=' * 50}")

print(f"\nFinal Equation: Price = {a:.4f} * Area + {b:.4f}")


# =============================================================================
# STEP 5: PREDICTION FUNCTION
# =============================================================================
def predict(feature):
    """
    Predict house price based on surface area using our trained model.
    """
    return a * feature + b


# Example prediction for a 130 m² house
predicted_price = predict(130)
print(f"\n{'=' * 50}")
print(f"PREDICTION EXAMPLE")
print(f"{'=' * 50}")
print(f"House area: 130 m²")
print(f"Predicted price: ${predicted_price:.2f} thousand")
print(f"Calculation: {a:.4f} * 130 + {b:.4f} = {predicted_price:.2f}")

# =============================================================================
# STEP 6: VISUALIZATION (Uncomment This if you want to see the plot from scratch)
# =============================================================================
# X_line = np.linspace(features.min() - 10, features.max() + 10, 100)
# y_line = b + a * X_line  # y = ax + b for all points in the line

# Create the plot
# plt.figure(figsize=(12, 7))

# Plot training data points
# plt.scatter(features, result, color='blue', s=100, label='Training Data (Actual Prices)',
#             zorder=5, edgecolors='darkblue', linewidth=1.5)

# Plot regression line
# plt.plot(X_line, y_line,color='red',linewidth=2.5,label=f'Regression Line: y = {a:.2f}x + {b:.2f}',
#          zorder=3)

# Mark the prediction point
# plt.scatter([130], [predicted_price],
#             color='green',
#             s=150,
#             marker='*',
#             label=f'Prediction: 130m² = ${predicted_price:.0f}k',
#             zorder=6,
#             edgecolors='darkgreen',
#             linewidth=2)

# Mark the centroid (mean point)
# plt.scatter([x_bar], [y_bar],
#             color='orange',
#             s=100,
#             marker='D',
#             label=f'Centroid (x̄={x_bar:.1f}, ȳ={y_bar:.1f})',
#             zorder=5,
#             edgecolors='darkorange',
#             linewidth=1.5)

# Add labels and formatting
# plt.xlabel('Surface Area (m²)', fontsize=12, fontweight='bold')
# plt.ylabel('Price ($1000)', fontsize=12, fontweight='bold')
# plt.title('Simple Linear Regression: House Price Prediction\n(From Scratch Implementation)',
#           fontsize=14, fontweight='bold', pad=20)
# plt.legend(loc='upper left', fontsize=10, framealpha=0.9)
# plt.grid(True, alpha=0.3, linestyle='--')
# plt.tight_layout()
# plt.show()

# =============================================================================
# STEP 7: MODEL EVALUATION METRICS
# =============================================================================
print(f"\n{'=' * 60}")
print(f"MODEL EVALUATION METRICS")
print(f"{'=' * 60}")

# Get predictions for all training data
predictions = predict(features)

# --- MEAN SQUARED ERROR (MSE) ---
# Lower is better (0 = perfect prediction)
# Formula: MSE = (1/n) * Σ(y_actual - y_predicted)²
mse = np.mean((result - predictions) ** 2)
print(f"\n1. MEAN SQUARED ERROR (MSE): {mse:.4f}")
print(f"   - Average squared error: ${mse:.2f}k²")
print(f"   - RMSE (root): ${np.sqrt(mse):.2f}k (average error in original units)")

# --- R² SCORE (COEFFICIENT OF DETERMINATION) ---
# Range: 0 to 1 (1 = perfect fit, 0 = no better than mean)
# Formula: R² = 1 - (SS_residual / SS_total)
#   SS_total = Σ(y - ȳ)² (total variance in data)
#   SS_residual = Σ(y - y_pred)² (variance not explained by model)

ss_total = np.sum((result - y_bar) ** 2)  # Total sum of squares
ss_residual = np.sum((result - predictions) ** 2)  # Residual sum of squares

r2 = 1 - (ss_residual / ss_total)
print(f"\n2. R² SCORE: {r2:.6f} ({r2 * 100:.2f}%)")
print(f"   - SS_total (total variance): {ss_total:.2f}")
print(f"   - SS_residual (unexplained): {ss_residual:.2f}")
print(f"   - Variance explained: {(1 - ss_residual / ss_total) * 100:.2f}%")
print(f"   - Interpretation: Model explains {r2 * 100:.2f}% of price variation")

# --- ADDITIONAL METRICS ---
mae = np.mean(np.abs(result - predictions))  # Mean Absolute Error
print(f"\n3. MEAN ABSOLUTE ERROR (MAE): {mae:.4f}")
print(f"   - Average absolute error: ${mae:.2f}k")

print(f"\n{'=' * 60}")


# =============================================================================
# STEP 8: COMPARISON WITH SCIKIT-LEARN IMPLEMENTATION
# =============================================================================
print(f"\n{'=' * 60}")
print("SCIKIT-LEARN COMPARISON")
print(f"{'=' * 60}")

# Scikit-learn expects 2D input for features
X_sklearn = features.reshape(-1, 1)
y_sklearn = result

# Create model
sk_model = LinearRegression()

# Train model
sk_model.fit(X_sklearn, y_sklearn)

# Extract learned parameters
sk_a = sk_model.coef_[0]       # slope
sk_b = sk_model.intercept_     # intercept

print("\nScikit-learn Model Parameters:")
print(f"Slope (a): {sk_a:.4f}")
print(f"Intercept (b): {sk_b:.4f}")

# Predictions
sk_predictions = sk_model.predict(X_sklearn)

# Metrics
sk_mse = mean_squared_error(y_sklearn, sk_predictions)
sk_r2 = r2_score(y_sklearn, sk_predictions)
sk_mae = mean_absolute_error(y_sklearn, sk_predictions)

print("\nScikit-learn Metrics:")
print(f"MSE: {sk_mse:.4f}")
print(f"RMSE: {np.sqrt(sk_mse):.4f}")
print(f"MAE: {sk_mae:.4f}")
print(f"R² Score: {sk_r2:.6f}")

# =============================================================================
# PARAMETER COMPARISON
# =============================================================================
print(f"\n{'=' * 60}")
print("PARAMETER COMPARISON")
print(f"{'=' * 60}")

print(f"Slope difference: {abs(a - sk_a):.10f}")
print(f"Intercept difference: {abs(b - sk_b):.10f}")

print("\nConclusion:")
print("differences are zero")

# =============================================================================
# STEP 9: COMBINED VISUALIZATION (FROM SCRATCH vs SCIKIT-LEARN)
# =============================================================================

# Prepare line values
X_line = np.linspace(features.min() - 10, features.max() + 10, 100)
y_line_manual = a * X_line + b
y_line_sklearn = sk_model.predict(X_line.reshape(-1,1))

# Create two plots in the same figure
fig, axes = plt.subplots(1, 2, figsize=(16,7))

# ==========================================================
# PLOT 1 — FROM SCRATCH MODEL
# ==========================================================
axes[0].scatter(features, result,
                color='blue',
                s=100,
                label='Training Data',
                edgecolors='darkblue')

axes[0].plot(X_line,
             y_line_manual,
             color='red',
             linewidth=2.5,
             label=f'y = {a:.2f}x + {b:.2f}')

axes[0].scatter([130], [predicted_price],
                color='green',
                s=150,
                marker='*',
                label=f'Prediction: ${predicted_price:.0f}k')

axes[0].scatter([x_bar], [y_bar],
                color='orange',
                s=100,
                marker='D',
                label='Centroid')

axes[0].set_title("From Scratch Linear Regression")
axes[0].set_xlabel("Surface Area (m²)")
axes[0].set_ylabel("Price ($1000)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.3)

# ==========================================================
# PLOT 2 — SCIKIT-LEARN MODEL
# ==========================================================
axes[1].scatter(features, result,
                color='blue',
                s=100,
                label='Training Data',
                edgecolors='darkblue')

axes[1].plot(X_line,
             y_line_sklearn,
             color='purple',
             linewidth=2.5,
             label=f'y = {sk_a:.2f}x + {sk_b:.2f}')

sk_predicted_price = sk_model.predict(np.array([[130]]))[0]

axes[1].scatter([130], [sk_predicted_price],
                color='green',
                s=150,
                marker='*',
                label=f'Prediction: ${sk_predicted_price:.0f}k')

axes[1].scatter([x_bar], [y_bar],
                color='orange',
                s=100,
                marker='D',
                label='Centroid')

axes[1].set_title("Scikit-learn Linear Regression")
axes[1].set_xlabel("Surface Area (m²)")
axes[1].set_ylabel("Price ($1000)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.3)

plt.suptitle("Linear Regression Comparison: From Scratch vs Scikit-learn", fontsize=16)
plt.tight_layout()

plt.show()