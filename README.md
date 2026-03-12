# Simple Linear Regression

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A pure Python implementation of simple linear regression using NumPy, converted from MATLAB university coursework. This project demonstrates the mathematical foundation of linear regression without relying on high-level ML libraries.

## 📚 Background

This project is a conversion of university MATLAB coursework into Python, implementing simple linear regression from scratch using NumPy to understand the underlying mathematics.

## 📊 Overview

This project implements the simple linear regression algorithm from scratch:

**Formula:** `y = ax + b`

Where:
- `a` = slope (coefficient)
- `b` = y-intercept
- `x` = feature (house surface area in m²)
- `y` = target (house price in thousands $)

## 🚀 Features

- ✅ Pure NumPy implementation (no scikit-learn)
- ✅ Manual calculation of regression coefficients
- ✅ Prediction function for new data 
- ✅ Code (converted from MATLAB)

## 📖 Usage

from linear_regression import predict

# Predict price for 130m² house
price = predict(130)
print(f"Predicted price: {price:.2f} thousand $")

🤝 Contributing
Contributions are welcome! Feel free to:

    Report bugs
    Suggest enhancements
    Add more features (multiple regression, R² calculation, etc.)

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

    Original MATLAB code from university coursework
    Inspired by basic machine learning mathematics courses
    Converted to Python for learning purposes

Author: N1N0u
Repository: https://github.com/N1N0u/Simple-Linear-regresson
