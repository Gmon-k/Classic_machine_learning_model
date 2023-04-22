"""
Extensions : Ridge and Lasso.

submitted by : Gmon Kuzhiyanikkal

"""

import pandas as pd
from sklearn.linear_model import Ridge, Lasso

# Load the data
data = pd.read_csv("insurance.csv")

# Define the dependent variable
y = data["charges"]

# Define the independent variables
x = data.drop("age", axis=1)

# Ridge regression with alpha=0.5
ridge_reg = Ridge(alpha=0.5)
ridge_reg.fit(x, y)
ridge_coef = ridge_reg.coef_
print("\n")
print("Ridge coefficients:", ridge_coef)
print("\n")

# Lasso regression with alpha=0.5
lasso_reg = Lasso(alpha=0.5)
lasso_reg.fit(x, y)
lasso_coef = lasso_reg.coef_
print("\n")
print("Lasso coefficients:", lasso_coef)
print("\n")

