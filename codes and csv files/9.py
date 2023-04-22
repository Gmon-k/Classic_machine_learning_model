"""
Implement multiple linear regression on the projected data
Using the data projected onto the eigenvectors, re-run your multiple linear regression
and examine if the results are different (note: do not use whitening on the projected
data). Were any of your independent variable highly correlated? Which eigenvectors 
are most strong related to the dependent data? What does each eigenvector represent?
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv("insurance.csv")

# Define the dependent variable
y = data["charges"]

# Perform PCA
pca = PCA(n_components=None, whiten=False)
pca.fit(data)
projected_data = pca.transform(data)

# Perform multiple linear regression on the projected data
reg = LinearRegression()
reg.fit(projected_data, y)

# Print the coefficients
print("\n")
print("Coefficients:", reg.coef_)

# Determine which eigenvectors are most strongly related to the dependent variable
eigenvector_importance = np.abs(reg.coef_)
most_important_eigenvectors = np.argsort(eigenvector_importance)[::-1]
print("\n")
print("Most important eigenvectors:", most_important_eigenvectors)
print("\n")



