
"""
6.Execute a linear regression with a polynomial model
For one of the numerical independent variables in your data set, find the best fit 3rd order polynomial 
model using multiple linear regression. Build the data frame with the three features manually (linear, 
squared, and cubic versions of the independent variable). Write this function yourself. Do not use the 
sklearn function for generating polynomial features. Plot the best fit polynomial.

submitted by : Gmon Kuzhiyanikkal
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read data from csv file
data = pd.read_csv('insurance.csv')

# Create a new data frame with the three features
x = data[['age']]
x['age_squared'] = x['age'] ** 2
x['age_cubed'] = x['age'] ** 3

y = data['charges'].values.reshape(-1, 1)

# Create the LinearRegression model
reg = LinearRegression()

# Fit the model to the data
reg.fit(x, y)

# Plot the data points
plt.scatter(x['age'], y)

# Plot the best fit polynomial
plt.plot(x['age'], reg.predict(x), color='red', linewidth=2)

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('3rd Order Polynomial Regression of Age vs Charges')

# Show the plot
plt.show()
