""""
5.Execute a multiple linear regression
Execute multiple linear regression on your data set with all of the independent numerical variables. 
Print out all of the coefficients of the model. Which variable is most strongly related to the dependent 
variable? Which variable is least related to the dependent variable? Do the variables have positive or 
negative correlations?

submitted by : Gmon Kuzhiyanikkal

"""

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

# Read data from csv file
data = pd.read_csv('insurance.csv')

# Define the independent and dependent variables
x = data[['age','bmi','children']]
y = data['charges'].values.reshape(-1, 1)

# Create the LinearRegression model
reg = LinearRegression()

# Fit the model to the data
reg.fit(x, y)

# Print the coefficients of the model
print("\n")
print("['age','bmi','children']")
print( reg.coef_)
print("\n")
