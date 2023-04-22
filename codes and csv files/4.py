"""
4.Execute a linear regression for each independent variable
For each independent variable, execute a linear regression with the dependent variable using a line model 
(y = mx + b). Use the linear_model package from sklearn. What does the slope and R coefficient tell you
 about the relationship between each independent variable and the dependent variable?

 submitted by: Gmon Kuzhiyanikkal
"""

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

# Read data from csv file
data = pd.read_csv('insurance.csv')

# Create the LinearRegression model
reg = LinearRegression()

# Define the independent and dependent variables
x = data['age'].values.reshape(-1, 1)
y = data['charges'].values.reshape(-1, 1)

# Fit the model to the data
reg.fit(x, y)

# Get the slope and R coefficient
slope = reg.coef_
r_coefficient = reg.score(x, y)
print("\n")
print("ages vs Charges")
print("---------------------------------")
print("Slope: ", slope)
print("R Coefficient: ", r_coefficient)
print("\n")

# Plot the data points
plt.scatter(x, y)

# Plot the best fit line
plt.plot(x, reg.predict(x), color='red', linewidth=2)

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Linear Regression of Age vs Charges')

# Show the plot
plt.show()




#charges vs BMI

# Define the independent and dependent variables
x = data['bmi'].values.reshape(-1, 1)
y = data['charges'].values.reshape(-1, 1)

# Fit the model to the data
reg.fit(x, y)

# Get the slope and R coefficient
slope = reg.coef_
r_coefficient = reg.score(x, y)
print("\n")
print("bmi vs Charges")
print("---------------------------------")
print("Slope: ", slope)
print("R Coefficient: ", r_coefficient)
print("\n")

# Plot the data points
plt.scatter(x, y)

# Plot the best fit line
plt.plot(x, reg.predict(x), color='red', linewidth=2)

# Add labels and title
plt.xlabel('bmi')
plt.ylabel('Charges')
plt.title('Linear Regression of BMI vs Charges')

# Show the plot
plt.show()






#children vs charges
# Define the independent and dependent variables
x = data['children'].values.reshape(-1, 1)
y = data['charges'].values.reshape(-1, 1)

# Fit the model to the data
reg.fit(x, y)

# Get the slope and R coefficient
slope = reg.coef_
r_coefficient = reg.score(x, y)
print("\n")
print("children vs Charges")
print("---------------------------------")
print("Slope: ", slope)
print("R Coefficient: ", r_coefficient)
print("\n")

# Plot the data points
plt.scatter(x, y)

# Plot the best fit line
plt.plot(x, reg.predict(x), color='red', linewidth=2)

# Add labels and title
plt.xlabel('children')
plt.ylabel('Charges')
plt.title('Linear Regression of children vs Charges')

# Show the plot
plt.show()

