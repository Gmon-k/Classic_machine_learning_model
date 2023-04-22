""""
3.Plot your data
Use the pyplot module from matplotlib to plot your data. Create one plot for each independent 
variable paired with the dependent variable.

submitted by: Gmon Kuzhiyanikkal
"""
#imported the libary
import matplotlib.pyplot as plt
import pandas as pd

#plotting for ages and insurance charges
data = pd.read_csv('insurance.csv')
data.groupby(['age'])['charges'].sum().plot(kind='bar')
plt.xlabel('age')
plt.ylabel('Total Charges')
plt.show()

#plotting for body masss index and insurance charges
data.groupby(['bmi'])['charges'].sum().plot()
plt.xlabel('body mass index')
plt.ylabel('Total Charges')
plt.show()

#plotting for number of children and insurance charges
data.groupby(['children'])['charges'].sum().plot(kind='bar')
plt.xlabel('total no of children')
plt.ylabel('Total Charges')
plt.show()

