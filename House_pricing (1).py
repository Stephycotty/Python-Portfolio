#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #data processing, i.e reading the data from CSV file(e.g pd.read_csv)
import matplotlib.pyplot as plt #Matlab-like way of plotting
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# sklearn package for machine learning in python:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,mean_absolute_percentage_error,accuracy_score


# In[2]:


# read the data
df = pd.read_csv('houseprice_data.csv')


# In[3]:


# display the data 
df
df.head(50)
# preview data
df.info()
#view the summary statistics
df.describe


# In[4]:


# select prefered data
x=df.iloc[:,[1,3]].values
y=df.iloc[:,0].values

#splitting the data into training and test
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size= 1/3,
random_state=0)

#fitting data
model = LinearRegression()
model.fit(x_train,y_train)

# The coefficients
print('Coefficients : ', model.coef_)
# The intercept
print('Intercept: ', model.intercept_)
# The mean squared error
print('Mean squared error: %.8f'% mean_squared_error(y_test, model.predict(x_test)))
# The R^2 value:
print('Coefficient of determination: %.2f'% r2_score(y_test, model.predict(x_test)))#


# In[15]:


# 3D visualition for multiple features
bedrooms = df['bedrooms']
sqft_living = df['sqft_living']
price = df['price']

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(bedrooms, sqft_living, price, c='r', marker='o')


ax.set_title('3D Plot Showing Relationship between price, bedrooms, and Sqft living')
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Sqft_living')
ax.set_zlabel('price')
plt.show()

 


# In[ ]:


print(df.corr()['price'].sort_values(ascending=False))


# In[40]:


# select another prefered data
x=df.iloc[:,[9,3]].values
y=df.iloc[:,0].values

#splitting the data into training and test
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size= 1/3,
random_state=0)

#fitting data
model = LinearRegression()
model.fit(x_train,y_train)

# The coefficients
print('Coefficients : ', model.coef_)
# The intercept
print('Intercept: ', model.intercept_)
# The mean squared error
print('Mean squared error: %.8f'% mean_squared_error(y_test, model.predict(x_test)))
# The R^2 value:
print('Coefficient of determination: %.2f'% r2_score(y_test, model.predict(x_test)))


# In[41]:


grade = df['grade']
sqft_living = df['sqft_living']
price = df['price']

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(bedrooms, sqft_living, price, c='r', marker='o')


ax.set_title('3D Plot Showing Relationship between price, grade, and Sqft living')
ax.set_xlabel('garde')
ax.set_ylabel('Sqft_living')
ax.set_zlabel('price')
plt.show()


# In[50]:


#Adding additional features 
x=df.iloc[:,[1,2,3,8,9,14]].values
y=df.iloc[:,0].values

#splitting the data into training and test
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size= 1/3,
random_state=0)

#fitting data
model = LinearRegression()
model.fit(x_train,y_train)

# The coefficients
print('Coefficients : ', model.coef_)
# The intercept
print('Intercept: ', model.intercept_)
# The mean squared error
print('Mean squared error: %.8f'% mean_squared_error(y_test, model.predict(x_test)))
# The R^2 value:
print('Coefficient of determination: %.2f'% r2_score(y_test, model.predict(x_test)))


# In[53]:


#Property Price Predictor
# Assuming you have a list or a single value for features
new_property_features = pd.DataFrame({'bedrooms': [5],'bathrooms': [2],'sqft Living': [1810],'condition': [3],'grade': [7],'zipcode': [98107]})#Replace values with your actual data 

predicted_value = model.predict(new_property_features)
print(f'Predicted Property Value: {predicted_value}')


# In[ ]:




