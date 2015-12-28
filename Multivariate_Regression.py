
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression


# In[24]:

def get_data(file_name):
    train = pd.read_excel("C:/Users/Shubhanshu/Desktop/data/train.xlsx")
    train = train.drop(['ID', 'DOJ', 'DOL', 'Designation', 'JobCity', 'Gender', 'DOB', '10board', '12board', 'Degree', 'Specialization', 'CollegeState'], axis=1)
    #Keeping only numeric data
    X_train = train.drop(['Salary'], axis=1)
    X_train = X_train._get_numeric_data()
    y_train = train.Salary
    return y_train, X_train


# In[26]:

Y, X = get_data("C:/Users/Shubhanshu/Desktop/data/train.xlsx")
#Y.dtypes
X.dtypes


# In[27]:

clf = LinearRegression()
clf = clf.fit(X, Y)


# In[ ]:




# In[28]:

clf.coef_


# In[29]:

clf.coef_[0]


# In[30]:

def get_test_data(file_name):
    test = pd.read_excel("C:/Users/Shubhanshu/Desktop/data/test.xlsx")
    test = test.drop(['ID', 'DOJ', 'DOL', 'Designation', 'JobCity', 'Gender', 'DOB', '10board', '12board', 'Degree', 'Specialization', 'CollegeState'], axis=1)
    #Keeping only numeric data
    X_test = test.drop(['Salary'], axis=1)
    X_test = X_test._get_numeric_data()
    y_test = test.Salary
    return y_test, X_test


# In[31]:

Y1, X1 = get_test_data("C:/Users/Shubhanshu/Desktop/data/test1.xlsx")
print X1.shape


# In[32]:

r_sqr = clf.score(X1, Y1)
y_pred = clf.predict(X1)


# In[33]:

print r_sqr
print y_pred


# In[ ]:



