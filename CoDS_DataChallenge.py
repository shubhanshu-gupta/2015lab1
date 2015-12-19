
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression


# In[9]:

def get_data(file_name):
    train = pd.read_excel("C:/Users/Shubhanshu/Desktop/train.xlsx")
    X_train = train.drop(['ID', 'Salary'], axis=1)
    #Keeping only numeric data
    X_train = X_train._get_numeric_data()
    y_train = train.Salary
    return y_train.shape, X_train.shape



# In[10]:

Y, X = get_data("C:/Users/Shubhanshu/Desktop/train.xlsx")
print Y
print X


# In[12]:

clf = LinearRegression()
clf = clf.fit(X, Y)


# In[13]:

def get_test_data(file_name):
    test = pd.read_excel("C:/Users/Shubhanshu/Desktop/test.xlsx")
    X_test = train.drop(['ID', 'Salary'], axis=1)
    #Keeping only numeric data
    X_test = X_train._get_numeric_data()
    y_test = test.Salary
    return X_test, y_test


# In[15]:

X1, Y1 = get_data("C:/Users/Shubhanshu/Desktop/test.xlsx")
r_sqr = clf.score(X1, Y1)
y_pred = clf.predict(X1)


# In[ ]:



