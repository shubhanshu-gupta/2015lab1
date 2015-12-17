
# coding: utf-8

# In[23]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


# In[24]:

def get_data(file_name):
    data = pd.read_csv("C:\Users\Shubhanshu\Desktop\houserate.csv")
    X_param = []
    Y_param = []
    for single_square_feet, single_price_value in zip(data['square_feet'], data['price']):
        X_param.append([float(single_square_feet)])
        Y_param.append(float(single_price_value))
    return X_param, Y_param
    


# In[25]:

X, Y = get_data('C:\Users\Shubhanshu\Desktop\houserate.csv')
print X
print Y


# In[26]:

def linear_model_main(X_param, Y_param, predict_val):
    regr = linear_model.LinearRegression()  #This is a linear regression object
    regr.fit(X_param, Y_param)
    predict_outcome = regr.predict(predict_val)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions


# In[27]:

X, Y = get_data('C:\Users\Shubhanshu\Desktop\houserate.csv')
predictval = 700
result = linear_model_main(X, Y, predictval)
print "Intercept Value: ", result['intercept']
print "Coefficient Value: ", result['coefficient']
print "Predicted Value: ", result['predicted_value']


# In[28]:

#Function showing the results of linear fit model
def show_linear_line(X_param, Y_param):
    regr = linear_model.LinearRegression()
    regr.fit(X_param, Y_param)
    plt.scatter(X_param, Y_param, color='blue')
    plt.plot(X_param, regr.predict(X_param), color='red', linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.show()


# In[ ]:

show_linear_line(X, Y)


# In[ ]:



