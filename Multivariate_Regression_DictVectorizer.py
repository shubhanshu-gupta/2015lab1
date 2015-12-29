
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC


# In[3]:

train = pd.read_excel('C:/Users/Shubhanshu/Desktop/data/train.xlsx')
X_train = train.drop(['ID', 'DOJ', 'DOL', 'Designation', 'JobCity','Salary','DOB'], axis=1)


# In[4]:

x_cat_train = X_train.T.to_dict().values()


# In[5]:

print x_cat_train


# In[5]:

X_train = train.select(['ID', 'DOJ', 'DOL', 'Designation', 'JobCity','Salary','DOB'], axis=1)


# In[6]:

X_train = train.select(['ID'], axis=1)


# In[7]:

train  = pd.read_excel('C:/Users/Shubhanshu/Desktop/data/vd_test.xlsx')
x_cat_train = train.T.to_dict().values()


# In[8]:

print x_cat_train


# In[9]:

dicvec = DV()
X_cat_training = dicvec.fit(train.T.to_dict().values())


# In[10]:

print X_cat_training


# In[13]:

writer = ExcelWriter('C:/Users/Shubhanshu/Desktop/data/output.xlsx')
X_cat_training.to_excel(writer, 'Sheet1')
writer.save()


# In[11]:

len(dicvec.feature_names_)


# In[12]:

dicvec.feature_names_


# In[17]:

vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )


# In[18]:

print vec_x_cat_train


# In[1]:




# In[14]:

train1 = pd.read_excel("C:/Users/Shubhanshu/Desktop/data/train.xlsx")
train1 = train1.drop(['ID', 'DOJ', 'DOL', 'Designation', 'JobCity', 'Gender', 'DOB', '10board', '12board', 'Degree', 'Specialization', 'CollegeState'], axis=1)


# In[15]:

train1.dtypes


# In[28]:

result = pd.concat(pd.DataFrame([train1, vec_x_cat_train]), axis=1, join_axes=[train1.index])


# In[21]:

print train1


# In[23]:

pd.merge(train1, vec_x_cat_train, on='collegeID')


# In[32]:

df = pd.DataFrame(vec_x_cat_train)


# In[33]:

df.to_excel('C:/Users/Shubhanshu/Desktop/data/output.xlsx', 'Sheet1', index=False)


# In[34]:

result = pd.concat([train1, vec_x_cat_train], axis=1, join_axes=[train1.index])


# In[35]:

dicvec


# In[36]:

print dicvec


# In[37]:

df1 = pd.DataFrame(dicvec)
df1


# In[40]:

train1.to_excel('C:/Users/Shubhanshu/Desktop/data/output1.xlsx', 'Sheet1', index=True)


# In[43]:

train2 = pd.read_excel("C:/Users/Shubhanshu/Desktop/data/output1.xlsx")
#train = train.drop(['ID', 'DOJ', 'DOL', 'Designation', 'JobCity', 'Gender', 'DOB', '10board', '12board', 'Degree', 'Specialization', 'CollegeState'], axis=1)
#Keeping only numeric data
X_train = train2.drop(['Salary'], axis=1)
X_train = X_train._get_numeric_data()
y_train = train2.Salary


# In[44]:

clf = LinearRegression()
clf = clf.fit(X_train, y_train)


# In[45]:

' + '.join([format(clf.intercept_, '0.4f')] + map(lambda (f): "(%0.4f)" % (f), zip(clf.coef_)))


# In[47]:

test = pd.read_excel('C:/Users/Shubhanshu/Desktop/data/sg_test.xlsx')
x_cat_test = test.T.to_dict().values()


# In[48]:

print x_cat_test


# In[49]:

vectorizer1 = DV( sparse = False )
vec_x_cat_test = vectorizer1.fit_transform( x_cat_test )


# In[50]:

print vec_x_cat_test


# In[52]:

df_test = pd.DataFrame(vec_x_cat_test)


# In[53]:

df_test.to_excel('C:/Users/Shubhanshu/Desktop/data/test_output.xlsx', 'Sheet1', index=False)


# In[54]:

test1 = pd.read_excel("C:/Users/Shubhanshu/Desktop/data/test.xlsx")
test1 = test1.drop(['ID', 'DOJ', 'DOL', 'Designation', 'JobCity', 'Gender', 'DOB', '10board', '12board', 'Degree', 'Specialization', 'CollegeState'], axis=1)


# In[55]:

test1.to_excel('C:/Users/Shubhanshu/Desktop/data/test_output1.xlsx', 'Sheet1', index=True)


# In[56]:

testing = pd.read_excel('C:/Users/Shubhanshu/Desktop/data/test_output1.xlsx')


# In[57]:

X_testing = testing.drop(['Salary'], axis=1)
#X_testing = X_testing._get_numeric_data()
y_testing = testing.Salary


# In[61]:

y_testing.dtypes


# In[58]:

r_sqr = clf.score(X_testing, y_testing)
y_pred = clf.predict(X_testing)


# In[ ]:



