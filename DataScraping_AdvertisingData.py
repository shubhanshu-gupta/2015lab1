
# coding: utf-8

# In[2]:

from pandas import Series
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from IPython.display import HTML
import numpy as np
import urllib2
import bs4
import time
import operator
import socket
import cPickle
import re

import seaborn as sns
sns.set_context("talk")
sns.set_style("white")


import statsmodels.formula.api as smf

get_ipython().magic(u'matplotlib inline')


# In[5]:

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
data.head()


# In[4]:

data.shape


# In[6]:

# visualize the relationship between the features and the response using scatterplots
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])


# In[6]:

print data.dtypes
print
print data.describe()


# In[7]:

data.head()
data['TV'].head()

columns_that_I_want = ['TV', 'Sales']
print data[columns_that_I_want].head()


# In[8]:

data[(data.Sales > 25)].head()


# In[9]:

print data.head()

grouped_data = data['Sales'].groupby(data['TV'])


# In[10]:

grouped_data = data['Sales'].groupby(data['TV'])
average_sales = grouped_data.mean()

print "Average Sales: "
print average_sales.head()
print



# In[11]:

avg_ratings = grouped_data.apply(lambda f: f.mean())
avg_ratings.head()


# In[12]:

group1 = data['TV'].groupby(data['Sales'])

print group1.head()


# In[13]:

#Fun time
#Let's  scrape the information from job advertisements for data scientists from indeed.com

url = 'http://www.indeed.com/jobs?q=data+scientist&l='
source = urllib2.urlopen(url).read()
bs_tree = bs4.BeautifulSoup(source)


# In[ ]:



