
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


# In[2]:

data = pd.read_csv('C:/Users/Shubhanshu/Desktop/CS109_Shubhanshu/2015lab1/AirPassengers.csv')


# In[3]:

print data.head()


# In[4]:

print '\n Data Types:'


# In[5]:

print '\n Data Types:'
print data.dtypes


# In[6]:

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')


# In[7]:

data = pd.read_csv('C:/Users/Shubhanshu/Desktop/CS109_Shubhanshu/2015lab1/AirPassengers.csv', parse_dates='Month', index_col='Month',
                   date_parser=dateparse)


# In[8]:

print data.head()


# In[9]:

data.index


# In[10]:

ts = data['#Passengers']
ts.head(10)


# In[11]:

ts['1949']


# In[12]:

#Testing the stationarity of our TS


# In[13]:

plt.plot(ts)


# In[14]:

from statsmodels.tsa.stattools import adfuller


# In[15]:

rolmean = pd.rolling_mean(ts, window=12)


# In[16]:

rolmean


# In[17]:

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput


# In[18]:

test_stationarity(ts)


# In[19]:

ts_log = np.log(ts)


# In[20]:

plt.plot(ts_log)


# In[21]:

moving_avg = pd.rolling_mean(ts_log,12)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')


# In[22]:

ts_log_moving_avg_diff = ts_log - moving_avg


# In[23]:

ts_log_moving_avg_diff.head(12)


# In[24]:

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)


# In[25]:

#the test statistic is smaller than the 5% critical values so we can say with 95% confidence that this is a stationary series.


# In[26]:

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)


# In[27]:

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)


# In[28]:

#In the case above, TS is stationary with 90% confidence.


# In[29]:

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

print decomposition


# In[30]:

trend = decomposition.trend


# In[31]:

trend


# In[32]:

seasonal = decomposition.seasonal
residual = decomposition.resid


# In[33]:

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# In[34]:

#Here the trend, seasonality are separated out from data and we can model the residuals.


# In[35]:

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)


# In[36]:

#Test statistic is even lower than 1% of the critical value. Hence, it's probably stationary now.


# In[ ]:



