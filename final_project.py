
# coding: utf-8

# In[1]:

#imports - make sure to run these - if you dont want the graph you can also not import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime 
#import seaborn as sns
#get_ipython().magic('matplotlib inline')


#these next commands just use bash to get the line count of files for subsampling - dont actually use them though
# In[2]:

#get_ipython().system('wc -l ./activity_rec/Phones_accelerometer.csv')


# In[3]:

#get_ipython().system('wc -l ./activity_rec/Phones_gyroscope.csv')


# In[4]:

#get_ipython().system('wc -l ./activity_rec/Watch_accelerometer.csv')


# In[5]:

#get_ipython().system('wc -l ./activity_rec/Watch_gyroscope.csv')


# In[13]:

#the actual line counts of the files
pa_n = 13062476
pg_n = 13932633
wa_n = 3540963
wg_n = 3205432


# In[43]:

#col names in case header is cut off
#header = pd.read_csv('./activity_rec/Watch_accelerometer.csv', header=0, nrows=1).columns


# In[37]:

#below chunk can subsample a file while reading in - was useful for testing, but don't need it anymore
#%time watches_acc = pd.read_csv('./activity_rec/Watch_accelerometer.csv', header=0)
#start = datetime.now()
#sample_size = round(wa_n * 0.001)
#filename = './activity_rec/Watch_accelerometer.csv'
#skip = sorted(random.sample(range(1, wa_n), wa_n - sample_size))

#watches_acc = pd.read_csv(filename, skiprows=skip, parse_dates=[1,2], infer_datetime_format=True, date_parser=dateparse)
#print(datetime.now() - start)


# In[46]:

#filename = './activity_rec/Watch_accelerometer_sample.csv'
#watches_acc = pd.read_csv(filename, parse_dates=[1,2], header=None, names=header)
#watches_acc.head()


# In[21]:

filename = './activity_rec/Watch_accelerometer.csv'
filename2 = './activity_rec/Watch_gyroscope.csv'
filename3 = './activity_rec/Phones_accelerometer.csv'
filename4 = './activity_rec/Phones_gyroscope.csv'

#parses the Unix timestamp in the arrival time column - the one I used for the time windows
def dateparse (time):
    time_p = time[:10] + '.' + time[10:]
    return datetime.fromtimestamp(float(time_p))


# In[204]:

#these commands read in the files - the second one only reads in the columns I ended up using (no index, device, model)

#watches_acc = pd.read_csv(filename, parse_dates=[1], infer_datetime_format=True, date_parser=dateparse, 
                          #names=header)

#watches_acc = pd.read_csv(filename3, parse_dates=[0], infer_datetime_format=True, date_parser=dateparse, 
                          header=0, usecols=[1,3,4,5,9])


# In[203]:

#print(len(watches_acc))
#watches_acc.head()


# In[205]:

#this block does the time subsampling and creates a new dataframe with the relevant features for the acc data
#watches_acc = watches_acc.set_index('Arrival_Time')
#w = watches_acc.resample('2S')
#df1 = w['x'].agg({'x_mean': np.mean, 'x_std': np.std})
#df2 = w['y'].agg({'y_mean': np.mean, 'y_std': np.std})
#df3 = w['z'].agg({'z_mean': np.mean, 'z_std': np.std})
#df4 = w['gt'].max()
#del watches_acc
#result = pd.concat([df1, df2, df3, df4], axis=1)
#result = result.dropna()
#result.head()


# In[207]:

##this block does the time subsampling and creates a new dataframe with the relevant features for the gyr data
#watches_gyr = pd.read_csv(filename4, parse_dates=[0], infer_datetime_format=True, date_parser=dateparse, 
#                          header=0, usecols=[1,3,4,5,9])
#watches_gyr = watches_gyr.set_index('Arrival_Time')
#wg= watches_gyr.resample('2S')
#df1 = wg['x'].agg({'x_mean': np.mean, 'x_std': np.std})
#df2 = wg['y'].agg({'y_mean': np.mean, 'y_std': np.std})
#df3 = wg['z'].agg({'z_mean': np.mean, 'z_std': np.std})
#df4 = wg['gt'].max()
#del watches_gyr 

#result_g = pd.concat([df1, df2, df3, df4], axis=1)
#result_g = result_g.dropna()
#result_g = result_g.drop('gt', axis=1)
#result_g.head()


# In[208]:

#merge the two together
#merge = result.merge(result_g, left_index=True, right_index=True, suffixes=('_acc', '_gyr'))
#merge['gt'].value_counts()


# In[209]:

#save to a file for quick loading in the future 
#merge.to_csv('resampled_phone_acc+gyr.csv', sep=',')


# In[2]:

###########################################


# In[2]:

data = pd.read_csv('resampled_phone_acc+gyr.csv', header=0)
data.head()


# In[21]:

data['gt'].value_counts().plot(kind='bar')


# In[9]:

#get rid of rows that have null as their groundtruth
data = data[data['gt'] != 'null']
data['gt'].value_counts()


# In[30]:

from sklearn.cross_validation import train_test_split

#separate the data - for now just try mean as the features 
y = data['gt'].values
#X = data.drop(['gt', 'Arrival_Time'], axis=1).values
X = data[['x_mean_acc', 'y_mean_acc', 'z_mean_acc', 'x_mean_gyr', 'y_mean_gyr', 'z_mean_gyr']].values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[13]:

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

clf = RandomForestClassifier(n_estimators=10)
scores = cross_validation.cross_val_score(clf, X, y, cv=20)

#clf = clf.fit(X_train, y_train)
#predicted = clf.predict(X_test)
#(len(predicted[predicted==y_test]) / float(len(y_test)))
scores.mean()


# In[75]:

#pd.Series(scores).plot(kind='box')


# In[64]:

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
scores = cross_validation.cross_val_score(gnb, X, y, cv=20)

#gnb = gnb.fit(X_train, y_train)
#predicted = gnb.predict(X_test)
#(len(predicted[predicted==y_test]) / float(len(y_test)))


# In[31]:

from sklearn import svm
kernels = ['rbf', 'linear', 'poly']
for kernel in kernels:
    clf = svm.SVC(kernel=kernel)
    scores = cross_validation.cross_val_score(clf, X, y, cv=20)
    print(kernel, scores.mean())


# In[28]:

#now lets try using mean and std dev 
y = data['gt'].values
X = data.drop(['gt', 'Arrival_Time'], axis=1).values
X.shape


# In[16]:

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

clf = RandomForestClassifier(n_estimators=10)
scores = cross_validation.cross_val_score(clf, X, y, cv=20)
scores.mean()


# In[29]:

from sklearn import svm
kernels = ['rbf', 'linear', 'poly']
for kernel in kernels:
    clf = svm.SVC(kernel=kernel)
    scores = cross_validation.cross_val_score(clf, X, y, cv=20)
    print(kernel, scores.mean())


# In[ ]:




# In[22]:

watches_acc = pd.read_csv(filename, parse_dates=[0], infer_datetime_format=True, date_parser=dateparse, 
                          header=0, usecols=[1,3,4,5,9])


# In[ ]:



