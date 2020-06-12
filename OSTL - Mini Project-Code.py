#!/usr/bin/env python
# coding: utf-8

# In[1]:


#input file voice.csv in /input/ directory inside project folder
#running this command to list 
from subprocess import check_output
print(check_output(["ls", "input"]).decode("utf8"))


# In[2]:


#importing reqired libraries
import pandas as p
import numpy as n
import seaborn as s

from sklearn.preprocessing import LabelEncoder #encoding
from sklearn.preprocessing import StandardScaler #standardisation
from sklearn.model_selection import train_test_split #train/test split
from sklearn.model_selection import cross_val_score #K-fold cross  validation

#SVM libraries
from sklearn.svm import SVC
from sklearn import metrics

from sklearn.model_selection import GridSearchCV #to find best parameter

import matplotlib.pyplot as m
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#reading csv file into dataframe
dataframe = p.read_csv('input/voice.csv')
dataframe.head()


# In[4]:


#correalation
dataframe.corr()


# In[5]:


dataframe.shape
#Features = 21, Instances = 3168


# In[6]:


print("Male samples = {}".format(dataframe[dataframe.label == 'male'].shape[0]))
print("Female samples = {}".format(dataframe[dataframe.label == 'female'].shape[0]))
#There are equal number of samples


# In[7]:


#Encoding features
X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]

encode = LabelEncoder()
y = encode.fit_transform(y)
y
#male = 1
#female = 0


# In[8]:


#Standardization of datasets is a common requirement for many machine learning estimators implemented 
#in scikit-learn; they might behave badly if the individual features do not more or less look like 
#standard normally distributed data.
scale = StandardScaler()
scale.fit(X)
X = scale.transform(X)


# In[9]:


#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[10]:


#Default hyperparameters
svc = SVC()
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
print('Accuracy of default hyperparameters :')
print(metrics.accuracy_score(y_test, y_predict))


# In[11]:


#Default linear kernal
svc=SVC(kernel='linear')
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
print('Accuracy of deafult linear kernal :')
print(metrics.accuracy_score(y_test, y_predict))


# In[12]:


#Default RBF Kernel
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
print('Accuracy of deafult RBF kernal:')
print(metrics.accuracy_score(y_test, y_predict))


# In[13]:


#Default Polynomial Kernel
svc = SVC(kernel='poly')
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
print('Accuracy of deafult polynomial kernal:')
print(metrics.accuracy_score(y_test, y_predict))


# In[14]:


#K-fold cross validation is a procedure used to estimate the skill of the model on new data.
#Its a resampling procedure used to evaluate machine learning models on a limited data sample.


# In[15]:


#Cross Validation on Linear Kernal
svc = SVC(kernel = 'linear')
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
print(scores.mean())


# In[16]:


#Cross Validation on RBF Kernal
svc = SVC(kernel = 'rbf')
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
print(scores.mean())


# In[17]:


#Cross Validation on Polynomial Kernal
svc = SVC(kernel = 'poly')
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
print(scores.mean())


# In[18]:


#Checking accuracy of kenal as linear with values of C
C_range = list(range(1,26))
acc = []
for c in C_range:
    svc = SVC(kernel = 'linear', C = c)
    scores = cross_val_score(svc, X, y, cv = 10, scoring = 'accuracy')
    acc.append(scores.mean())
print(acc) 


# In[19]:


C_values = list(range(1,26))

m.plot(C_values,acc)
m.xticks(n.arange(0,27,2))
m.xlabel('Value of C for SVC')
m.ylabel('Cross-Validated Accuracy')


# In[20]:


C_range = list(n.arange(0.1,6,0.1))
acc = []
for c in C_range:
    svc = SVC(kernel = 'linear', C = c)
    scores = cross_val_score(svc, X, y, cv = 10, scoring='accuracy')
    acc.append(scores.mean())
print(acc)  


# In[21]:


C_values=list(n.arange(0.1,6,0.1))

m.plot(C_values,acc)
m.xticks(n.arange(0.0,6,0.3))
m.xlabel('Value of C for SVC ')
m.ylabel('Cross-Validated Accuracy')


# In[ ]:


#C gives highest accuracy at C = 0.1


# In[22]:


#Taking polynomial kernel with different degree
degree = [2,3,4,5,6]
acc = []
for d in degree: 
    svc = SVC(kernel = 'poly', degree = d)
    scores = cross_val_score(svc, X, y, cv = 10, scoring = 'accuracy')
    acc.append(scores.mean())
print(acc)   


# In[23]:


degree=[2,3,4,5,6]

m.plot(degree,acc,color='r')
m.xlabel('degrees for SVC ')
m.ylabel('Cross-Validated Accuracy')


# In[ ]:


#degree gives hishest accuracy at degree = 3.0


# In[24]:


#performing SVM by taking hyperparameter C=0.1 and kernel as linear
svc = SVC(kernel = 'linear', C=0.1)
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
acc = metrics.accuracy_score(y_test, y_predict)
print(acc)


# In[25]:


#With K-fold cross validation(where K=10)
svc = SVC(kernel = 'linear', C = 0.1)
scores = cross_val_score(svc, X, y, cv = 10, scoring = 'accuracy')
print(scores.mean())


# In[26]:


#performing SVM by taking hyperparameter gamma=0.01 and kernel as rbf
svc = SVC(kernel='rbf', gamma=0.01)
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
metrics.accuracy_score(y_test, y_predict)


# In[27]:


#With K-fold cross validation(where K=10)
svc = SVC(kernel = 'linear', gamma = 0.01)
scores = cross_val_score(svc, X, y, cv = 10, scoring = 'accuracy')
print(scores.mean())


# In[28]:


#performing SVM by taking hyperparameter degree=3 and kernel as poly
svc = SVC(kernel = 'poly', degree = 3)
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
acc = metrics.accuracy_score(y_test, y_predict)
print(acc)


# In[29]:


#With K-fold cross validation(where K=10)
svc = SVC(kernel = 'poly', degree = 3)
scores = cross_val_score(svc, X, y, cv = 10, scoring = 'accuracy')
print(scores.mean())


# In[30]:


#performing Grid search technique to find the best parameter
svm = SVC()
tuned_parameters = {
 'C': (n.arange(0.1,1,0.1)) , 'kernel': ['linear'],
 'C': (n.arange(0.1,1,0.1)) , 'gamma': [0.01,0.02,0.03,0.04,0.05], 'kernel': ['rbf'],
 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05], 'C':(n.arange(0.1,1,0.1)) , 'kernel':['poly']
                   }


# In[32]:


model = GridSearchCV(svm, tuned_parameters,cv=10,scoring='accuracy')


# In[33]:


model.fit(X_train, y_train)
print(model.best_score_)


# In[34]:


print(model.best_params_)


# In[35]:


y_pred= model.predict(X_test)
print(metrics.accuracy_score(y_pred, y_test))


# In[ ]:




