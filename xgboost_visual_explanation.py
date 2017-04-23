
# coding: utf-8

# # Setup

# In[24]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd

import xgboost 

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics


# # [Example 1](http://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/)

# In[25]:

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
           header=None)
data = np.array(data)

X_train = data[:, :-1]

y_train = data[:, -1]

mdl = xgboost.XGBClassifier()

mdl.fit(X_train, y_train)


# In[26]:

xgboost.plot_importance(mdl)


# In[27]:

xgboost.plot_tree(mdl)


# # Example 2: Iris

# In[28]:

data = sklearn.datasets.load_iris()

X_train = data.data

y_train = data.target

mdl = xgboost.XGBClassifier()

mdl.fit(X_train, y_train)


# In[29]:

xgboost.plot_tree(mdl)


# In[30]:

xgboost.plot_importance(mdl)

