
# coding: utf-8

# # Setup

# In[8]:

get_ipython().magic('matplotlib inline')
import pickle
import numpy as np
import pandas as pd

import xgboost as xgb

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics


# # Early stopping 
# https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py
# 

# In[11]:

# Early-stopping
digits = sklearn.datasets.load_digits(2)

X = digits['data']
y = digits['target']
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=0)
clf = xgb.XGBClassifier()
clf.fit(X_train,
        y_train,
        early_stopping_rounds=10,
        eval_metric="auc",
        eval_set=[(X_test, y_test)])


# # Grid search does not support early stopping
# See: https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction/discussion/15235

# In[10]:

# Early-stopping does not support early stopping
boston = sklearn.datasets.load_boston()

y = boston['target']
X = boston['data']
xgb_model = xgb.XGBRegressor()
clf = sklearn.model_selection.GridSearchCV(
    xgb_model, {
        'max_depth': [2, 4, 6],
        'n_estimators': [50, 100, 200],
        'early_stopping_rounds': [10, 100]
    },
    verbose=1)
clf.fit(X, y)
print(clf.best_score_)
print(clf.best_params_)

