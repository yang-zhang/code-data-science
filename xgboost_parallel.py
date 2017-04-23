
# coding: utf-8

# In[32]:

import os
import time

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston
import xgboost


# In[33]:

rng = np.random.RandomState(31337)

print("Parallel Parameter optimization")


# In[34]:

os.environ["OMP_NUM_THREADS"] = "2"  # or to whatever you want


# In[35]:

os.environ["OMP_NUM_THREADS"] = "3"  # or to whatever you want


# In[36]:

boston = load_boston()

y = boston['target']
X = boston['data']
xgb_model = xgboost.XGBRegressor()
clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],
                               'n_estimators': [50, 100, 200]}, verbose=1,
                   n_jobs=3)
clf.fit(X, y)
print(clf.best_score_)
print(clf.best_params_)


# In[38]:

results = []
num_threads = [1, 2, 3, 4, -1]
for n in num_threads:
    start = time.time()
    model = xgboost.XGBRegressor(nthread=n)
    model.fit(X, y)
    elapsed = time.time() - start
    print(n, elapsed)
    results.append(elapsed)


# In[39]:

results = []
n_jobs = [1, 2, 3, 4, -1]
for n in n_jobs:
    start = time.time()
    clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],
                               'n_estimators': [50, 100, 200]}, verbose=1,
                   n_jobs=n)
    clf.fit(X, y)
    elapsed = time.time() - start
    print(n, elapsed)
    results.append(elapsed)


# References
# - https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_parallel.py
# - http://machinelearningmastery.com/best-tune-multithreading-support-xgboost-python/
