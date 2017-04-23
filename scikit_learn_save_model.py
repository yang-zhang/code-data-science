
# coding: utf-8

# In[6]:

import numpy as np
import sklearn.datasets
import pickle
import xgboost as xgb


# In[7]:

boston = sklearn.datasets.load_boston()
y = boston.target
X = boston.data
clf = xgb.XGBRegressor().fit(X, y)
pickle.dump(clf, open("xgb_boston.pkl", "wb"))
clf2 = pickle.load(open("xgb_boston.pkl", "rb"))
print(np.allclose(clf.predict(X), clf2.predict(X)))

