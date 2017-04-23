
# coding: utf-8

# # Setup

# In[136]:

import pandas as pd
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.dummy
import sklearn.linear_model
import sklearn.preprocessing


# # logloss

# ### Binary

# In[42]:

iris = sklearn.datasets.load_iris()


# In[62]:

ind_bin = iris.target < 2
X = iris.data[ind_bin]
y = iris.target[ind_bin]


# In[63]:

sklearn.metrics.log_loss(y,  y)


# In[64]:

sklearn.metrics.log_loss(y,  np.clip(y, 0.05, 0.95))


# In[65]:

sklearn.metrics.log_loss(y,  np.ones_like(y)*y.mean())


# In[86]:

y_duplicated = np.concatenate([y]*5)

sklearn.metrics.log_loss(y_duplicated,  np.ones_like(y_duplicated)*y.mean())


# In[69]:

sklearn.metrics.log_loss(y,  np.random.uniform(low=0.0, high=1.0, size=len(y)))


# In[83]:

y


# In[88]:

def flip(y):
    return -(y-0.5) + 0.5


# In[89]:

y_flip = flip(y)
sklearn.metrics.log_loss(y,  y_flip)


# In[90]:

y_duplicated_flip = flip(y_duplicated)
sklearn.metrics.log_loss(y_duplicated,  y_duplicated_flip)


# In[108]:

mdl = sklearn.dummy.DummyClassifier()
mdl = sklearn.linear_model.LogisticRegression()
mdl = sklearn.svm.SVC(probability=True)


# In[109]:

mdl.fit(X, y)


# In[110]:

pred = mdl.predict(X)
pred_prob = mdl.predict_proba(X)


# In[111]:

sklearn.metrics.log_loss(y, pred)
sklearn.metrics.log_loss(y, pred_prob)


# ### Multiple class

# In[120]:

X = iris.data
y = iris.target


# In[155]:

y_encoded = sklearn.preprocessing.LabelBinarizer().fit_transform(y)
sklearn.metrics.log_loss(y, y_encoded)


# In[177]:

mdl = sklearn.dummy.DummyClassifier()
# mdl = sklearn.linear_model.LogisticRegression()
# mdl = sklearn.svm.SVC(probability=True)


# In[178]:

mdl.fit(X, y)


# In[179]:

pred = mdl.predict(X)
pred_encoded = sklearn.preprocessing.LabelBinarizer().fit_transform(pred)

pred_prob = mdl.predict_proba(X)


# In[180]:

sklearn.metrics.log_loss(y, pred_encoded)


# In[181]:

sklearn.metrics.log_loss(y, pred_prob)

