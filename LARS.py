
# coding: utf-8

# In[221]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs

from sklearn import linear_model


# In[177]:

X, y = [[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111]

X = np.array(X)

y = np.array(y)
y = y.reshape(-1, 1)


# ### Linear regression

# In[178]:

mdl = linear_model.LinearRegression().fit(X, y)
A = mdl.coef_
b = mdl.intercept_

np.allclose(np.dot(X, A.T) + b, y)


# ### Lars in sklearn

# In[179]:

mdl = linear_model.Lars().fit(X, y)
A = mdl.coef_
b = mdl.intercept_

np.allclose(np.dot(X, A) + b, y.T)


# ### Inspect lars

# #### inside lars

# In[186]:

mdl.fit


# In[194]:

y.shape


# In[196]:

linear_model.least_angle.lars_path(X, y)


# In[200]:

Xy=None
Gram=None
max_iter=500
alpha_min=0
method='lar'
copy_X=True,
eps=np.finfo(np.float).eps
copy_Gram=True
verbose=0
return_path=True
return_n_iter=False
positive=False


# In[204]:

n_features = X.shape[1]
n_samples = y.size
max_features = min(max_iter, n_features)

if return_path:
    coefs = np.zeros((max_features + 1, n_features))
    alphas = np.zeros(max_features + 1)
else:
    coef, prev_coef = np.zeros(n_features), np.zeros(n_features)
    alpha, prev_alpha = np.array([0.]), np.array([0.])  # better ideas?

n_iter, n_active = 0, 0
active, indices = list(), np.arange(n_features)
# holds the sign of covariance
sign_active = np.empty(max_features, dtype=np.int8)
drop = False


# In[205]:

return_path


# In[206]:

X.shape


# In[207]:

y.size


# In[212]:

sign_active


# In[214]:

active, indices


# In[222]:

L = np.zeros((max_features, max_features), dtype=X.dtype)
swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (X,))
solve_cholesky, = get_lapack_funcs(('potrs',), (X,))


# In[224]:

Xy, Gram


# In[225]:

copy_X


# In[227]:

X = X.copy('F')


# In[228]:




# In[229]:

if Xy is None:
    Cov = np.dot(X.T, y)


# In[230]:

X.T


# In[231]:

y


# In[232]:

Cov.size


# In[233]:

positive


# In[235]:

if Cov.size:
    if positive:
        C_idx = np.argmax(Cov)
    else:
        C_idx = np.argmax(np.abs(Cov))

    C_ = Cov[C_idx]
    if positive:
        C = C_
    else:
        C = np.fabs(C_)


# In[236]:

method


# In[238]:

return_path


# In[239]:

mdl.coef_


# In[240]:

mdl.coef_path_


# In[ ]:



