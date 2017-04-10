
# coding: utf-8

# In[2]:

from ds_utils.imports import *


# ## Plot image data 

# In[3]:

from sklearn import datasets
digits = sklearn.datasets.load_digits()
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)

