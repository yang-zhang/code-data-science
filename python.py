
# coding: utf-8

# In[2]:

from ds_utils.imports import *


# ### Plot image data

# In[30]:

plt.imshow(np.random.randn(1000, 1000, 3));


# In[31]:

plt.imshow(np.random.randn(1000, 1000));


# In[32]:

plt.imshow(np.random.randn(1000, 1000), cmap=plt.cm.gray_r);


# In[39]:

from sklearn import datasets
digits = sklearn.datasets.load_digits()
image = digits.images[np.random.choice(digits.images.shape[0])]


# In[40]:

plt.imshow(image)


# In[41]:

plt.imshow(image, cmap=plt.cm.gray_r)


# In[ ]:



