
# coding: utf-8

# In[50]:

import numpy as np


# In[51]:

x = 1


# In[55]:

y = np.array(x)
print(y)
print(y.shape)


# In[56]:

y = np.array(x)[np.newaxis]
print(y)
print(y.shape)


# In[58]:

y = np.array(x)[np.newaxis, np.newaxis]
print(y)
print(y.shape)


# In[59]:

y = np.array(x)[np.newaxis, np.newaxis, np.newaxis]
print(y)
print(y.shape)


# In[61]:

x = [1, 2, 3]


# In[62]:

y = np.array(x)
print(y)
print(y.shape)


# In[63]:

y = np.array(x)[np.newaxis]
print(y)
print(y.shape)


# In[71]:

y = np.array(x)[np.newaxis, np.newaxis]
print(y)
print(y.shape)


# In[74]:

y = np.array(x)[np.newaxis, :, np.newaxis]
print(y)
print(y.shape)


# In[65]:

y = np.array(x)[:, np.newaxis]
print(y)
print(y.shape)


# In[67]:

x = [[1, 2, 3], [4, 5, 6]]


# In[68]:

y = np.array(x)
print(y)
print(y.shape)


# In[69]:

y = np.array(x)[np.newaxis]
print(y)
print(y.shape)


# In[ ]:



