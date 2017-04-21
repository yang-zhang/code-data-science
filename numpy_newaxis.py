
# coding: utf-8

# In[45]:

import numpy as np


# In[46]:

x = 1


# In[47]:

y = np.array(x)
y


# In[48]:

y.shape


# In[49]:

y = np.array(x)[np.newaxis]


# In[19]:

np.array(x)[np.newaxis, np.newaxis]


# In[20]:

np.array(x)[np.newaxis, np.newaxis, np.newaxis]


# In[25]:

x = [1, 2, 3]


# In[26]:

np.array(x)


# In[27]:

np.array(x)[np.newaxis]


# In[31]:

np.array(x)[np.newaxis, np.newaxis]


# In[33]:

np.array(x)[:, np.newaxis]


# In[35]:

x = [[1, 2, 3], [4, 5, 6]]


# In[36]:

np.array(x)


# In[38]:

np.array(x)[np.newaxis]


# In[ ]:



