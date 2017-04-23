
# coding: utf-8

# In[95]:

import numpy as np


# In[96]:

def show_array(y):
    print('array:', y)
    print('array.ndim:', y.ndim)
    print('array.shape:', y.shape)


# ### 0-D

# In[97]:

x = np.array(5)
show_array(x)


# #### 0-D to 1-D

# In[98]:

y = np.array(x)[np.newaxis]
show_array(y)


# In[99]:

y = np.expand_dims(x, axis=0)
show_array(y)


# Any number >= 0 does the same.

# In[100]:

y = np.expand_dims(x, axis=123456)
show_array(y)


# In[101]:

y = x.reshape(-1,)
show_array(y)


# #### 0-D to 2-D

# In[102]:

y = np.array(x)[np.newaxis, np.newaxis]
show_array(y)


# In[103]:

y = np.expand_dims(x, axis=0)
y = np.expand_dims(y, axis=0)
show_array(y)


# In[104]:

y = x.reshape(-1, 1)
show_array(y)


# ### 1-D

# In[105]:

x = np.array([5, 6, 7])
show_array(x)


# #### 1-D to 2-D

# ##### Vector to row matrix

# In[106]:

y = np.array(x)[np.newaxis, :]
show_array(y)


# In[107]:

y = np.array(x)[np.newaxis] # This is short hand of y = np.array(x)[np.newaxis, :]
show_array(y)


# In[108]:

y = np.expand_dims(x, axis=0)
show_array(y)


# In[109]:

y = x.reshape(1, -1)
show_array(y)


# ##### Vector to column matrix

# In[110]:

y = np.array(x)[:, np.newaxis]
show_array(y)


# In[111]:

y = np.expand_dims(x, axis=1)
show_array(y)


# Any number >= 1 does the same.

# In[112]:

y = np.expand_dims(x, axis=123456)
show_array(y)


# In[113]:

y = x.reshape(-1, 1)
show_array(y)


# ### 2-D

# In[114]:

x = np.array([[1, 2, 3], [4, 5, 6]])
show_array(x)


# #### 2-D to 3-D

# ##### Case 1

# In[115]:

y = np.array(x)[np.newaxis, :, :]
show_array(y)


# In[116]:

y = np.array(x)[np.newaxis, :]
show_array(y)


# In[117]:

y = np.array(x)[np.newaxis]
show_array(y)


# In[118]:

y = np.expand_dims(x, axis=0)
show_array(y)


# In[119]:

y = x.reshape(-1, 2, 3)
show_array(y)


# In[126]:

y = x.reshape(-1, *x.shape)
show_array(y)


# ##### Case 2

# In[121]:

y = np.array(x)[:, np.newaxis, :]
show_array(y)


# In[122]:

y = np.array(x)[:, np.newaxis]
show_array(y)


# In[123]:

y = np.expand_dims(x, axis=1)
show_array(y)


# In[124]:

y = x.reshape(2, 1, 3)
show_array(y)


# In[127]:

y = x.reshape(x.shape[0], -1, x.shape[1])
show_array(y)


# ##### Case 3

# In[24]:

y = np.array(x)[:, :, np.newaxis]
show_array(y)


# In[25]:

y = np.expand_dims(x, axis=2)
show_array(y)


# Any number >= 2 does the same.

# In[26]:

y = np.expand_dims(x, axis=123456)
show_array(y)


# In[128]:

y = x.reshape(*x.shape, -1)
show_array(y)


# In[ ]:



