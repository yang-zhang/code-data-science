
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

def show_array(y):
    print('array:', y)
    print('array.ndim:', y.ndim)
    print('array.shape:', y.shape)


# ### 0-D

# In[3]:

x = np.array(5)
show_array(x)


# #### 0-D to 1-D

# In[4]:

y = np.array(x)[np.newaxis]
show_array(y)


# In[5]:

y = np.expand_dims(x, axis=0)
show_array(y)


# Any number >= 0 does the same.

# In[6]:

y = np.expand_dims(x, axis=123456)
show_array(y)


# #### 0-D to 2-D

# In[7]:

y = np.array(x)[np.newaxis, np.newaxis]
show_array(y)


# In[8]:

y = np.expand_dims(x, axis=0)
y = np.expand_dims(y, axis=0)
show_array(y)


# ### 1-D

# In[9]:

x = np.array([5, 6, 7])
show_array(x)


# #### 1-D to 2-D

# ##### Vector to row matrix

# In[10]:

y = np.array(x)[np.newaxis, :]
show_array(y)


# In[11]:

y = np.array(x)[np.newaxis] # This is short hand of y = np.array(x)[np.newaxis, :]
show_array(y)


# In[12]:

y = np.expand_dims(x, axis=0)
show_array(y)


# ##### Vector to column matrix

# In[13]:

y = np.array(x)[:, np.newaxis]
show_array(y)


# In[14]:

y = np.expand_dims(x, axis=1)
show_array(y)


# Any number >= 1 does the same.

# In[15]:

y = np.expand_dims(x, axis=123456)
show_array(y)


# ### 2-D

# In[16]:

x = np.array([[1, 2, 3], [4, 5, 6]])
show_array(x)


# #### 2-D to 3-D

# ##### Case 1

# In[17]:

y = np.array(x)[np.newaxis, :, :]
show_array(y)


# In[18]:

y = np.array(x)[np.newaxis, :]
show_array(y)


# In[19]:

y = np.array(x)[np.newaxis]
show_array(y)


# In[20]:

y = np.expand_dims(x, axis=0)
show_array(y)


# ##### Case 2

# In[21]:

y = np.array(x)[:, np.newaxis, :]
show_array(y)


# In[22]:

y = np.array(x)[:, np.newaxis]
show_array(y)


# In[23]:

y = np.expand_dims(x, axis=1)
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

