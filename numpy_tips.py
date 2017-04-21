
# coding: utf-8

# In[2]:

import numpy as  np


# ### ndarray.size

# In[200]:

a = np.arange(12).reshape(3, 4)

a.shape
a.size


# ### `np.fromfunction()` 

# In[171]:

def f(i, j):
    return i*100 + j

np.fromfunction(f, (3, 4), dtype=int)


# ###  One-dimensional `ndarray[start:end:step]` 

# In[108]:

type(np.arange(10))


# In[102]:

np.arange(100)[0:50:3]


# In[103]:

np.arange(100)[::3]


# In[104]:

np.arange(100)[::-3]


# In[105]:

np.arange(100)[::-1]


# In[107]:

print range(100)[::-1]


# ### Multidimensional ndarray indexing using "`...`"

# In[12]:

def f(i, j, k, m):
    return (i+1)*1000 + (j+1)*100 + (k+1)*10 + (m+1)

b = np.fromfunction(f, (3, 4, 5, 6), dtype=int)
b.shape
b


# In[158]:

b[2]


# In[177]:

b[2, :, :, 3]


# In[178]:

b[2,..., 3]


# ### flatten

# In[201]:

b = np.arange(12).reshape(3, 4)


# In[202]:

b.reshape(b.size)


# In[203]:

b.ravel()


# In[206]:

bf = b.flat
b.flat
for i in bf:
    print i


# ### resize()

# In[213]:

a = np.arange(12).reshape(3, 4)
a
a.resize(4, 3)
a


# ### reshape(i,-1)

# In[8]:

a = np.arange(12).reshape(3, 4)
a
a.reshape(4, -1)
a.reshape(-1,)
a.squeeze()


# ### newaxis

# In[3]:

import numpy as np
from numpy import newaxis


# In[5]:

a = np.array([1, 2])
a


# In[6]:

a.reshape(2, 1)


# In[7]:

a[:, newaxis]


# In[9]:

a[newaxis, :]
a[newaxis, :].shape


# In[10]:

a[:, newaxis, newaxis]
a[:, newaxis, newaxis].shape


# ### Stacking

# In[243]:

np.vstack((np.array([1,2]), np.array([3, 4])))


# In[244]:

np.hstack((np.array([1,2]), np.array([3, 4])))


# In[246]:

np.vstack((np.array([1,2]).reshape(2,1), np.array([3, 4]).reshape(2,1)))


# In[247]:

np.hstack((np.array([1,2]).reshape(2,1), np.array([3, 4]).reshape(2,1)))


# In[248]:

np.column_stack((np.array([1,2]), np.array([3, 4])))


# ### `c_[]`

# In[18]:

a1 = np.arange(10)
a2 = np.arange(10, 20)

b = np.concatenate([a1.reshape(-1,1), a2.reshape(-1,1)], axis=1)
b


# In[17]:

b = np.c_[a1, a2]
b


# ### Assign, view, and copy

# In[249]:

a = np.arange(12)


# In[250]:

b = a
c = a.view()
d = a.copy()


# In[252]:

b is a
c is a
c.base is a


# In[253]:

b.shape = 3, 4


# In[254]:

a.shape


# #### View: share data but different shape

# In[257]:

c.resize(2, 6)
c[0,4]=1234


# In[258]:

c


# In[259]:

a


# ### Indexing by array 

# In[267]:

a = np.arange(12)*2
a


# In[270]:

b = np.array([[1, 3, 7, 5]])
a[b]


# In[271]:

b = np.array([[1, 3], [7, 5]])
a[b]


# #### `mgrid`

# In[274]:

np.mgrid[0:5,0:5]


# In[275]:

np.mgrid[0:5:1,0:5:1]


# In[278]:

np.mgrid[0:5:3j,0:5:10j]


# In[279]:

np.ogrid[0:5, 0:5]


# In[280]:

np.ogrid[0:5:3j,0:5:10j]


# ### `vectorize()`

# In[282]:

def foo(x, y):
    return max(x-y, y-x)


# In[284]:

vec_foo = np.vectorize(foo)


# In[287]:

vec_foo([1, 3, 5], [2, 1, 5])


# ### `cast['type']()`

# In[292]:

type(np.pi)


# In[288]:

np.cast['i'](np.pi)


# In[295]:

np.cast['f'](np.pi)


# ### `select()`

# In[299]:

x = np.arange(10)


# In[300]:

x


# In[308]:

np.select([x<3, x>5], [x, x**2], default=12345)


# ### Use `np.source()` to see sourcecode

# In[310]:

np.source(np.select)


# ### References
# - https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
# - https://docs.scipy.org/doc/scipy/reference/tutorial/basic.html
