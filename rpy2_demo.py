
# coding: utf-8

# In[ ]:

get_ipython().magic('matplotlib inline')


# # R packages

# ## Importing packages

# In[ ]:

from rpy2.robjects.packages import importr


# In[ ]:

base = importr('base')


# In[ ]:

utils = importr('utils')


# ## Installing packages

# In[2]:

import rpy2.robjects.packages as rpackages


# In[3]:

utils = rpackages.importr('utils')


# In[4]:

utils.chooseCRANmirror(ind=1)


# In[5]:

packnames = ('gbm', 'glmnet')


# In[6]:

from rpy2.robjects.vectors import StrVector


# In[7]:

names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]


# In[8]:

if len(names_to_install)>0:
    utils.install_packages(StrVector(names_to_install))


# In[9]:

importr('glmnet')


# # The r instance

# In[ ]:

from rpy2 import robjects


# In[ ]:

pi = robjects.r('pi')


# In[ ]:

pi


# In[ ]:

type(pi)


# In[ ]:

print pi


# In[ ]:

pi[0]


# In[ ]:

robjects.r('''
        # create a function `f`
        f <- function(r, verbose=FALSE) {
            if (verbose) {
                cat("I am calling f().\n")
            }
            2 * pi * r
        }
        # call the function `f` with argument value 3
        f(3)
        ''')


# In[ ]:

r_f = robjects.globalenv['f']


# In[ ]:

print r_f.r_repr()


# In[ ]:

r_f = robjects.r['f']


# In[ ]:

res = r_f(3)


# In[ ]:

res


# ## Interpolating R objects into R code strings

# In[ ]:

letters = robjects.r['letters']


# In[ ]:

rcode = 'paste(%s, collapse="--")' % (letters.r_repr())


# In[ ]:

res = robjects.r(rcode)


# In[ ]:

print res


# # R vectors

# In[ ]:

res = robjects.StrVector(['abc', 'def'])
print res.r_repr()


# In[ ]:

res = robjects.IntVector([1, 2.8, 3])
print res.r_repr()


# In[ ]:

res = robjects.FloatVector([1.1, 2, 3])
print res.r_repr()


# In[ ]:

[1.1, 2, 3]


# # Calling R functions

# In[ ]:

rsum = robjects.r['sum']


# In[ ]:

rsum


# In[ ]:

rsum(robjects.IntVector([1, 2, 3]))[0]


# In[ ]:

robjects.r.sum(robjects.IntVector([1, 2, 3]))


# # Examples

# ## Graphics and plots

# In[ ]:

r = robjects.r

x = robjects.IntVector(range(10))

print x


# In[ ]:

y = r.rnorm(10)

r.rnorm

r.layout(r.matrix(robjects.IntVector([1,2,3,2]), nrow=2))

r.plot(r.runif(10), y, xlab='runif', ylab='foo/bar', col='red')


# ## Linear models

# R code:
# ```
# ctl <- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
# trt <- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
# group <- gl(2, 10, 20, labels = c("Ctl","Trt"))
# weight <- c(ctl, trt)
# 
# anova(lm.D9 <- lm(weight ~ group))
# 
# summary(lm.D90 <- lm(weight ~ group - 1))# omitting intercept
# ```

# In[ ]:

from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
stats = importr('stats')
base = importr('base')


# In[ ]:

ctl = FloatVector([4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14])
trt = FloatVector([4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69])
group = base.gl(2, 10, 20, labels = ["Ctl","Trt"])
weight = ctl + trt


# In[ ]:

robjects.globalenv["weight"] = weight
robjects.globalenv["group"] = group
lm_D9 = stats.lm("weight ~ group")
print stats.anova(lm_D9)


# In[ ]:

# omitting the intercept
lm_D90 = stats.lm("weight ~ group - 1")
print base.summary(lm_D90)


# In[ ]:

lm_D9.rclass


# In[ ]:

lm_D9


# In[ ]:

lm_D9.r_repr()


# In[ ]:

print lm_D9.rclass


# In[ ]:

print lm_D9.names


# # References: 
# - [Introduction to rpy2](https://rpy2.readthedocs.io/en/version_2.8.x/introduction.html#introduction-to-rpy2)

# In[ ]:



