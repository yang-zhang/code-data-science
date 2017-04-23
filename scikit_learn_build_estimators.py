
# coding: utf-8

# ## Check if estimator adhere to sklearn interface using `check_estimator`

# In[1]:

from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import LinearSVC
check_estimator(LinearSVC)


# ## Templates provided by [scikit-learn-contrib project](https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/template.py)

# In[2]:

import sys
sys.path.insert(1, '/Users/yangzhang/git/project-template/skltemplate/')
from template import TemplateEstimator
check_estimator(TemplateEstimator)


# ## Use template on data

# In[3]:

from sklearn.datasets import load_iris
iris = load_iris()
X_train = iris.data
y_train = iris.target


# In[4]:

from template import TemplateClassifier


# In[5]:

mdl = TemplateClassifier()
mdl.fit(X_train, y_train)
mdl.predict(X_train)


# ## Use template in cross validation

# In[7]:

import sklearn.model_selection
mdl = TemplateClassifier()
sklearn.model_selection.cross_val_score(mdl, X_train, y_train, cv=5)


# ## Use template in a pipeline

# In[8]:

import sklearn.pipeline
import sklearn.preprocessing
pipeline = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.StandardScaler(),    
    TemplateClassifier(),
)


# In[10]:

pipeline.fit(X_train, y_train)


# In[11]:

pipeline.predict(X_train)


# ## Use template in a pipeline in cross validation

# In[12]:

sklearn.model_selection.cross_val_score(pipeline, X_train, y_train, cv=5)


# ## References
# - http://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator
# - http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
