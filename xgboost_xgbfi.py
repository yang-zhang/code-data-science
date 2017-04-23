
# coding: utf-8

# In[3]:

import sklearn.datasets
import sklearn.model_selection
import xgboost
import xgbfir


# In[59]:

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target

mdl = xgboost.XGBClassifier().fit(X, y)
xgbfir.saveXgbFI(mdl, feature_names=iris.feature_names, OutputXlsxFile = 'irisFI.xlsx')


# In[62]:

boston = sklearn.datasets.load_boston()
X = boston.data
y = boston.target

mdl = xgboost.XGBRegressor().fit(X, y)
xgbfir.saveXgbFI(mdl, feature_names=boston.feature_names, OutputXlsxFile = 'bostonFI.xlsx')


# In[ ]:

boston = sklearn.datasets.load_boston()
X = boston.data
y = boston.target

mdl = sklearn.model_selection.GridSearchCV(
    estimator = xgboost.XGBRegressor(),
    param_grid = {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]},
)
mdl.fit(X, y)

xgbfir.saveXgbFI(mdl.best_estimator_, feature_names=boston.feature_names, OutputXlsxFile = 'bostonFI_grid.xlsx')


# In[8]:

ls *.xlsx


# References
# - https://github.com/limexp/xgbfir
# - http://projects.rajivshah.com/blog/2016/08/01/xgbfi/
