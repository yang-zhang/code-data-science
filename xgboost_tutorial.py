
# coding: utf-8

# # Setup

# In[1]:

get_ipython().magic('matplotlib inline')
import pickle
import numpy as np
import pandas as pd

import xgboost as xgb

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics


# # [Basic sklearn examples](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py)

# ## models

# In[2]:

rng = np.random.RandomState(31337)


# In[3]:

print("Zeros and Ones from the Digits dataset: binary classification")
digits = sklearn.datasets.load_digits(2)
y = digits['target']
X = digits['data']
kf = sklearn.model_selection.KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(sklearn.metrics.confusion_matrix(actuals, predictions))


# In[4]:

print("Iris: multiclass classification")
iris = sklearn.datasets.load_iris()
y = iris['target']
X = iris['data']
kf = sklearn.model_selection.KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(sklearn.metrics.confusion_matrix(actuals, predictions))


# In[5]:

print("Boston Housing: regression")
boston = sklearn.datasets.load_boston()
y = boston['target']
X = boston['data']
kf = sklearn.model_selection.KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBRegressor().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(sklearn.metrics.mean_squared_error(actuals, predictions))


# ## grid search

# In[6]:

print("Parameter optimization")
y = boston['target']
X = boston['data']
xgb_model = xgb.XGBRegressor()
clf = sklearn.model_selection.GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
clf.fit(X,y)
print(clf.best_score_)
print(clf.best_params_)


# ## pickle model

# In[7]:

# The sklearn API models are picklable
print("Pickling sklearn API models")
# must open in binary format to pickle
pickle.dump(clf, open("best_boston.pkl", "wb"))
clf2 = pickle.load(open("best_boston.pkl", "rb"))
print(np.allclose(clf.predict(X), clf2.predict(X)))


# ## early stopping

# In[8]:

# Early-stopping

X = digits['data']
y = digits['target']
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=0)
clf = xgb.XGBClassifier()
clf.fit(X_train,
        y_train,
        early_stopping_rounds=10,
        eval_metric="auc",
        eval_set=[(X_test, y_test)])


# # [Mini course](http://machinelearningmastery.com/xgboost-python-mini-course/)

# ## Early stopping

# In[21]:

data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
    header=None)
data = np.array(data)
X = data[:, :-1]
y = data[:, -1]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.3)
mdl = xgb.XGBClassifier()
eval_set = [(X_test, y_test),]
mdl.fit(X_train,
        y_train,
        early_stopping_rounds=10,
        eval_metric="logloss",
        eval_set=eval_set,
        verbose=True)

y_pred = mdl.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# ## Feature Importance

# In[23]:

mdl.feature_importances_

xgb.plot_importance(mdl)


# ## How to Configure Gradient Boosting

# A number of configuration heuristics were published in the original gradient boosting papers. They can be summarized as:
# 
# Learning rate or shrinkage (learning_rate in XGBoost) should be set to 0.1 or lower, and smaller values will require the addition of more trees.
# The depth of trees (tree_depth in XGBoost) should be configured in the range of 2-to-8, where not much benefit is seen with deeper trees.
# Row sampling (subsample in XGBoost) should be configured in the range of 30% to 80% of the training dataset, and compared to a value of 100% for no sampling.

# ## Hyperparameter Tuning

# In[26]:

n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

kfold = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = sklearn.model_selection.GridSearchCV(mdl, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
result = grid_search.fit(X, y)


# The parameters to consider tuning are:
# 
# - The number and size of trees (n_estimators and max_depth).
# - The learning rate and number of trees (learning_rate and n_estimators).
# - The row and column subsampling rates (subsample, colsample_bytree and colsample_bylevel).

# In[49]:

result.best_estimator_


# # References
# - https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py
# - http://machinelearningmastery.com/xgboost-python-mini-course/
# - https://github.com/yang-zhang/xgboost/tree/master/demo
# - https://github.com/dmlc/xgboost/tree/master/demo/kaggle-higgs
# - [Owen Zhang Slides](https://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1/12)
# - [XGBoost Parameters](http://xgboost.readthedocs.io/en/latest//parameter.html)
# -[Notes on Parameter Tuning](https://xgboost.readthedocs.io/en/latest//how_to/param_tuning.html)
# - [How to tune hyperparameters of xgboost trees?](http://stats.stackexchange.com/questions/171043/how-to-tune-hyperparameters-of-xgboost-trees)
