
# coding: utf-8

# scikit-learn [User Guide](http://scikit-learn.org/stable/user_guide.html)

# #  Supervised learning

# ## Generalized Linear Models

# ### Least Angle Regression
# The advantages of LARS are:
# It is numerically efficient in contexts where p >> n (i.e., when the number of dimensions is significantly greater than the number of points)

# ### Bayesian Regression
# A good introduction to Bayesian methods is given in C. Bishop: Pattern Recognition and Machine learning
# 

# #### Bayesian Ridge Regression

# In[31]:

from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer
from sklearn import datasets
boston = datasets.load_boston()
X = boston.data
X = Normalizer().fit_transform(X)
y = boston.target
bayesian_ridge = BayesianRidge()
linear_regression = LinearRegression()

cross_val_score(bayesian_ridge, X, y, cv=5)
cross_val_score(linear_regression, X, y, cv=5)


# In[32]:

cross_val_score(bayesian_ridge, X, y, scoring='r2', cv=5)
cross_val_score(linear_regression, X, y, scoring='r2', cv=5)


# In[34]:

cross_val_score(bayesian_ridge, X, y, scoring='neg_mean_squared_error', cv=5)
cross_val_score(linear_regression, X, y, scoring='neg_mean_squared_error', cv=5)


# ### Logistic regression
# For large dataset, you may also consider using `SGDClassifier` with `log` loss.

# ## Linear and Quadratic Discriminant Analysis

# ## Kernel ridge regression

# ## Support Vector Machines

# ## Stochastic Gradient Descent

# ## Nearest Neighbors

# ## Gaussian Processes

# ## Cross decomposition

# ## Naive Bayes

# ## Decision Trees

# ## Ensemble methods

# ## Multiclass and multilabel algorithms

# ## Feature selection

# ## Semi-Supervised

# ## Isotonic regression

# ## Probability calibration

# ## Neural network models (supervised)

# # Unsupervised learning

# ## Gaussian mixture models
# ## Manifold learning
# ## Clustering
# ## Biclustering
# ## Decomposing signals in components (matrix factorization problems)
# ## Covariance estimation
# ## Novelty and Outlier Detection
# ## Density Estimation
# ## Neural network models (unsupervised)
# # Model selection and evaluation
# ## Cross-validation: evaluating estimator performance
# ## Tuning the hyper-parameters of an estimator
# ## Model evaluation: quantifying the quality of predictions
# ## Model persistence
# ## Validation curves: plotting scores to evaluate models
# # Dataset transformations
# ## Pipeline and FeatureUnion: combining estimators
# ## Feature extraction
# ## Preprocessing data
# ## Unsupervised dimensionality reduction
# ## Random Projection
# ## Kernel Approximation
# ## Pairwise metrics, Affinities and Kernels
# ## Transforming the prediction target (y)
# # Dataset loading utilities
# ## General dataset API
# ## Toy datasets
# ## Sample images
# ## Sample generators
# ## Datasets in svmlight / libsvm format
# ## Loading from external datasets
# ## The Olivetti faces dataset
# ## The 20 newsgroups text dataset
# ## Downloading datasets from the mldata.org repository
# ## The Labeled Faces in the Wild face recognition dataset
# ## Forest covertypes
# ## RCV1 dataset
# ## Boston House Prices dataset
# ## Breast Cancer Wisconsin (Diagnostic) Database
# ## Diabetes dataset
# ## Optical Recognition of Handwritten Digits Data Set
# ## Iris Plants Database
# ## Linnerrud dataset
# # Strategies to scale computationally: bigger data
# ## Scaling with instances using out-of-core learning
# # Computational Performance
# ## Prediction Latency
# ## Prediction Throughput
# ## Tips and Tricks

# In[ ]:



