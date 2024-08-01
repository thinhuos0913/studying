# Grid Search for Algorithm Tuning
import numpy
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
# param_grid = dict(alpha=alphas)
# # print(param_grid)
# model = Ridge()
# grid = GridSearchCV(estimator=model, param_grid=param_grid)
# grid.fit(X, Y)
# print(grid.best_score_)
# print(grid.best_estimator_.alpha)
# print(grid.cv_results_)
# print(grid.best_params_)
# print(grid.best_index_)
# print(grid.scorer_)
# print(grid.n_splits_)
# print(grid.refit_time_)
# Randomized for Algorithm Tuning
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
param_grid = {'alpha': uniform()}
model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, random_state=7)
rsearch.fit(X, Y)
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)
print(rsearch.best_params_)
print(rsearch.n_splits_)
print(rsearch.best_index_)
# print(rsearch.scorer_)
# print(rsearch.cv_results_)