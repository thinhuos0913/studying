# Linear Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10)
model = LinearRegression() #fit_intercept=False
lr=model.fit(X,Y)
# print(lr.score(X,Y))
# print(lr.coef_)
# print(lr.intercept_)
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
# Ridge Regression
from sklearn.linear_model import Ridge
model = Ridge()
rr=model.fit(X,Y)
# print(rr.coef_)
# print(rr.intercept_)
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
# LASSO Regression
from sklearn.linear_model import Lasso
model = Lasso()
lsr=model.fit(X,Y)
# print(lsr.coef_)
# print(lsr.sparse_coef_)
# print(lsr.intercept_)
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
# ElasticNet Regression
from sklearn.linear_model import ElasticNet
model = ElasticNet()
enr=model.fit(X,Y)
# print(enr.coef_)
# print(enr.sparse_coef_)
# print(enr.intercept_)
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
# KNN Regression
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
knn=model.fit(X,Y)
# print(knn.effective_metric_)
# print(knn.effective_metric_params_)
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
cart=model.fit(X,Y)
# print(cart.feature_importances_)
# print(cart.max_features_)
# print(cart.n_features_)
# print(cart.n_outputs_)
# print(cart.tree_)
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
# SVM Regression
from sklearn.svm import SVR
model = SVR() #kernel='linear'
svr=model.fit(X,Y)
print(svr.intercept_)
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())