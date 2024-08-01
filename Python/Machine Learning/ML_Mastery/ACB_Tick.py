import numpy
import pandas as pd
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

path = 'D:/STUDYING/MACHINE LEARNING/Master/Topics/Data Analysis/ACB.csv'
dataset = read_csv(path, delimiter = ',', header = 0)
dataset['DTYYYYMMDD'] = pd.to_datetime(dataset['DTYYYYMMDD']) # format column to datetime type
print(dataset)
set_option('precision', 1)
print(dataset.describe())
set_option('precision', 2)
print(dataset.corr(method='pearson'))
# dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
# pyplot.show()
# dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=True, fontsize=1)
# pyplot.show()
# dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8)
# pyplot.show()
# scatter_matrix(dataset)
# pyplot.show()
# array = dataset.values
# X = array[:,2:6]
# Y = array[:,6]
# validation_size = 0.3
# seed = 7
# X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
# # # Test options and evaluation metric
# num_folds = 10
# scoring = 'neg_mean_squared_error'
# # scoring = 'r2'
# # Spot-Check Algorithms
# models = []
# models.append(('LR', LinearRegression()))
# models.append(('LASSO', Lasso(max_iter=6700)))
# models.append(('EN', ElasticNet()))
# models.append(('KNN', KNeighborsRegressor()))
# models.append(('CART', DecisionTreeRegressor()))
# models.append(('SVR', SVR()))
# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
# 	kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
# 	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# 	print(msg)

# # fig = pyplot.figure()
# # fig.suptitle('Algorithm Comparison')
# # ax = fig.add_subplot(111)
# # pyplot.boxplot(results)
# # ax.set_xticklabels(names)
# # pyplot.show()

# # Standardize the dataset
# pipelines = []
# pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
# pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
# pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
# pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
# pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
# pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
# results = []
# names = []
# for name, model in pipelines:
# 	kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
# 	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# 	print(msg)

# # Compare Algorithms
# # fig = pyplot.figure()
# # fig.suptitle('Scaled Algorithm Comparison')
# # ax = fig.add_subplot(111)
# # pyplot.boxplot(results)
# # ax.set_xticklabels(names)
# # pyplot.show()

# model = LinearRegression()
# model.fit(X_train,Y_train)
# W = model.coef_
# b = model.intercept_
# print(X_train.shape)
# print(W)
# print(b)
# predict = X_train @ W
# print(predict)

# print(Y_validation)
# pred = model.predict(X_validation)
# print(pred)

# print(mean_squared_error(Y_validation, pred))

# scaler = StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# model = LinearRegression()
# model.fit(rescaledX,Y_train)
# W = model.coef_
# b = model.intercept_
# print('rescaledX:\n', rescaledX)
# print(W)
# print(b)
# predict = rescaledX @ W
# print(predict)

# rescaledValidationX = scaler.transform(X_validation)
# predictions = model.predict(rescaledValidationX)
# print(mean_squared_error(Y_validation, predictions))

# from sklearn.preprocessing import Normalizer

# scaler = Normalizer().fit(X_train)
# normalizedX = scaler.transform(X_train)
# print('Normalized:\n', normalizedX)
# model = LinearRegression()
# model.fit(normalizedX,Y_train)
# W = model.coef_
# b = model.intercept_
# print(W)
# print(b)
# predict = normalizedX @ W
# print(predict)
# normValidationX = scaler.transform(X_validation)
# predictions = model.predict(normValidationX)
# print(mean_squared_error(Y_validation, predictions))
split_date = '12-31-2019'
# X_1 = dataset[dataset['DTYYYYMMDD'] < split_date]
# print(X_1)

def inc_dec(c, o): # input là chỉ số close và open của từng ngày và output là kết quả so sánh hai giá trị: Increase, Decrease hay Equal
    if c > o:
        value="Increase"
    elif c < o:
        value="Decrease"
    else:
        value="Equal"
    return value

dataset["Status"]=[inc_dec(c,o) for c, o in zip(dataset.Close,dataset.Open)]
data = dataset.drop(['Ticker'], axis = 1) # Drop unnecessary cols
# data = data.drop(['Volume'], axis=1) 
data = data.dropna()

print(data)

X_1 = data[data['DTYYYYMMDD'] < split_date]
print('X_1:\n', X_1)
X_train=X_1[['Close']] 
print('X_train:\n', X_train)
X_train['S_3'] = X_train.loc[:,'Close'].shift(1).rolling(window=3).mean() 
print(X_train)
X_train['S_10']= X_train.loc[:,'Close'].shift(1).rolling(window=10).mean()
print(X_train) 

X_train= X_train.dropna() 

print(X_train)