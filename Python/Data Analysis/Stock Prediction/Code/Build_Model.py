import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from numpy import set_printoptions
from bokeh.plotting import figure, show, output_file
from datetime import date, time
from pandas_datareader import data
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
# GET DATA:
# today = date.today()
# start_date = datetime.datetime(2018,1,1)
# end_date = today
# df = data.DataReader(name='GOOG',data_source='yahoo',start = start_date,end=end_date)
# df.to_csv('GOOG.csv')
raw = pd.read_csv('GOOG.csv', header=0)
# data = raw.drop(['Close'], axis = 1)
# data = data.dropna()
# print(raw)
# print(raw)
# UNDERSTANDING DATA:
# print(raw.head(20))
# print(raw.dtypes)
# set_option('display.width', 100)
# set_option('precision', 3)
# description = raw.describe()
# print(description)
# correlations = raw.corr(method='pearson')
# print(correlations)
# print(data.skew())
# VISUALIZE:
# data.hist()
# plt.show()
# data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
# plt.show()
# data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
# plt.show()
# correlations = data.corr()
# plot correlation matrix
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations, vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,9,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)
# plt.show()
# scatter_matrix(data)
# plt.show()

def inc_dec(c, o): # input là chỉ số close và open của từng ngày và output là kết quả so sánh hai giá trị: Increase, Decrease hay Equal
    if c > o:
        value="Increase"
    elif c < o:
        value="Decrease"
    else:
        value="Equal"
    return value

# for c, o in zip(raw.Close,raw.Open):
# 	print(c,o)

raw["Status"]=[inc_dec(c,o) for c, o in zip(raw.Close,raw.Open)] # Add "Status" column 
# raw.to_csv('GOOG_new.csv')
# print(raw)
data = raw.drop(['Close'], axis = 1) # Drop unnecessary cols
data = data.drop(['Volume'], axis=1) 
data = data.dropna()
# print(data)
# class_counts = data.groupby('Status').size()
# print(class_counts)
# print(data)
# Features Selection:
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# from sklearn.decomposition import PCA
# new_data = data.values
# X = new_data[:,1:5]
# y = new_data[:,5]
# X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=7)
# print(X)
# print(y)
# test = SelectKBest(score_func=chi2, k=4)
# fit = test.fit(X, y)
# set_printoptions(precision=3)
# print(fit.scores_)
# features = fit.transform(X)
# summarize selected features
# print(features[0:5,:])
# feature extraction with RFE
# model = LogisticRegression(max_iter=6700)
# rfe = RFE(model, 3)
# fit = rfe.fit(X, y)
# print("Num Features: %d" % fit.n_features_)
# print("Selected Features: %s" % fit.support_)
# print("Feature Ranking: %s" % fit.ranking_)
# feature extraction with PCA
# pca = PCA(n_components=3)
# fit = pca.fit(X)
# summarize components
# print("Explained Variance: %s" % fit.explained_variance_ratio_)
# print(fit.components_)
# print(fit.explained_variance_)
# ExtraTreesClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# model = ExtraTreesClassifier()
# model.fit(X, y)
# print('ExtraTrees features importances:',model.feature_importances_)
# COMPARE CLF ALGORITHMS:
from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# models = []
# models.append(('LR', LogisticRegression(max_iter=6700)))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
# results=[]
# names=[]
# scoring='accuracy'
# for name, model in models:
# 	kfold=KFold(n_splits=10, shuffle=True, random_state=7)
# 	cv_results=cross_val_score(model, X_train, y_train, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	msg='%s:%f(%f)' %(name,cv_results.mean(),cv_results.std())
# 	print(msg)

# Standardize the dataset
# pipelines = []
# pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
# LogisticRegression())])))
# pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA',
# LinearDiscriminantAnalysis())])))
# pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
# KNeighborsClassifier())])))
# pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
# DecisionTreeClassifier())])))
# pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
# GaussianNB())])))
# pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
# results = []
# names = []
# for name, model in pipelines:
# 	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# 	print(msg)
# ENSEMBLE METHODS:
# ensembles = []
# ensembles.append(('AB', AdaBoostClassifier()))
# ensembles.append(('GBM', GradientBoostingClassifier()))
# ensembles.append(('RF', RandomForestClassifier()))
# ensembles.append(('ET', ExtraTreesClassifier()))
# results = []
# names = []
# for name, model in ensembles:
# 	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# 	print(msg)

# model = LogisticRegression(max_iter=6700)
# model.fit(X_train,y_train)
# predictions = model.predict(X_validation)
# print(accuracy_score(y_validation, predictions))
# print(confusion_matrix(y_validation, predictions))
# print(classification_report(y_validation, predictions))

# X_new = [[1500, 1500, 1550, 1600]]
# pred_new = model.predict_proba(X_new)
# print(pred_new)

# plt.plot(predictions)  
# plt.plot(y_validation)  
# plt.legend(['predicted','actual'])  
# plt.ylabel("Status")
# plt.show()

# SPLIT ACCORDING TO DATE:
print(data)
split_date = '2020-05-31'
X = data[data['Date'] < split_date]
print(X)
X_train = X[['High','Low','Open','Adj Close']]
y_train = X[['Status']]

print(X_train)
print(y_train)

X_1 = data[data['Date'] >= split_date]
print(X_1)
X_test = X_1[['High','Low','Open','Adj Close']]
y_test = X_1[['Status']]
print(X_test)
print(y_test)
y_train = y_train.loc[:,'Status'] # Convert into 1d array
y_test = y_test.loc[:,'Status']
print(y_train)
# model = LogisticRegression(max_iter=6700)
model = SVC(C=10000)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(predictions)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

predictions = pd.DataFrame(predictions,index=y_test.index,columns = ['Status']) # Reconvert into DataFrame
print(predictions)
# predictions.to_csv('predicted.csv')