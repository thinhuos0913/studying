from pandas_datareader import data
import datetime
import pandas as pd
from bokeh.plotting import figure, show, output_file
from datetime import date
import numpy
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

# today = date.today()
# start_date = datetime.datetime(2019,1,1)
# end_date = today
# df = data.DataReader(name='STB',data_source='yahoo',start = start_date,end=end_date)
# df.to_csv('STB.csv')
df = pd.read_csv('STB_new.csv', delimiter = ',', header = 0)
print(df)
df = df.drop(["Close"], axis = 1)
print(df)
df = df.values
X = df[:,1:5]
y = df[:,5]
# X.astype(float)
# X[:,3] = float(X[:,3])
print(X)
print(y)
# split_date = '2020/07/17'
# X_1 = df[df['Date'] < split_date]
# print(X_1)
# X_train=X_1[['Adj Close']] 
# print(X_train)
# X_train['S_3'] = X_train.loc[:,'Adj Close'].shift(1).rolling(window=3).mean() 
# print(X_train)
# X_train['S_10']= X_train.loc[:,'Adj Close'].shift(1).rolling(window=10).mean()
# print(X_train) 
validation_size = 0.2
seed = 7
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso(max_iter=6700)))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
results = []
names = []

for name, model in models:
	kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso(max_iter=6700))])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
	kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# scaler = StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)

model = LinearRegression()
model.fit(X_train,y_train)
# # transform the validation dataset
# rescaledValidationX = scaler.transform(X_validation)
# print('RescaledValidation:\n', rescaledValidationX)
predictions = model.predict(X_validation)
# print(mean_squared_error(y_validation, predictions))

X_new = [[13.9, 13.5, 13.7, 9734960]]
pred_new = model.predict(X_new)
print(pred_new)

# split_date = '2020/07/17'
# X_1 = df[df['Date'] < split_date]
# print(X_1)
# X_train=X_1[['Adj Close']] 
# print(X_train)
# X_train['S_3'] = X_train.loc[:,'Adj Close'].shift(1).rolling(window=3).mean() 
# print(X_train)
# X_train['S_10'] = X_train.loc[:,'Adj Close'].shift(1).rolling(window=10).mean()
# print(X_train)