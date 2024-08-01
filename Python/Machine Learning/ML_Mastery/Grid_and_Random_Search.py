# Random Search logistic regression model on the sonar dataset
from scipy.stats import loguniform
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
# load dataset
url = 'sonar.all-data.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = LogisticRegression(max_iter=6700)
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
# space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
# space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 100)
# define search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

###----------------------------------------------------###
# Grid Search logistic regression model on the sonar dataset
# from pandas import read_csv
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score
# # load dataset
# url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# dataframe = read_csv(url, header=None)
# # split into input and output elements
# data = dataframe.values
# X, y = data[:, :-1], data[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# # define model
# model = LogisticRegression(max_iter = 6700)
# # define evaluation
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# # define search space
# space = dict()
# # space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
# # space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
# space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
# # define search
# search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
# # execute search
# result = search.fit(X_train, y_train)
# # summarize result
# print('Best Score: %s' % result.best_score_)
# print('Best Hyperparameters: %s' % result.best_params_)

# model = LogisticRegression(max_iter=6700, C=10)
# model = model.fit(X_train,y_train)
# pred_train = model.predict(X_train)
# train_err = accuracy_score(y_train,pred_train)
# print('Train_set error = ', train_err)
# pred_test = model.predict(X_test)
# test_err = accuracy_score(y_test,pred_test)
# print('Test_set error = ', test_err)

