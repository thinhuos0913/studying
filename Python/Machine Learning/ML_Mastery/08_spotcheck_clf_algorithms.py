from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
#print(dataframe.head(20))
num_folds = 10
kfold = KFold(n_splits=10)  #shuffle=True, random_state=7
model = LogisticRegression(solver='liblinear', multi_class='ovr') #solver='liblinear', multi_class='ovr'
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
X_new=[[2,100,50,38,170,41.7,2.3,29]]
pred=model.fit(X,Y).predict(X_new)
print('predict:\n', pred)
# Linear Discriminant Analysis Classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
X_new=[[2,100,50,38,170,41.7,2.3,29]]
pred=model.fit(X,Y).predict(X_new)
print('predict:\n', pred)
# KNN Classification
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
X_new=[[2,100,50,38,170,41.7,2.3,29]]
pred=model.fit(X,Y).predict(X_new)
print('predict:\n', pred)
# Gaussian Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
X_new=[[2,100,50,38,170,41.7,2.3,29]]
pred=model.fit(X,Y).predict(X_new)
print('predict:\n', pred)
# CART Classification
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
X_new=[[2,100,50,38,170,41.7,2.3,29]]
pred=model.fit(X,Y).predict(X_new)
print('predict:\n', pred)
# SVM Classification
from sklearn.svm import SVC
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
X_new=[[2,100,50,38,170,41.7,2.3,29]]
pred=model.fit(X,Y).predict(X_new)
print('predict:\n', pred)