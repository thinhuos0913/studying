# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
#print(dataframe.shape)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, shuffle=True, random_state=seed) #shuffle=True, random_state=seed
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
# model.fit(X,Y)
# print(model.oob_decision_function_) <=> oob_score=True
# print(model.oob_score_)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
max_features = 3
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features,oob_score=True)
model.fit(X,Y)
# print(model.n_features_)
# print(model.feature_importances_)
# print(model.oob_decision_function_)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
# Extra Trees Classification
from sklearn.ensemble import ExtraTreesClassifier
max_features = 7
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
# AdaBoost Classification
from sklearn.ensemble import AdaBoostClassifier
num_trees = 30
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X,Y)
# print(model.estimator_weights_)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
# Stochastic Gradient Boosting Classification
from sklearn.ensemble import GradientBoostingClassifier
num_trees = 100
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed,subsample=0.5)
model.fit(X,Y)
# print(model.oob_improvement_.shape)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
# Voting Ensemble for Classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
# create the sub models
estimators = []
model1 = LogisticRegression(max_iter=1000)
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
#print(estimators[0][0])
# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X,Y)
# print(ensemble.named_estimators_)
results = cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())