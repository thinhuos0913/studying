# Create a pipeline that standardizes the data then creates a model
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# create pipeline
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)
# X_new=[[2,200,70,40,0,16.6,0.61,41]]
# predict=model.fit(X,Y).predict(X_new)
# print(predict)
# evaluate pipeline
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
X_new=[[2,100,50,38,170,41.7,2.3,29]]
pred=model.fit(X,Y).predict(X_new)
print(pred)
# Create a pipeline that extracts features from the data then creates a model
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
# create feature union
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
print(features[1][1])
kbest=features[1][1].fit(X,Y)
print(kbest.scores_)
feat=kbest.transform(X)
print(feat[:5,:])
print(features[0][1])
components=features[0][1].fit(X)
print(components.explained_variance_)
feature_union = FeatureUnion(features)
print(feature_union)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
# estimators.append(('logistic', LogisticRegression(max_iter=1000)))
estimators.append(('SVM', SVC()))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
# predict:
X_new=[[2,100,50,38,170,41.7,2.3,29]]
pred=model.fit(X,Y).predict(X_new)
print(pred)