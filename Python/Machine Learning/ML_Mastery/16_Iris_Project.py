# STEP 1: PREPARING DATA

# IMPORT LIBRARIES:

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold

# LOAD DATASET:

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# STEP 2: SUMMARIZE THE DATASET

#print(dataset.shape) # to see dimension of dataset
#print(dataset.head(20)) # to see the first 20 rows of data
#print(dataset.describe()) # to see statistical summary of each attributes (count, mean, min and max values)
#print(dataset.groupby('class').size()) # to see class distribution

# STEP 3: DATA VISUALIZATION

# Univariate plots: to better understand each attribiutes

#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()
#dataset.hist() # histogram
#pyplot.show()

# Multivariate plots: to better understand the relationship between attributes

#scatter_matrix(dataset) # scatter plot matrix
#pyplot.show()

# STEP 4: EVALUATE SOME ALGORITHMS

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Test Harness: we use stratified 10-fold cross validation to estimate model accuracy.
# Build Models
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
#pyplot.boxplot(results, labels=names)
#pyplot.title('Algorithm Comparison')
#pyplot.show()

# STEP 5: MAKE PREDICTIONS
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
#print(predictions)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Make prediction on single row data:
Xnew = [[0.79415228, 2.10495117, 1.34562103, 0.37829028]] # create single row data
model=SVC(gamma='auto',probability=True)
model.fit(X_train,Y_train)
Ynew=model.predict(Xnew)
print("Xnew=%s, Predicted=%s" % (Xnew, Ynew[0]))
# Probability prediction:
ynew = model.predict_proba(Xnew) 
#print(ynew)
print("Xnew=%s, Predicted=%s" % (Xnew, ynew))
# Show X_validation result:
for i in range(len(X_validation)):
	print("X_validation=%s, Predicted=%s" % (X_validation[i], predictions[i]))
# Multiple class prediction:
X_class, _ = make_blobs(n_samples=3, centers=2, n_features=4, center_box=(0,9),random_state=1)
print(X_class)
print(_)
Y_class=model.predict(X_class)
for i in range(len(X_class)):
	print("X_class=%s, Predicted=%s" % (X_class[i], Y_class[i]))

Y_class=model.predict_proba(X_class)
for i in range(len(X_class)):
	print("X_class=%s, Predicted=%s" % (X_class[i], Y_class[i]))

# A, b = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2, random_state=0)
# print(A)
# print(b)