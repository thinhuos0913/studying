# Save Model Using Pickle
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# Fit the model on 33%
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
# Save the model to disk
filename = 'finalized_model.sav'
dump(model, open(filename, 'wb'))
# some times later........
# load the model from disk to use for ML
loaded_model = load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

# Save Model Using JobLib: deprecated 
from joblib import dump
from joblib import load
# save the model to disk
filename = 'finalized_model_using_JobLib.sav'
dump(model, filename)
# some times later.....
# load the model from disk
loaded_model = load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)