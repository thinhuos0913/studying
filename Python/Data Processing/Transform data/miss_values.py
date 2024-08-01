import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer # instead of sklearn.preprocessing import Imputer => latest version
#from sklearn.preprocessing import Imputer # old version
data=pd.read_csv('missvalue_test.csv',header=None)
print(data)
#X=data.values
#print(X)
#imp=SimpleImputer(missing_values=np.NaN,strategy='mean') # Insert missing values by mean
imp=SimpleImputer(missing_values=np.NaN,strategy='most_frequent') # Insert missing values by most frequent number
imp.fit(data)
result=imp.transform(data)
print('Result is: ')
print(result)