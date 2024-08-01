# 1. Peek at data
# View first 20 rows
from pandas import read_csv
from pandas import set_option
filename = "pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
peek = data.head(20)
print(peek)
# 2. Dimensions of data
shape = data.shape
print(shape)
# 3. Data Type For Each Attribute
types = data.dtypes
print(types)
# 4. Descriptive Statistics
set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
print(description)
# 5. Class Distribution (Classification Only)
class_counts = data.groupby('class').size()
print(class_counts)
# 6. Correlations Between Attributes
correlations = data.corr(method='pearson')
print(correlations)
# 7. Skew of Univariate Distributions
skew = data.skew()
print(skew)