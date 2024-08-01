# UNIVARIATE PLOTS:
# 1. Histogram
# Univariate Histograms
from matplotlib import pyplot
from pandas import read_csv
import numpy
from pandas.plotting import scatter_matrix
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
# data.hist()
#pyplot.show()
# 2. Density plot
# data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
#pyplot.show()
# 3. Box and Whisker Plots
# data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
#pyplot.show()
# MULTIVARIATE PLOTS:
# 1. Correction Matrix Plot
correlations = data.corr()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
#pyplot.show()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
#pyplot.show()
# 2. Scatter Plot Matrix
scatter_matrix(data)
pyplot.show()