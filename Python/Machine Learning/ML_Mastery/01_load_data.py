# Load CSV Using Python Standard Library
import csv
import numpy
filename = './datasets/pima-indians-diabetes.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',',quoting=csv.QUOTE_NONE)
x = list(reader)
#print(x)
data = numpy.array(x).astype('float')
print(data.shape)
# Load CSV using NumPy
from numpy import loadtxt
filename = './datasets/pima-indians-diabetes.csv'
raw_data = open(filename, 'rb')
data = loadtxt(raw_data, delimiter=",")
print(data.shape)
# Load CSV from URL using NumPy
# from numpy import loadtxt
# from urllib.request import urlopen
# url = 'https://goo.gl/vhm1eU'
# raw_data = urlopen(url)
# dataset = loadtxt(raw_data, delimiter=",")
# print(dataset.shape)
# Load CSV using Pandas
from pandas import read_csv
filename = './datasets/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
print(data)