import numpy
import pandas as pd
import matplotlib.pyplot as plt
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix

path = './datasets/ACB.csv'
dataset = read_csv(path, delimiter = ',', header = 0)
print(dataset.head())
# types = dataset.dtypes
# print(types)

dataset['DTYYYYMMDD'] = pd.to_datetime(dataset['DTYYYYMMDD']) # format column to datetime type
# types = dataset.dtypes
# print(types)

def inc_dec(c, o): # input là chỉ số close và open của từng ngày và output là kết quả so sánh hai giá trị: Increase, Decrease hay Equal
    if c > o:
        value="Increase"
    elif c < o:
        value="Decrease"
    else:
        value="Equal"
    return value

split_date = '12/31/2019'
dataset["Status"]=[inc_dec(c,o) for c, o in zip(dataset.Close,dataset.Open)]
data = dataset.drop(['Ticker'], axis = 1) # Drop unnecessary cols
# data = data.drop(['Volume'], axis=1) 
data = data.dropna()

X_1 = data[data['DTYYYYMMDD'] < split_date]
X_train=X_1[['Volume','Open','High','Low']]
y_train = X_1[['Close']]
# print(X_train)
# print(y_train)

X_2 = data[data['DTYYYYMMDD'] >= split_date]
# print(X_2)
X_test = X_2[['Volume','Open','High','Low']]
y_test = X_2[['Close']]
# print(X_test)
# print(y_test)
# X_train['S_3'] = X_train.loc[:,'Close']
# print(X_train)
# X_train = X_train.shift(1)
# print(X_train)
# X_train = X_train.rolling(window = 3).mean()
# print(X_train)
# X_1.to_csv('splitdate.csv')
# print(data)
# print(data[data['DTYYYYMMDD'] < split_date])
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

model = LinearRegression().fit(X_train,y_train) 

W = model.coef_
print(W)

predict_price = model.predict(X_test)
print('mse = ',mean_squared_error(y_test,predict_price))

# print(predict_price)
# Convert np array into pd frame:
predict_price = pd.DataFrame(predict_price,index=y_test.index,columns = ['price'])  
# predict_price.to_csv('predict_price.csv')
plt.plot(predict_price)
plt.plot(y_test)
# predict_price.plot()  
# y_test.plot()
plt.legend(['predicted_price','actual_price'])  
plt.ylabel("Price") 
plt.show()

r2_score = model.score(X_test,y_test)*100  
print(float("{0:.2f}".format(r2_score)))