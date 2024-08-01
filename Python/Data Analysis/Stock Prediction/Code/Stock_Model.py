from pandas_datareader import data
import numpy
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from datetime import date, time
from numpy import arange
from pandas import read_csv
from pandas import set_option

# today = date.today()
# start_date = datetime.datetime(2020,1,1)
# end_date = today
# df = data.DataReader(name='ACB',data_source='yahoo',start = start_date,end=end_date)
# df.to_csv('ACB_new.csv')
# data = df[['Adj Close']]
# data = data.dropna()
# print(df)
# data.plot(figsize=(10,5)) 
# plt.ylabel("GOOGLE Stock Prices")
# ax1 = plt.subplot()
raw = pd.read_csv('ACB/ACB_new.csv', header = 0)
# ax1.hist(raw['Adj Close'])
# plt.show()
# print(raw)
# plt.show()
# ax1 = plt.subplot()
# ax1.hist(raw['Date'].values, bins=100)
# ax1.set_title('GOOG Daily Price Changes')
# plt.show()
split_date = '2020-06-01'
# ax1 = plt.subplot()
# ax1.set_xticks([datetime.date(i,j,1) for i in range(2019,2020) for j in [1,8]])
# ax1.set_xticklabels('')
# ax1.plot(raw[raw['Date'] < split_date]['Date'].astype(datetime.datetime),
#          raw[raw['Date'] < split_date]['Adj Close'], 
#          color='#B08FC7', label='Training')
# ax1.plot(raw[raw['Date'] >= split_date]['Date'].astype(datetime.datetime),
#          raw[raw['Date'] >= split_date]['Adj Close'], 
#          color='#8FBAC8', label='Test')
# ax1.set_xticklabels('')
# ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
# plt.show()
X_1 = raw[raw['Date'] < split_date]
print(X_1)
X_train=X_1[['Adj Close']] 
print(X_train)
X_train['S_3'] = X_train.loc[:,'Adj Close'].shift(1).rolling(window=3).mean() 
print(X_train)
X_train['S_10']= X_train.loc[:,'Adj Close'].shift(1).rolling(window=10).mean()
print(X_train) 
# Drop rows with missing values 
X_train= X_train.dropna() 

y_train = X_train.loc[:,'Adj Close']
X_train = X_train[['S_3', 'S_10']]
# print(X_train)
X_2 =  raw[raw['Date'] >= split_date]
print(X_2)
X_test =X_2[['Adj Close']] 
print(X_test)
X_test['S_3'] = X_test.loc[:,'Adj Close'].shift(1).rolling(window=3).mean() 

X_test['S_10']= X_test.loc[:,'Adj Close'].shift(1).rolling(window=10).mean() 
# Drop rows with missing values 
X_test= X_test.dropna() 

y_test = X_test.loc[:,'Adj Close']
X_test = X_test[['S_3', 'S_10']] 
print(X_test)
# print(X_test)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 

# sc = StandardScaler()

# y_train = y_train.values
# print(y_train.shape)
# X_train = X_train.values
# print(X_train.shape)
# y_train = y_train.reshape(-1,1)
# train_scaled = sc.fit_transform(X_train)
# y_train_scaled = sc.fit_transform(y_train.reshape(-1,1))

# linear = LinearRegression().fit(train_scaled,y_train_scaled)

linear = LinearRegression().fit(X_train,y_train)
print(linear.coef_[0])
print(linear.coef_[1])
print(linear.intercept_)
# print (" Price =", round(linear.coef_[0],2),  "* 3 Days Moving Average", 
# 	round(linear.coef_[1],2),  "* 10 Days Moving Average +", round(linear.intercept_,2))

# test_scaled = sc.fit_transform(X_test)

# y_test = y_test.values
# y_test_scaled = sc.fit_transform(y_test.reshape(-1,1))
predicted_price = linear.predict(X_test)  

# predicted_price = linear.predict(test_scaled)
# print(predicted_price)

predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])  
# predicted_price = pd.DataFrame(predicted_price)  

predicted_price.plot(figsize=(10,5))  

# y_test_scaled = pd.DataFrame(y_test_scaled)
y_test.plot()  

plt.legend(['predicted_price','actual_price'])  

plt.ylabel("Price")  

plt.show()

r2_score = linear.score(X_test,y_test)*100  

print(float("{0:.2f}".format(r2_score)))