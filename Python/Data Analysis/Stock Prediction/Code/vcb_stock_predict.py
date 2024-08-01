import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path
# from os import path
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout


# Doc du lieu VCB 2009->2018
dataset_train = pd.read_csv('vcb_2009_2018.csv')
print('Dataset:\n', dataset_train)
training_set = dataset_train.iloc[:, 1:2]
print('Training_set:\n', training_set)
training_set = dataset_train.iloc[:, 1:2].values
print('Training_set:\n', training_set)
# Thuc hien scale du lieu gia ve khoang 0,1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
print('Scaled set:\n', training_set_scaled)
# # Tao du lieu train, X = 60 time steps, Y =  1 time step
X_train = []
y_train = []
no_of_sample = len(training_set)
print('Numb of sample:', no_of_sample)

for i in range(60, no_of_sample):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

print('X_train:\n', len(X_train))
print('y_train:\n', len(y_train))

import h5py
# fn = h5py.File('mymodel.h5', 'r')

X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape, y_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)
# print((X_train.shape[1], 1))

# # # Xay dung model LSTM
# # regressor = Sequential()
# # regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# # regressor.add(Dropout(0.2))
# # regressor.add(LSTM(units = 50, return_sequences = True))
# # regressor.add(Dropout(0.2))
# # regressor.add(LSTM(units = 50, return_sequences = True))
# # regressor.add(Dropout(0.2))
# # regressor.add(LSTM(units = 50))
# # regressor.add(Dropout(0.2))
# # regressor.add(Dense(units = 1))
# # regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# # Neu ton tai file model thi load
# # if path.exists("mymodel.h5"):
# #     regressor.load_weights("mymodel.h5")
# # else:
# #     # Con khong thi train
# #     regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# #     regressor.save("mymodel.h5")

# # Load du lieu tu 1/1/2019 - 2/10/2019
dataset_test = pd.read_csv('vcb_2019.csv')
print('Test set:\n', dataset_test.iloc[:,1:2])
real_stock_price = dataset_test.iloc[:, 1:2].values
print('Real price:\n', real_stock_price)
# # Tien hanh du doan
# dataset_total = pd.concat((dataset_train['CLOSE'], dataset_test['CLOSE']), axis = 0)
# # dataset_total.to_csv('full_set.csv')
# # print('Full dataset:\n', dataset_total)
# # print('Length of full set:\n', len(dataset_total))
# inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# # print(len(dataset_test))
# # print(inputs.shape)
# inputs = inputs.reshape(-1,1)
# # print(inputs.shape)
# inputs = sc.transform(inputs)
# # print(inputs)
# X_test = []
# no_of_sample = len(inputs) # =246

# for i in range(60, no_of_sample):
#     # print(i)
#     X_test.append(inputs[i-60:i, 0])

# # print(len(X_test))
# X_test = np.array(X_test)
# # print('X_test:\n', X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# # print('X_test_reshaped:\n', X_test)
# # predicted_stock_price = regressor.predict(X_test)
# # predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# # Ve bieu do gia that va gia du doan
# # plt.plot(real_stock_price, color = 'red', label = 'Real VCB Stock Price')
# # plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted VCB Stock Price')
# # plt.title('VCB Stock Price Prediction')
# # plt.xlabel('Time')
# # plt.ylabel('VCB Stock Price')
# # plt.legend()
# # plt.show()

# # Du doan tiep gia cac ngay tiep theo den 30/10

# # print(len(dataset_test))
# dataset_test = dataset_test['CLOSE'][len(dataset_test)-60:len(dataset_test)].to_numpy()
# # print('dataset_test:\n', dataset_test)
# dataset_test = np.array(dataset_test)
# # print(dataset_test.shape)
# # print(len(dataset_test))

# inputs = dataset_test
# inputs = inputs.reshape(-1,1)
# inputs = sc.transform(inputs)
# print('inputs:\n', len(inputs))

# i = 0
# while i<28:
#     X_test = []
#     no_of_sample = len(dataset_test)

#     # Lay du lieu cuoi cung
#     X_test.append(inputs[no_of_sample - 60:no_of_sample, 0])
#     # print('X_test_list:\n', X_test)
#     X_test = np.array(X_test)
#     # print('X_test_arr:\n', X_test)
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#     # print('X_test_tensor:\n', X_test)

#     # Du doan gia
#     # predicted_stock_price = regressor.predict(X_test)

#     # chuyen gia tu khoang (0,1) thanh gia that
#     # predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#     # Them ngay hien tai vao
#     # dataset_test = np.append(dataset_test, predicted_stock_price[0], axis=0)
#     # inputs = dataset_test
#     # inputs = inputs.reshape(-1, 1)
#     # inputs = sc.transform(inputs)

#     # print('Stock price ' + str(i+3) + '/10/2019 of VCB : ', predicted_stock_price[0][0])
#     i = i + 1



