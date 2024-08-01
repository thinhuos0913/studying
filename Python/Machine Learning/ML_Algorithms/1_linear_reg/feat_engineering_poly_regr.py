# import numpy as np
# import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
# np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# # create target data
# x = np.arange(0, 20, 1)

# print(x.shape)
# X = x.reshape(-1,1)
# res = np.ones((X.shape[0], 1))
# Xbar = np.concatenate((res, X), axis = 1)
# print(Xbar)
# y = 1 + x**2
# # y = np.cos(x/2)
# print(y)

# # regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
# # regr.fit(Xbar, y)

# # w = regr.coef_
# # print(w)

# # plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
# # plt.plot(x, Xbar@w, label="Predicted Value")
# # plt.xlabel("X"); plt.ylabel("y")
# # plt.legend()
# # plt.show()

# # Swap X for X**2
# # X = x**2 # feature engineering
# # print(X)
# # X = X.reshape(-1, 1)
# # print(X)

# # Xbar = np.concatenate((res, X), axis = 1)
# # print(Xbar)

# # regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
# # regr.fit(Xbar, y)

# # w = regr.coef_
# # print(w)

# # plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
# # plt.plot(x, Xbar@w, label="Predicted Value")
# # plt.xlabel("X"); plt.ylabel("y")
# # plt.legend()
# # plt.show()

# # engineer features .
# X = np.c_[x, x**2, x**3]   #<-- added engineered feature
# # y = x**2
# Xbar = np.concatenate((res, X), axis = 1)
# print(Xbar)
# # X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
# # print(X)

# regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
# regr.fit(Xbar, y)

# w = regr.coef_

# print(w)

# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
# plt.plot(x, Xbar@w, label="Predicted Value")
# plt.xlabel("X"); plt.ylabel("y")
# plt.legend()
# plt.show()

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(4)


x = np.random.rand(30, 1)*5
print(x.shape)
X = x.reshape(-1,1)
print(X.shape)

y = 3*(x - 2) * (x - 3)*(x - 4) +  10*np.random.randn(30, 1)

# print(y)

# regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
# regr.fit(X, y)

# w = regr.coef_
# print(w)

# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
# plt.plot(x, X@w, label="Predicted Value")
# plt.xlabel("X"); plt.ylabel("y")
# plt.legend()
# plt.show()

# # Swap X for X**2
# X = x**2 # feature engineering
X = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7]

regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(X, y)

w = regr.coef_
print(w.shape)
print(X.shape)

plt.scatter(x, y, marker='o', c='r', label="Actual Value"); plt.title("no feature engineering")
l1, = plt.plot(x, X@w.T, label="Predicted Value")
plt.xlabel("X"); plt.ylabel("y")
plt.legend()
plt.show()