from __future__ import division, print_function, unicode_literals
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(4)
from sklearn import datasets, linear_model
from matplotlib.backends.backend_pdf import PdfPages

N = 30
N_test = 20 
X = np.random.rand(N, 1)*5
y = 3*(X - 2) * (X - 3)*(X - 4) +  10*np.random.randn(N, 1)

X_test = (np.random.rand(N_test,1) - 1/2) *2
y_test = 3*(X_test -2) * (X_test - 3)*(X_test-4) +  10*np.random.randn(N_test, 1)

# print(y)
plt.scatter(X, y, c = 'r', s = 40)
# plt.show()
# print(X)
res = np.ones((X.shape[0],1))
# print(res)
for i in range(1,4):
	res = np.concatenate((res, X**i), axis = 1)

# print(res)

Xbar = res
# print(Xbar)

model = linear_model.LinearRegression(fit_intercept=False)
model.fit(Xbar, y)

w = model.coef_
print('w = \n', w)
# print(w[0][0])
# print(w[0][1])
print(w.shape)
print(Xbar.shape)

y_hat = Xbar@w.T

print(y_hat.shape)

plt.scatter(X, y_hat, label="Predicted Value")
plt.show()


# Polynomial regression
# def buildX(X, d = 2):
#     res = np.ones((X.shape[0], 1))
#     for i in range(1, d+1):
#         res = np.concatenate((res, X**i), axis = 1)
#     return res 

# def myfit(X, y, d):
#     Xbar = buildX(X, d)
#     regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
#     regr.fit(Xbar, y)

#     w = regr.coef_

#     # Display result
#     w_0 = w[0][0]
#     w_1 = w[0][1]
#     x0 = np.linspace(-1, 7, 200, endpoint=True)
#     y0 = np.zeros_like(x0)
#     ytrue = 5*(x0 - 2)*(x0 - 3)*(x0 - 4)
#     for i in range(d+1):
#         y0 += w[0][i]*x0**i

#     # Draw the fitting line 
#     with PdfPages('polyreg.pdf') as pdf:
#         plt.scatter(X.T, y.T, c = 'r', s = 40)     # data 
#         print(X_test.shape, y_test.shape)
#         plt.scatter(X_test.T, y_test.T, c = 'y', s = 40, label = 'Test samples')     # data 

#         l1, = plt.plot(x0, y0, 'b', linewidth = 2, label = "Predicted model")   # the fitting line
#         plt.legend(handles = [l1], fontsize = 18)
#         plt.plot(x0, ytrue, '--g', linewidth = 2, label = "Trained model")   # the fitting line
#         plt.xticks([], [])
#         plt.yticks([], [])


#         plt.title('Polynomial regression')
#         plt.axis([-4, 10, np.amax(y_test)-100, np.amax(y) + 30])
#         plt.legend(loc="best")

#         fn = 'linreg_' + str(d) + '.png'

#         plt.xlabel('$x$', fontsize = 20);
#         plt.ylabel('$y$', fontsize = 20);

#         pdf.savefig(bbox_inches='tight') #, dpi = 600)

#         plt.show()
#     print(w)

# myfit(X, y, 3)