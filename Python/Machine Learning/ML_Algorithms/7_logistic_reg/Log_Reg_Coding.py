import numpy as np
import matplotlib.pyplot as plt

def sigmoid(S):
	""" S: numpy array
	"""
	return 1/(1 + np.exp(-S))

def prob(w, X):
	"""
	X: a 2d numpy array of shape (N, d). N datapoints, each with size d
	w: a 1d numpy array of shape (d)
	"""
	return sigmoid(X.dot(w))

def loss(w, X, y, lamda):
	"""
	X, w as in prob()
	y: a 1d numpy array of shape (N). Each elem = 0 or 1
	"""
	a = prob(w, X)
	loss_0 = -np.mean(y*np.log(a) + (1-y)*np.log(1-a))
	weight_decay = 0.5*lamda/X.shape[0]*np.sum(w*w)
	return loss_0 + weight_decay

def logistic_regression(w_init, X, y, lamda, lr = 0.1, nepoches = 2000):
	# lamda: regulariza paramether, lr: learning rate, nepoches: # epoches
	N, d = X.shape[0], X.shape[1]
	w = w_old = w_init
	# store history of loss in loss_hist
	loss_hist = [loss(w_init, X, y, lamda)]
	ep = 0
	while ep < nepoches:
		ep += 1
		mix_ids = np.random.permutation(N) # stochastic
		for i in mix_ids:
			xi = X[i]
			yi = y[i]
			ai = sigmoid(xi.dot(w))
			# update
			w = w - lr*((ai - yi)*xi + lamda*w)
			loss_hist.append(loss(w, X, y, lamda))
		if np.linalg.norm(w - w_old)/d < 1e-6:
			break
		w_old = w
	return w, loss_hist

def separate(p):
	for i in range(len(p)):
		if p[i] >= 0.5: 
			p[i] = 1
		else:
			p[i] = 0
	return p
# Applied for Pass/Fail of 20 students problem:
np.random.seed(2)
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
print(X.shape)
# bias trick
Xbar = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
print(Xbar.shape[1])
w_init = np.random.randn(Xbar.shape[1])
lamda = 0.0001
w, loss_hist = logistic_regression(w_init, Xbar, y, lamda, lr = 0.05, nepoches = 500)
print('Solution of Logistic Regression:', w)
print('Final loss:', loss(w, Xbar, y, lamda))
print('Xbar: ', Xbar)
probability_of_pass=sigmoid(Xbar @ w)
print(probability_of_pass)
# plt.plot(loss_hist)
# plt.show()
sep = separate(probability_of_pass)
print(sep)