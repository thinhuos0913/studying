import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)
# data=pd.read_csv('data_classification.csv',header=None)
# print(data.shape)

data = np.loadtxt('data01.txt',delimiter=',')
# print(data.shape)
# print(data)
X = data[:,:3]
y = data[:,3]
print(X.shape)
print(y.shape)

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def sigmoid_grad(z):
	return sigmoid(z)*(1-sigmoid(z))

def separate(p):
	for i in range(len(p)):
		if p[i] >= 0.5: 
			p[i] = 1
		else:
			p[i] = 0
	return p

def mlp_init(d0, d1, d2):
	""" Initialize W1, b1, W2, b2
	d0: dimension of input data
	d1: number of hidden unit
	d2: number of output unit = number of classes
	"""
	W1 = 0.01*np.random.randn(d0, d1)
	b1 = np.zeros(d1)
	W2 = 0.01*np.random.randn(d1, d2)
	b2 = np.zeros(d2)
	return (W1, b1, W2, b2)

def loss_function(Yhat, y):
	"""
	Y-hat: a numpy array of shape (N-points, n-Classes) --- predicted output
	y: a numpy array of shape (N-points) --- ground truth.
	NOTE: We don’t need to use the one-hot vector here since most of
	elements are zeros. When programming in numpy, in each row of Yhat, we
	need to access to the corresponding index only.
	"""
	# id0 = range(Yhat.shape[0])
	# return -np.mean(y*np.log(Yhat) + (1-y)*np.log(1-Yhat))
	return -np.mean(y*np.log(Yhat) + (1-y)*np.log(1-Yhat))

def predict(X, W1, b1, W2, b2):
	"""Suppose the network has been trained, predict class of new points.
	X: data matrix, each ROW is one data point.
	W1, b1, W2, b2: learned weight matrices and biases
	"""
	Z1 = X.dot(W1) + b1 # shape (N, d1)
	A1 = sigmoid(Z1) # shape (N, d1)
	Z2 = A1.dot(W2) + b2 # shape (N, d2)
	return sigmoid(Z2)
# Random Initialization:	
W1, b1, W2, b2 = mlp_init(3,4,1)
# print(W1.shape)
# print(b1.shape)
# print(W2.shape)
# print(b2.shape)
alpha = 0.0000005 # learning rate
loss_hist = [] 
for i in range(500):
	# Forward Propagation:
	Z1 = X.dot(W1) + b1
	# print(Z1.shape) #(54,4)
	A1 = sigmoid(Z1)
	# print(A1.shape) #(54,4)
	Z2 = A1.dot(W2) + b2
	# print(Z2.shape) #(54,1)
	A2 = sigmoid(Z2)
	# print(A2.shape) #(54,1)
	if i %100 == 0: # print loss after each 1000 iterations
		loss = loss_function(A2, y)
		print("iter %d, loss: %f" %(i, loss))
	loss_hist.append(loss)
	# Back Propagation:
	A2 = A2.reshape(54,)
	dZ2 = A2 - y
	# print(dZ2.shape) #(54,)
	m = X.shape[0]
	dW2 = 1/m*np.dot(dZ2.T,A1)
	# print(dW2.shape) #(4,)
	dW2 = dW2.reshape(4,1)
	db2 = 1/m*np.sum(dZ2, axis = 0, keepdims = True)
	# print(db2.shape) #(1,)
	dZ1 = np.dot(dZ2.reshape(54,1),W2.T)*sigmoid_grad(Z1)
	# print(dZ1.shape) #(54,4)
	dW1 = 1/m*np.dot(X.T,dZ1)
	# print(dW1.shape) #(3,4)
	db1 = 1/m*np.sum(dZ1, axis = 0, keepdims = True)
	# print(db1.shape) #(1,1)
	db1 = db1.reshape(4,)
	# Update gradient descent
	W1 += -alpha*dW1
	b1 += -alpha*db1
	W2 += -alpha*dW2
	b2 += -alpha*db2

print('W1:',W1)
print('b1:',b1)
print('W2:',W2)
print('b2:',b2)

pred = predict(X,W1,b1,W2,b2)
print(pred)

p = separate(pred)
p = p.reshape(54,)
print(p)
print(y)

acc = 100*np.mean(p == y)
print(acc)

plt.plot(loss_hist)
plt.show()