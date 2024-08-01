import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data=pd.read_csv('data_classification.csv',header=None)
# #print(data.values)

# true_x=[]
# true_y=[]
# false_x=[]
# false_y=[]

# # Classify Pass (1) and Fail (0)
# for item in data.values:
# 	if item[2]==1.:
# 		true_x.append(item[0])
# 		true_y.append(item[1])
# 	else:
# 		false_x.append(item[0])
# 		false_y.append(item[1])
# Show data 
#plt.scatter(true_x,true_y,marker='o',c='b')
#plt.scatter(false_x,false_y,marker='s',c='r')
#plt.show()
# X=np.array(data)
# y=np.copy(X[:,2])
# #y=y.reshape(1,-1) # matrix(100,1) => matrix(1,100)
# X[:,2]=X[:,1]
# X[:,1]=X[:,0]
# X[:,0]=1
# Define Sigmoid function:
def sigmoid(z):
	return 1.0/(1+np.exp(-z))
# Define Separate function:
def separate(p):
	for i in range(len(p)):
		if p[i] >= 0.5: 
			p[i] = 1
		else:
			p[i] = 0
	return p
# Define Predict function:
def predict(X, w):
	z=np.dot(X,w)
	return sigmoid(z)

def cost_function(X, y, w):
	n=len(y)
	prediction=predict(X,w) 
	cost_class1=-y*np.log(prediction) # h(theta) function <=> predict function
	cost_class0=-(1-y)*np.log(1-prediction)
	cost=cost_class0+cost_class1
	return cost.sum()/n

def update_w(X,y,w,learning_rate): # Gradient Descent
	n=len(y)
	prediction=predict(X,w)
	grad=np.dot(X.T,(prediction-1))
	grad=grad/n*learning_rate
	w=w-grad
	return w
def training(X,y,w,learning_rate,iter):
	cost_history=[]
	for i in range(iter):
		w=update_w(X,y,w,learning_rate)
		cost=cost_function(X, y, w)
		cost_history.append(cost)
	return w,cost_history

fp = 'data01.csv'
data = pd.read_csv(fp, delimiter = ',', header = None)
data = data.values
X = data[:,0:2]
y = data[:,2]
print(X.shape)
print(y.shape)
y = y.reshape(-1,1)
# print(y.shape)

w = np.zeros((X.shape[1],1))
print(w)

J = cost_function(X,y,w)
print(J)

w, cost_history = training(X,y,w,0.000001,200)
print(w)
# print(cost_history)

# plt.plot(cost_history)
# plt.show()

predictions = predict(X,w)
pred_sep = separate(predictions)
# print(pred_sep)
acc = np.mean((pred_sep==y))*100
print(acc)