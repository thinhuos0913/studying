import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.gca(projection='3d')
np.random.seed(2)

data=pd.read_csv('./data/data_classification.csv',header=None)

X=np.array(data)
y=np.copy(X[:,2])
X[:,2]=X[:,1]
X[:,1]=X[:,0]
X[:,0]=1
#print(X.shape)
#print(y.shape)
X=X.T
#print(X.shape)

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def separate(p):
    if p>=0.5: 
        return 1
    else:
        return 0

def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init]    
    it = 0
    N = X.shape[1] # N=100
    d = X.shape[0] # d=3
    count = 0
    check_w_after = 100
    while count < max_count:
        # mix data 
        mix_id = np.random.permutation(N) 
        for i in mix_id:
            xi = X[:,i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi)) #w[-1] = bias (w0)
            w_new = w[-1] + eta*(yi - zi)*xi
            count += 1
            # stopping criteria
            if count%check_w_after == 0:                
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w

eta = 0.05 
d = X.shape[0]
w_init = np.random.randn(d, 1)
w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w_init)
print(w[-1])
#print(sigmoid(np.dot(w[-1].T, X)))
x=np.array([1, 3.150954837, 10.899420416])
s=np.dot(x,w[-1])
print(s)
print(sigmoid(s))
print(separate(sigmoid(s)))