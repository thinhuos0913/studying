# importing libraries 
# from sklearn.datasets.samples_generator => deprecated
  
# matplotlib for ploting
from matplotlib import pyplot as plt  
from matplotlib import style 

# Creating Test DataSets using sklearn.datasets.make_blobs 
from sklearn.datasets import make_blobs 
from matplotlib import pyplot as plt  
from matplotlib import style 

style.use("fivethirtyeight") 
  
X, y = make_blobs(n_samples = 100, centers = 3,  
               cluster_std = 1, n_features = 2) 
  
plt.scatter(X[:, 0], X[:, 1], s = 40, color = 'g') 
plt.xlabel("X") 
plt.ylabel("Y") 
  
plt.show() 
plt.clf() 

# Creating Test DataSets using sklearn.datasets.make_moon 
# from sklearn.datasets import make_moons 
# from matplotlib import pyplot as plt  
# from matplotlib import style 
  
# X, y = make_moons(n_samples = 1000, noise = 0.1) 
# plt.scatter(X[:, 0], X[:, 1], s = 40, color ='g') 
# plt.xlabel("X") 
# plt.ylabel("Y") 
  
# plt.show() 
# plt.clf()

# Creating Test DataSets using sklearn.datasets.make_circles 
# from sklearn.datasets import make_circles 
# from matplotlib import pyplot as plt  
# from matplotlib import style 
  
# style.use("fivethirtyeight") 
  
# X, y = make_circles(n_samples = 100, noise = 0.02) 
# plt.scatter(X[:, 0], X[:, 1], s = 40, color ='g') 
# plt.xlabel("X") 
# plt.ylabel("Y") 
  
# plt.show() 
# plt.clf() 