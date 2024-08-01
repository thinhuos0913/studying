import cv2
import numpy as np
import matplotlib.pyplot as plt

train_data=np.random.randint(0,100,(25,2)).astype(np.float32)
print(train_data)

result=np.random.randint(0,2,(25,1)).astype(np.float32)
print(result)
red=train_data[result.ravel()==1]
blue=train_data[result.ravel()==0]
new=np.random.randint(0,100,(1,2)).astype(np.float32)
# print(red)
# print(blue)
# print(new)
plt.scatter(red[:,0],red[0:,1],100,'r','s')
plt.scatter(blue[:,0],blue[0:,1],100,'b','^')
plt.scatter(new[:,0],new[:,1],100,'g','o')
knn=cv2.ml.KNearest_create()
knn.train(train_data,0,result)
temp, predict, neighbours, distances = knn.findNearest(new,3)
print('Predict: {}'.format(predict))
print('Neighbours: {} '.format(neighbours))
print('Distances: {}'.format(distances))
plt.show()