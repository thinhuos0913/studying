import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('digits.png', 0)
imgtest = cv2.imread('n8.png', 0)
cells=[np.hsplit(row,100) for row in np.vsplit(img,50)] # split small image from digits.png
x=np.array(cells)
x2=np.array(imgtest)
#print(x2)
train=x[:,:50].reshape(-1,400).astype(np.float32)
test=x2.reshape(-1,400).astype(np.float32)
k=np.arange(10)
train_labels=np.repeat(k,250)[:,np.newaxis]
knn=cv2.ml.KNearest_create()
knn.train(train,0,train_labels)
result=knn.findNearest(test,5)
#print(result)
temp, predict, neighbours, distances = knn.findNearest(test,5)
print('Predict: {}'.format(predict))
print('Neighbours: {} '.format(neighbours))
print('Distances: {}'.format(distances))