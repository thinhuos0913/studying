import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('digits.png', 0)
#cv2.imshow('Handwriting', img)
cells=[np.hsplit(row,100) for row in np.vsplit(img,50)]
# print(cells[15][30])
# cv2.imwrite('digit_split.jpg',cells[15][30])
x=np.array(cells)
# #print(x.shape)
# # xx=x[0,0].reshape(-1,400)
# # print(xx.shape)
train=x[:,:50].reshape(-1,400).astype(np.float32)
test=x[:,50:100].reshape(-1,400).astype(np.float32)
# print('train:', train.shape)
# print('test:', test.shape)
# Seal labels for train data
k=np.arange(10)
# print(k)
train_labels=np.repeat(k,250)[:,np.newaxis]
#print(train_labels)
# Recognize:
knn=cv2.ml.KNearest_create()
knn.train(train,0,train_labels)
result=knn.findNearest(test,5)
print(result)
temp, predict, neighbours, distances = knn.findNearest(test,5)
print('Predict: {}'.format(predict))
print('Neighbours: {} '.format(neighbours))
print('Distances: {}'.format(distances))
print(predict[505])
print(train_labels[505])
print(len(predict))
# cv2.waitKey(0)
# cv2.destroyAllWindows()