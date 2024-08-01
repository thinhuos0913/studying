import numpy as np
import cv2

img=cv2.imread('edited.jpg',1)
# Access to every img's pixel points
px=img[100][100]
print(px)
for i in range(100):
	# img[i][i]=[1,1,1]
	# img[i+2][i+2]=[1,1,1]
	for j in range(100):
		if img[i,j,0]>100:
			img[i,j]=1
cv2.imwrite('edited2.jpg', img)
