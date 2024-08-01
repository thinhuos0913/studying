import cv2
import numpy as np
img=np.uint8([[[255,0,0]]])
# cv2.imshow('img',img)
# convert bgr to hsv
hsv_img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(hsv_img)
# cv2.imshow('hsv_img',hsv_img)
img2=cv2.imread('edited.jpg',1)
hsv_img2=cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
# cv2.imshow('This is image',hsv_img2)

min_color=np.array([120,255,232])
max_color=np.array([122,255,232])
mask=cv2.inRange(hsv_img2,min_color,max_color)
final=cv2.bitwise_and(img2,img2,mask=mask)
# cv2.imshow('This is image',mask)
cv2.imshow('This is image',final)
cv2.waitKey(0)
cv2.destroyAllWindows()