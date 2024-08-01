import cv2
import numpy as np
# Check Image Attributes:
# img1 = cv2.imread('01.jpg',1)
# img2 = cv2.imread('02.jpg',1)
# img1=img1[100:600,100:700]
# img2=img2[100:600,100:700]

#cv2.imshow('image',img1)
#cv2.imshow('image',img2)
# img3=cv2.add(img1,img2) # combine 2 images 1&2, with the same size, the same channels (r,g,b)
# cv2.imshow('image',img3)


img1 = cv2.imread('frame0.jpg',1)
img2 = cv2.imread('frame36.jpg',1)
img3 = cv2.imread('frame100.jpg',1)
img4=cv2.add(img1,img2,img3)
cv2.imwrite('new_test.jpg', img4)
cv2.imshow('image',img4)

cv2.waitKey()
cv2.destroyAllWindows()