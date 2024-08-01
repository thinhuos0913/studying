import numpy as np
import cv2
img=cv2.imread('82014.png',1)
cv2.line(img, (0,0), (400,300), (255,0,0), 5)
cv2.imwrite('edited.jpg', img)
cv2.imshow('Edited',img)
cv2.waitKey(0)
cv2.destroyAllWindows()