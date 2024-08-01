import cv2
img=cv2.imread('im.png',0)
img2=cv2.Canny(img,100,255)
cv2.imshow('image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()